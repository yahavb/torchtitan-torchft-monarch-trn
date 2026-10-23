"""Qwen3-14B pretrained from scratch on C4, as torchft replicas on Trainium.

This is the torchft device-taint demo's job (torchft-neuron
examples/kubernetes/neuron_m1/config_registry.py, Image 14), built on the demo's
own Neuron torchft classes from the image: NeuronFaultTolerantTrainer,
NeuronTorchFTManager (ProcessGroupNeuron, cross-pod RDMA over EFA), and its
optimizer and checkpoint subclasses. Each pod is one 16-rank torchrun world and
one torchft replica.

The demo hardcodes its shape (4 hosts x 64 ranks); m1_shape.py resizes it to
2 pods x 16 ranks, and its TP4 x FSDP16 layout check is replaced below.
"""

from __future__ import annotations

import inspect
import os
from dataclasses import fields, replace
from typing import Any

# m1_shape first: it must resize the demo before the demo modules load.
from qwen3_pretrain import m1_shape  # isort: skip

import examples.kubernetes.neuron_m1.neuron_attention as _m1_attention
import examples.kubernetes.neuron_m1.trainer as _m1_trainer
from examples.kubernetes.neuron_m1.checkpoint import NeuronCheckpointManager
from examples.kubernetes.neuron_m1.frontier import TransitionError
from examples.kubernetes.neuron_m1.manager import NeuronTorchFTManager
from examples.kubernetes.neuron_m1.optimizer import NeuronTorchFTOptimizersContainer
from examples.kubernetes.neuron_m1.trainer import NeuronFaultTolerantTrainer
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.models.qwen3.config_registry import qwen3_14b
from torchtitan.tools.logging import logger

# Same seed as the demo, so a loss curve here is comparable to one replica's there.
_SEED = 20260729

_REPLICAS = m1_shape.REPLICAS
_LOCAL_RANKS = m1_shape.LOCAL_RANKS

# The demo trainer also insists on its TP4 x FSDP16 layout; ours is TP4 x FSDP4.
_PARALLELISM = {
    "data_parallel_replicate_degree": 1,
    "data_parallel_shard_degree": 4,
    "tensor_parallel_degree": 4,
    "pipeline_parallel_degree": 1,
    "context_parallel_degree": 1,
}


def _validate_parallelism_config(config: Any) -> None:
    if any(getattr(config.parallelism, k, None) != v for k, v in _PARALLELISM.items()):
        raise _m1_trainer.TrainerConfigurationError(
            f"this job requires parallelism {_PARALLELISM}"
        )


NeuronFaultTolerantTrainer._validate_parallelism_config = staticmethod(
    _validate_parallelism_config
)

# The demo trainer writes a status record per rank (frontier.py) for its own
# readiness probe, which this job does not use; training never reads it. Its
# transition rules reject a rejoiner's 1-step catch-up heal ("a membership change
# cannot advance the committed frontier") and that crashed the pod. Log instead.
_original_publish_frontier = NeuronFaultTolerantTrainer._publish_frontier


def _publish_frontier(self: Any, phase: str) -> None:
    try:
        _original_publish_frontier(self, phase)
    except TransitionError as error:
        logger.warning(f"frontier record not updated (status only): {error}")


NeuronFaultTolerantTrainer._publish_frontier = _publish_frontier

# The demo's Neuron attention adapter checks that torch.compile, flex_attention and
# create_block_mask have EXACTLY the torch 2.13 parameters; torch 2.14 differs
# (torch.compile lost shapes_spec), which fails the check though the adapter does not
# use those parameters. Keep a guard against real drift: for torch.compile, the
# parameters the adapter passes must exist; for the others, every expected
# parameter must exist with the same kind. Added parameters are allowed and logged.
_original_validate_parameters = _m1_attention._validate_parameters
# Parameters the adapter actually passes, per checked callable (neuron_attention.py's
# install_neuron_attention); for these only the used ones must exist.
_USED_PARAMETERS = {"torch.compile": {"model", "backend", "options"}}


def _validate_parameters(target: Any, *, label: str, expected: tuple) -> None:
    try:
        actual = tuple(
            (p.name, p.kind) for p in inspect.signature(target).parameters.values()
        )
    except (TypeError, ValueError):
        return _original_validate_parameters(target, label=label, expected=expected)
    names = {name for name, _ in actual}
    if label in _USED_PARAMETERS:
        # Only what the adapter passes has to exist (it calls
        # torch.compile(fn, backend=..., options=...)); log other differences.
        missing = sorted(_USED_PARAMETERS[label] - names)
        removed = [name for name, _ in expected if name not in names]
        if removed and not missing:
            logger.info(f"neuron attention: {label} no longer has {removed}; unused, allowed")
    else:
        missing = [p for p in expected if p not in actual]
    if missing:
        raise _m1_attention.NeuronAttentionContractError(
            f"{label} signature drifted: missing or changed {missing}"
        )
    added = [name for name, _ in actual if name not in {n for n, _ in expected}]
    if added:
        logger.info(f"neuron attention: {label} has new parameters {added}; allowed")


_m1_attention._validate_parameters = _validate_parameters

# A rejoiner replays the dataloader up to the cohort's committed step, so the
# replay bound must cover the whole run (the demo's Image 14 note).
_MAX_STEPS = 10000


# The dataset: "c4" (allenai/c4 en, streamed from Hugging Face; default) or
# "c4_test" (TorchTitan's 2000-document sample in the image: quick tests only; the
# model memorizes it within a few hundred steps).
_DATASET = os.environ.get("QWEN3_DATASET", "c4")
# C4 uses the real Qwen3 tokenizer, fetched at pod start by fetch_tokenizer.sh;
# c4_test keeps TorchTitan's test tokenizer, resolved against the image's WORKDIR.
_TOKENIZER = (
    "./tests/assets/tokenizer"
    if _DATASET == "c4_test"
    else os.environ.get("QWEN3_TOKENIZER_DIR", "/tmp/hf/Qwen3-14B")
)


def _config_values(config: Any) -> dict[str, Any]:
    return {item.name: getattr(config, item.name) for item in fields(config)}


def qwen3_14b_c4() -> NeuronFaultTolerantTrainer.Config:
    """Qwen3 14B, random init, C4, torchft replicas x 16 logical NeuronCores.

    Each pod holds one l-lnc2-trn-efa claim: 4 devices x 4 logical cores = 16 ranks,
    TP4 x FSDP4 inside the pod. Each step the lighthouse names the live replicas
    and torchft all-reduces gradients among them; one may die and rejoin.
    """

    base = qwen3_14b()
    values = _config_values(base)
    values.update(
        {
            "hf_assets_path": _TOKENIZER,
            # TorchTitan's outputs (structured logs, ...) default to ./outputs in its
            # checkout, which is read-only here (the deps volume).
            "dump_folder": os.environ.get("QWEN3_DUMP_FOLDER", "/tmp/torchtitan-outputs"),
            "dataloader": HuggingFaceTextDataLoader.Config(
                dataset=_DATASET,
                infinite=True,
                num_workers=0,
            ),
            "optimizer": NeuronTorchFTOptimizersContainer.Config(
                **_config_values(base.optimizer)
            ),
            # One sequence per rank x 4 FSDP shards x 2 replicas = 8 per step.
            "training": replace(
                base.training,
                local_batch_size=1,
                global_batch_size=-1,
                seq_len=4096,
            ),
            "max_cursor_replay_steps": _MAX_STEPS,
            # Replication across pods is torchft's job: in-world replicate is 1.
            "parallelism": replace(
                base.parallelism,
                data_parallel_replicate_degree=1,
                data_parallel_shard_degree=4,
                tensor_parallel_degree=4,
                pipeline_parallel_degree=1,
                context_parallel_degree=1,
            ),
            "checkpoint": NeuronCheckpointManager.Config(
                **_config_values(base.checkpoint)
            ),
            "activation_checkpoint": FullAC.Config(),
            "compile": replace(
                base.compile,
                enable=True,
                components=["model", "loss"],
                backend="neuron",
            ),
            "debug": replace(base.debug, seed=_SEED),
            # Keep training with 1 of 2 replicas (lighthouse --min_replicas 1).
            "fault_tolerance": NeuronTorchFTManager.Config(
                enable=True,
                group_size=_REPLICAS,
                min_replica_size=_REPLICAS - 1,
                transport_stride=_LOCAL_RANKS,
            ),
        }
    )
    return NeuronFaultTolerantTrainer.Config(**values)
