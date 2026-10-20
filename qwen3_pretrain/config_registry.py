"""Qwen3-14B pretrained from scratch on c4_test, two trn3-dev1 hosts, with torchft.

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

from dataclasses import fields, replace
from typing import Any

# m1_shape first: it must resize the demo before the demo modules load.
from qwen3_pretrain import m1_shape  # isort: skip

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

# A rejoiner replays the dataloader up to the cohort's committed step, so the
# replay bound must cover the whole run (the demo's Image 14 note).
_MAX_STEPS = 10000


def _config_values(config: Any) -> dict[str, Any]:
    return {item.name: getattr(config, item.name) for item in fields(config)}


def qwen3_14b_c4_test() -> NeuronFaultTolerantTrainer.Config:
    """Qwen3 14B, random init, c4_test, 2 torchft replicas x 16 logical NeuronCores.

    Each pod holds one l-lnc2-trn-efa claim: 4 devices x 4 logical cores = 16 ranks,
    TP4 x FSDP4 inside the pod. Each step the lighthouse names the live replicas
    and torchft all-reduces gradients among them; one may die and rejoin.
    """

    base = qwen3_14b()
    values = _config_values(base)
    values.update(
        {
            # TorchTitan's test tokenizer, resolved against the image's WORKDIR
            # (/opt/torchtitan). No HF download, no token.
            "hf_assets_path": "./tests/assets/tokenizer",
            # torchtitan/tests/assets/c4_test: 2000 English C4 documents, ~4.7 MB.
            "dataloader": HuggingFaceTextDataLoader.Config(
                dataset="c4_test",
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
