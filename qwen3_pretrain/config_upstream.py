"""Qwen3-14B on C4 as torchft replicas on Trainium, with upstream torchft.

TorchTitan's own torchft integration (``experiments/torchft``: FaultTolerantTrainer,
TorchFTManager, TorchFTOptimizersContainer, TorchFTCheckpointManager) on upstream
torchft, with torchstore's NIXL process group and healing transport (ft_nixl.py).
Neuron-only additions here: TorchTitan's attention routed to the pinned Neuron
callables (neuron_attention.py) and the chunked loss bound to lm_head. Each pod is
one 16-rank world (TP4 x FSDP4) and one torchft replica.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace
from typing import Any

from qwen3_pretrain import ft_nixl
from qwen3_pretrain.neuron_attention import install_neuron_attention
from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.experiments.torchft.checkpoint import TorchFTCheckpointManager
from torchtitan.experiments.torchft.config.job_config import FaultTolerance
from torchtitan.experiments.torchft.optimizer import TorchFTOptimizersContainer
from torchtitan.experiments.torchft.trainer import FaultTolerantTrainer
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.models.qwen3.config_registry import qwen3_14b

_SEED = 20260729
_REPLICAS = int(os.environ.get("NNODES", "4"))
_DATASET = os.environ.get("QWEN3_DATASET", "c4")
_TOKENIZER = (
    "./tests/assets/tokenizer"
    if _DATASET == "c4_test"
    else os.environ.get("QWEN3_TOKENIZER_DIR", "/tmp/hf/Qwen3-14B")
)


class NeuronFTTrainer(FaultTolerantTrainer):
    """TorchTitan's FaultTolerantTrainer plus the two Neuron bindings."""

    @dataclass(kw_only=True, slots=True)
    class Config(FaultTolerantTrainer.Config):
        pass

    def __init__(self, config: Config) -> None:
        install_neuron_attention()
        ft_nixl.install()
        super().__init__(config)
        if isinstance(self.loss_fn, ChunkedLossWrapper) and len(self.model_parts) == 1:
            model = self.model_parts[0]
            lm_head = getattr(model, "lm_head", None)
            if lm_head is not None:
                self.loss_fn.set_lm_head(lm_head)
                model._skip_lm_head = True


    def init_distributed(self) -> ParallelDims:
        """TorchTitan's FT init_distributed without its ``ranks=`` list.

        Upstream derives the global ranks as ``replica_id * group_size + i``, with
        ``group_size`` the replica count, so a 16-rank replica's process group got 4
        global ranks it does not have and the Neuron collectives failed ("Invalid
        NEFF, instruction, or input" in the first allgather). Each pod is its own
        world here; torchft joins the replicas.
        """
        dist_utils.init_distributed(
            self.config.comm,
            enable_cpu_backend=self.config.training.enable_cpu_offload,
            base_folder=self.config.dump_folder,
        )
        self.ft_manager = self.config.fault_tolerance.build()
        return ParallelDims.from_config(self.config.parallelism, int(os.environ["WORLD_SIZE"]))


def _values(config: Any) -> dict[str, Any]:
    return {item.name: getattr(config, item.name) for item in fields(config)}


def qwen3_14b_c4() -> NeuronFTTrainer.Config:
    base = qwen3_14b()
    values = _values(base)
    values.update(
        {
            "hf_assets_path": _TOKENIZER,
            "dump_folder": os.environ.get("QWEN3_DUMP_FOLDER", "/tmp/torchtitan-outputs"),
            "dataloader": HuggingFaceTextDataLoader.Config(
                dataset=_DATASET, infinite=True, num_workers=0
            ),
            "optimizer": TorchFTOptimizersContainer.Config(**_values(base.optimizer)),
            "training": replace(
                base.training, local_batch_size=1, global_batch_size=-1, seq_len=4096
            ),
            "parallelism": replace(
                base.parallelism,
                data_parallel_replicate_degree=1,
                data_parallel_shard_degree=4,
                tensor_parallel_degree=4,
                pipeline_parallel_degree=1,
                context_parallel_degree=1,
            ),
            # TorchTitan gives the torchft Manager its state (model, optimizer, LR,
            # train state: what a healing replica receives) only when checkpointing
            # is on. On, but never due: no disk saves; no per-replica dataloader ones.
            "checkpoint": TorchFTCheckpointManager.Config(
                **{
                    **_values(base.checkpoint),
                    "enable": True,
                    "interval": 10**9,
                    "folder": "checkpoint",
                },
                enable_ft_dataloader_checkpoints=False,
            ),
            "activation_checkpoint": FullAC.Config(),
            "compile": replace(
                base.compile, enable=True, components=["model", "loss"], backend="neuron"
            ),
            "debug": replace(base.debug, seed=_SEED),
            "metrics": replace(
                base.metrics, log_freq=int(os.environ.get("QWEN3_LOG_FREQ", "1"))
            ),
            "fault_tolerance": FaultTolerance(
                enable=True,
                process_group="gloo",  # ft_nixl swaps in ProcessGroupNixl
                process_group_timeout_ms=int(os.environ.get("TORCHFT_TIMEOUT_SEC", "120")) * 1000,
                replica_id=int(os.environ.get("QWEN3_REPLICA_ID", "0")),
                group_size=_REPLICAS,
                min_replica_size=_REPLICAS - 1,
            ),
        }
    )
    return NeuronFTTrainer.Config(**values)
