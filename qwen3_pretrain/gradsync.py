"""GRADSYNC log lines: what each rank's cross-replica gradient average covered.

Every GRADSYNC_LOG_EVERY steps (and step 1), after the optimizer step, each rank
logs its torchft replica id, the size of its local gradient slice, and two
fingerprints: the sum of that slice's gradients and of its weights. The same rank
in each pod averages the same slice, so after torchft's all-reduce their
fingerprints must be identical; a difference means the replicas drifted apart.
Only local rank 0 of each pod is printed (torchrun --local-ranks-filter 0).
"""

from __future__ import annotations

import json
import os
import time

import torch
import torch.distributed as dist
from examples.kubernetes.neuron_m1.trainer import NeuronFaultTolerantTrainer as Trainer
from torchtitan.tools.logging import logger

_EVERY = int(os.environ.get("GRADSYNC_LOG_EVERY", "100"))


def _local(t: torch.Tensor) -> torch.Tensor:
    return t.to_local() if hasattr(t, "to_local") else t


def _log(trainer: Trainer) -> None:
    ft = trainer.config.fault_tolerance

    params = [p for m in trainer.model_parts for p in m.parameters()]
    grads = [_local(p.grad) for p in params if p.grad is not None]
    numel = sum(g.numel() for g in grads)
    nbytes = sum(g.numel() * g.element_size() for g in grads)
    dtype = grads[0].dtype if grads else None
    grad_fp = torch.stack([g.float().sum() for g in grads]).sum().item()
    weight_fp = torch.stack([_local(p).float().sum() for p in params]).sum().item()

    logger.info(
        f"GRADSYNC step {trainer.step}: replica {ft.replica_id} of {ft.group_size}, "
        f"rank {dist.get_rank()}, slice {numel / 1e6:.1f}M values {dtype} "
        f"({nbytes / 1e9:.2f} GB), grad fingerprint {grad_fp:.6e}, "
        f"weight fingerprint {weight_fp:.6e}"
    )


# Readiness heartbeat (readiness.py): local rank 0 records each finished step.
HEARTBEAT = "/run/torchft-neuron-m1/qwen3-heartbeat.json"
_LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))


def _beat(step: int) -> None:
    tmp = f"{HEARTBEAT}.tmp"
    with open(tmp, "w") as f:
        json.dump({"step": step, "time": time.time()}, f)
    os.replace(tmp, HEARTBEAT)


def install() -> None:
    """Wrap Trainer.train_step: a GRADSYNC line every logged step, and the
    readiness heartbeat after every step."""
    original = Trainer.train_step

    def train_step(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if self.step == 1 or self.step % _EVERY == 0:
            _log(self)
        if _LOCAL_RANK == 0:
            _beat(self.step)
        return result

    Trainer.train_step = train_step
