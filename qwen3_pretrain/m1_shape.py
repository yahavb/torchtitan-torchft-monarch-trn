"""Resize the torchft demo's hardcoded cohort shape to this job's.

The demo's Neuron torchft code (examples.kubernetes.neuron_m1 in the image) is
written for 4 hosts x 64 ranks: manifest.py defines ACTIVE_HOSTS and LOCAL_RANKS,
and trainer.py, frontier.py and others copy them at import time. This job runs
2 pods x 16 ranks. Imported first (from qwen3_pretrain/__init__.py): it loads
every demo module that reads the shape, then sets the values in each of them.
"""

from __future__ import annotations

import importlib
import os
import sys

# Pods (torchft replicas) and ranks per pod; job.yaml sets the matching env.
REPLICAS = int(os.environ.get("NNODES", "2"))
LOCAL_RANKS = 16

_PACKAGE = "examples.kubernetes.neuron_m1"
# Demo modules that read the shape at run time (frontier.py validates the world
# size; trainer.py the replica and rank counts).
_MODULES = ("manifest", "frontier", "trainer")


def apply() -> None:
    for module in _MODULES:
        importlib.import_module(f"{_PACKAGE}.{module}")
    for name, module in list(sys.modules.items()):
        if module is None or not (name == _PACKAGE or name.startswith(_PACKAGE + ".")):
            continue
        if hasattr(module, "ACTIVE_HOSTS"):
            module.ACTIVE_HOSTS = REPLICAS
        if hasattr(module, "LOCAL_RANKS"):
            module.LOCAL_RANKS = LOCAL_RANKS


apply()
