"""Qwen3-14B pretraining from scratch on Trainium (TorchTitan + the demo's Neuron torchft)."""

import os

# Must run before any demo module is imported; see m1_shape.py. Skipped in the
# Monarch controller (QWEN3_CONTROLLER=1), which trains nothing and must not
# load torch or the demo trainer.
if os.environ.get("QWEN3_CONTROLLER") != "1":
    from qwen3_pretrain import m1_shape as _m1_shape  # noqa: F401
