"""Qwen3-14B pretraining from scratch on Trainium (TorchTitan + the demo's Neuron torchft)."""

import os

# Must run before any demo module is imported; see m1_shape.py. Skipped:
# - in the Monarch controller (QWEN3_CONTROLLER=1), which trains nothing;
# - in Monarch worker processes (QWEN3_DEFER_M1_SHAPE=1), where it would import
#   TorchTitan before torch-neuronx is loaded, and TorchTitan would then pick "cuda"
#   as its device type; replica_actor.py imports it after torch-neuronx instead.
if os.environ.get("QWEN3_CONTROLLER") != "1" and os.environ.get("QWEN3_DEFER_M1_SHAPE") != "1":
    from qwen3_pretrain import m1_shape as _m1_shape  # noqa: F401
