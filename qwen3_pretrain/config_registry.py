"""TorchTitan config registry for this job: upstream torchft (default) or the fork.

QWEN3_TORCHFT=upstream: upstream torchft + TorchTitan's torchft trainer, NIXL from
torchstore (config_upstream.py). QWEN3_TORCHFT=fork: the torchft-neuron fork's
Neuron trainer and native RDMA (config_fork.py), kept as the native baseline.
"""

import os

if os.environ.get("QWEN3_TORCHFT", "upstream") == "fork":
    from qwen3_pretrain.config_fork import qwen3_14b_c4  # noqa: F401
else:
    from qwen3_pretrain.config_upstream import qwen3_14b_c4  # noqa: F401
