"""Entrypoint, one process per rank under torchrun.

The stock TorchTitan main. The demo's NeuronFaultTolerantTrainer (see
config_registry) installs the Neuron attention adapter itself when it is built.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from qwen3_pretrain import cloudwatch, gradsync
from torchtitan.config import ConfigManager
from torchtitan.observability import structured_logger as sl
from torchtitan.tools.logging import init_logger

_DEFAULT_ARGS = ["--module", "qwen3_pretrain", "--config", "qwen3_14b_c4"]


def main(args: Sequence[str] | None = None) -> None:
    init_logger()
    argv = list(sys.argv[1:] if args is None else args)
    # Extra CLI flags (e.g. --training.steps 200) override the registry config.
    config = ConfigManager().parse_args(_DEFAULT_ARGS + argv)
    sl.init_structured_logger(
        source="training",
        output_dir=config.dump_folder,
        enable=config.debug.enable_structured_logging,
    )

    gradsync.install()
    cloudwatch.install()

    trainer = config.build()
    try:
        trainer.train()
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
