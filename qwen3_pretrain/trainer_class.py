"""The trainer class in use, for the step hooks (gradsync, cloudwatch)."""

import os


def trainer_class():
    if os.environ.get("QWEN3_TORCHFT", "upstream") == "fork":
        from examples.kubernetes.neuron_m1.trainer import NeuronFaultTolerantTrainer

        return NeuronFaultTolerantTrainer
    from qwen3_pretrain.config_upstream import NeuronFTTrainer

    return NeuronFTTrainer
