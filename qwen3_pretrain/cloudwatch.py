"""Training metrics to CloudWatch, sent by each replica's local rank 0, and a
compact METRICS line in the pod log (`kubectl logs <pod> | grep METRICS`).

Two hooks, both on rank 0 of every replica (pod):
- TorchTitan's MetricsProcessor.log (every logged step): loss, grad norm,
  tokens/s, MFU, step time, device memory -- the numbers of the "step:" line;
- the trainer's train_step (every step): the torchft quorum size, and one
  StepsCompleted count (its per-minute Sum shows a replica stepping or not).

A background thread batches and sends them once a minute with PutMetricData,
using the node's IAM role (as grpo-neuron-deploy's metrics pod does). Nothing
here may stop training: a CloudWatch error is logged once and the batch dropped.

The METRICS line is always printed (rank 0 only). Sending to CloudWatch is off
unless CW_NAMESPACE is set. Dimensions: RunId (CW_RUN_ID) and Replica.
The logged loss is torchft's sum over the participating replicas, so Loss is
that sum divided by the quorum size (LossSum keeps the raw value).
"""

from __future__ import annotations

import os
import queue
import threading
import time
from typing import Any

from torchtitan.tools.logging import logger

NAMESPACE = os.environ.get("CW_NAMESPACE", "")
RUN_ID = os.environ.get("CW_RUN_ID", "qwen3-pretrain")
FLUSH_EVERY_S = float(os.environ.get("CW_FLUSH_EVERY_S", "60"))
RANK0 = os.environ.get("LOCAL_RANK", "0") == "0"
ENABLED = bool(NAMESPACE) and RANK0

_queue: queue.Queue = queue.Queue()
_participants = 0
_warned: set[str] = set()

# TorchTitan metric key -> (CloudWatch name, unit)
_STEP_METRICS = {
    "grad_norm": ("GradNorm", "None"),
    "throughput(tps)": ("TokensPerSecond", "Count/Second"),
    "mfu(%)": ("MFU", "Percent"),
    "time_metrics/end_to_end(s)": ("StepTime", "Seconds"),
    "memory/max_active(%)": ("DeviceMemoryActive", "Percent"),
}


def _warn_once(key: str, message: str) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning(f"cloudwatch: {message}")


def _replica() -> str:
    return os.environ.get("TORCHFT_NEURON_M1_LOGICAL_SLOT", os.environ.get("JOB_COMPLETION_INDEX", "?"))


def record(name: str, value: float, unit: str) -> None:
    if ENABLED:
        _queue.put((name, float(value), unit, time.time()))


def _publish_forever() -> None:
    try:
        import boto3  # noqa: PLC0415
    except ImportError as error:
        _warn_once("import", f"boto3 not importable ({error}); metrics are off")
        return
    # region_name explicitly: botocore reads AWS_DEFAULT_REGION, not AWS_REGION.
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    client = boto3.client("cloudwatch", region_name=region)
    dims = [{"Name": "RunId", "Value": RUN_ID}, {"Name": "Replica", "Value": _replica()}]
    logger.info(f"cloudwatch: publishing to {NAMESPACE} every {FLUSH_EVERY_S:.0f} s, dims {dims}")
    while True:
        time.sleep(FLUSH_EVERY_S)
        batch = []
        while True:
            try:
                name, value, unit, ts = _queue.get_nowait()
            except queue.Empty:
                break
            batch.append(
                {
                    "MetricName": name,
                    "Value": value,
                    "Unit": unit,
                    "Timestamp": ts,
                    "Dimensions": dims,
                }
            )
        for i in range(0, len(batch), 500):
            try:
                client.put_metric_data(Namespace=NAMESPACE, MetricData=batch[i : i + 500])
            except Exception as error:
                _warn_once(type(error).__name__, f"put_metric_data failed, dropping: {error}")


def _summary(metrics: dict[str, Any], step: int, loss: float | None) -> str:
    def fmt(key: str, spec: str, suffix: str = "") -> str:
        value = metrics.get(key)
        return "n/a" if value is None else f"{value:{spec}}{suffix}"

    return (
        f"METRICS replica={_replica()} step={step} "
        f"loss={'n/a' if loss is None else f'{loss:.4f}'} "
        f"grad_norm={fmt('grad_norm', '.3f')} tps={fmt('throughput(tps)', '.0f')} "
        f"mfu={fmt('mfu(%)', '.2f', '%')} step_time={fmt('time_metrics/end_to_end(s)', '.2f', 's')} "
        f"participants={_participants or 'n/a'}"
    )


def _on_step_metrics(metrics: dict[str, Any], step: int) -> None:
    loss_sum = metrics.get("loss_metrics/global_avg_loss")
    loss = loss_sum / _participants if loss_sum is not None and _participants else None
    print(_summary(metrics, step, loss), flush=True)
    if loss_sum is not None:
        record("LossSum", loss_sum, "None")
    if loss is not None:
        record("Loss", loss, "None")
    for key, (name, unit) in _STEP_METRICS.items():
        if metrics.get(key) is not None:
            record(name, metrics[key], unit)
    record("Step", step, "Count")


def install() -> None:
    """Hook TorchTitan's step metrics and the trainer's train_step (rank 0); start
    the CloudWatch sender if CW_NAMESPACE is set."""
    if not RANK0:
        return
    try:
        from torchtitan.components.metrics import MetricsProcessor  # noqa: PLC0415
    except ImportError as error:
        _warn_once("hook", f"no TorchTitan MetricsProcessor ({error}); metrics are off")
        return
    from examples.kubernetes.neuron_m1.trainer import NeuronFaultTolerantTrainer  # noqa: PLC0415

    original_log = MetricsProcessor.log

    def log(self, *args, **kwargs):
        # The metrics dict reaches self.logger.log; tap it there, once per instance.
        if not getattr(self, "_cloudwatch_tapped", False):
            sink = self.logger.log

            def tapped(metrics, step, *a, **kw):
                try:
                    _on_step_metrics(metrics, step)
                except Exception as error:
                    _warn_once("step", f"step metrics skipped: {error}")
                return sink(metrics, step, *a, **kw)

            self.logger.log = tapped
            self._cloudwatch_tapped = True
        return original_log(self, *args, **kwargs)

    MetricsProcessor.log = log

    original_step = NeuronFaultTolerantTrainer.train_step

    def train_step(self, *args, **kwargs):
        global _participants
        result = original_step(self, *args, **kwargs)
        try:
            _participants = int(self.ft_manager.manager.num_participants())
            record("Participants", _participants, "Count")
        except Exception as error:
            _warn_once("participants", f"quorum size unavailable: {error}")
        record("StepsCompleted", 1, "Count")
        return result

    NeuronFaultTolerantTrainer.train_step = train_step
    if ENABLED:
        threading.Thread(target=_publish_forever, name="cloudwatch", daemon=True).start()
