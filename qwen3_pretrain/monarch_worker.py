"""Monarch worker: what each worker pod runs instead of torchrun.

It listens and waits for the controller (controller.py) to tell it what to do:
the controller spawns this pod's training processes and its PodHealth actor.
Failure decisions are the controller's: it marks the pod failed through
PodHealth, the liveness probe sees the mark, and Kubernetes restarts the pod.

The only thing this process decides is a clean exit (0) once training has
completed, so the Job counts the pod as done.

Same listening entrypoint as GRPO's grpo_monarch_worker.py.
"""

from __future__ import annotations

import os
import signal
import threading
import time

from monarch.actor import enable_transport, run_worker_loop_forever

PORT = int(os.environ.get("MONARCH_WORKER_PORT", "26600"))
COMPLETED_MARK = "/run/torchft-neuron-m1/qwen3-training-completed"


def _log(message: str) -> None:
    print(f"[worker {time.strftime('%H:%M:%S')}] {message}", flush=True)


def _exit_when_completed() -> None:
    while not os.path.exists(COMPLETED_MARK):
        time.sleep(5)
    # Let the other ranks return from training before the pod goes.
    time.sleep(30)
    _log("training completed; exiting 0")
    os._exit(0)


def _exit_on_sigterm(signum: int, frame: object) -> None:
    # This process is PID 1 (the container command execs it), and PID 1 ignores
    # SIGTERM unless it handles it: Kubernetes then waited the whole grace period
    # (90 s) before killing the pod. Exiting at once lets the restart start in
    # seconds; the training processes go with the container.
    _log("SIGTERM from Kubernetes; exiting")
    os._exit(143)


def main() -> None:
    pod_ip = os.environ["POD_IP"]
    signal.signal(signal.SIGTERM, _exit_on_sigterm)
    threading.Thread(target=_exit_when_completed, daemon=True).start()
    # Callbacks from the controller arrive on this address.
    enable_transport(f"tcp://{pod_ip}:0")
    _log(f"monarch worker listening on tcp://{pod_ip}:{PORT}")
    run_worker_loop_forever(ca="trust_all_connections", address=f"tcp://{pod_ip}:{PORT}")


if __name__ == "__main__":
    main()
