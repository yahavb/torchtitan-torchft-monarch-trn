"""The Monarch actors the controller runs in each worker pod.

ReplicaTrainer -- one per rank, on the pod's 16 training processes:
1. prepare (rank 0 only): the per-launch identity the demo trainer checks
   (launch.json, as the demo launcher writes it), and drop the old heartbeat so
   the pod is not ready until this attempt finishes a step.
2. train (all 16 ranks): set what torchrun used to set (RANK, WORLD_SIZE, ...)
   plus this attempt's torchft identity, then run train.py's main(). When it
   completes, rank 0 writes a marker so the worker pod exits 0.

PodHealth -- one, on its own process in the pod, so it outlives a crashed
training process. The controller uses it to read the pod's step heartbeat and,
when it decides the replica failed, to mark the pod failed: the pod's liveness
and readiness probes read that mark, and Kubernetes restarts the pod.
"""

from __future__ import annotations

import json
import os
import sys
import time

from monarch.actor import Actor, current_rank, endpoint

LOCAL_RANKS = 16
RUN_DIR = "/run/torchft-neuron-m1"
HEARTBEAT = os.path.join(RUN_DIR, "qwen3-heartbeat.json")
FAILED_MARK = os.path.join(RUN_DIR, "qwen3-failed")
COMPLETED_MARK = os.path.join(RUN_DIR, "qwen3-training-completed")

# What the deps init container built (workers.yaml), first on the path. Monarch 0.8
# ships its own top-level `examples` package, which otherwise shadows the demo
# trainer's examples.kubernetes.neuron_m1 in the processes Monarch spawns.
_DEPS_PATHS = ("/deps/torchft-examples", "/deps/torchtitan", "/deps/site")


def _use_deps_paths() -> None:
    for path in reversed(_DEPS_PATHS):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)
    loaded = sys.modules.get("examples")
    if loaded is not None and not str(getattr(loaded, "__file__", "")).startswith(_DEPS_PATHS[0]):
        for name in [n for n in sys.modules if n == "examples" or n.startswith("examples.")]:
            del sys.modules[name]


def _identity_env(replica: int, worker_inc: str, process_inc: str) -> dict[str, str]:
    """The demo trainer's per-launch identity (set in job.yaml's command before)."""
    return {
        "TORCHFT_NEURON_M1_LOGICAL_SLOT": str(replica),
        "TORCHFT_NEURON_M1_REPLICA_ID": f"torchft-neuron-m1-replica-{replica}",
        "TORCHFT_NEURON_TRANSPORT_SLOT": str(replica),
        "TORCHFT_NEURON_M1_WORKER_INCARNATION": worker_inc,
        "TORCHFT_NEURON_M1_PROCESS_INCARNATION": process_inc,
        "TORCHFT_LOCAL_STORE_GENERATION": (
            f"{os.environ['TORCHFT_NEURON_M1_POD_UID']}:{worker_inc}:{process_inc}"
        ),
    }


class PodHealth(Actor):
    @endpoint
    def heartbeat_age(self) -> float | None:
        """Seconds since this pod's replica last finished a step; None if it has not
        stepped yet (compiling or healing)."""
        try:
            with open(HEARTBEAT) as f:
                return time.time() - float(json.load(f)["time"])
        except (OSError, ValueError, KeyError):
            return None

    @endpoint
    def mark_failed(self, reason: str) -> None:
        """The controller's decision: this replica failed. The probes read it and
        Kubernetes restarts the pod."""
        with open(FAILED_MARK, "w") as f:
            f.write(reason)
        print(f"pod marked failed by the controller: {reason}", flush=True)


class ReplicaTrainer(Actor):
    @endpoint
    def prepare(self, replica: int, worker_inc: str, process_inc: str) -> None:
        _use_deps_paths()
        from examples.kubernetes.neuron_m1.launcher import DEFAULT_SENTINEL, claim_launch

        os.environ.update(_identity_env(replica, worker_inc, process_inc))
        for name in ("launch.json", "qwen3-heartbeat.json", "torchrun.pid"):
            try:
                os.unlink(os.path.join(RUN_DIR, name))
            except FileNotFoundError:
                pass
        if not claim_launch(
            DEFAULT_SENTINEL,
            pod_uid=os.environ["TORCHFT_NEURON_M1_POD_UID"],
            run_id=os.environ["TORCHFT_NEURON_M1_RUN_ID"],
            worker_incarnation=worker_inc,
            process_incarnation=process_inc,
        ):
            raise RuntimeError("launch claim already exists")

    @endpoint
    def train(
        self,
        replica: int,
        worker_inc: str,
        process_inc: str,
        master_port: int,
        args: list[str],
    ) -> None:
        _use_deps_paths()
        rank = current_rank().rank
        os.environ.update(_identity_env(replica, worker_inc, process_inc))
        # What torchrun --standalone --nproc_per_node=16 set for each rank.
        os.environ.update(
            {
                "RANK": str(rank),
                "LOCAL_RANK": str(rank),
                "WORLD_SIZE": str(LOCAL_RANKS),
                "LOCAL_WORLD_SIZE": str(LOCAL_RANKS),
                "GROUP_RANK": "0",
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(master_port),
            }
        )
        # Like torchrun --local-ranks-filter 0: only rank 0 prints; the others log
        # to files in the pod.
        pod_log = os.dup(2)  # the pod log, kept for failures of any rank
        if rank != 0:
            log = os.open(f"/tmp/rank-{rank:02d}.log", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            os.dup2(log, 1)
            os.dup2(log, 2)
        print(f"replica {replica} rank {rank}: training (master port {master_port})", flush=True)
        if rank == 0:
            # For the fault demo: kill -9 "$(cat /run/torchft-neuron-m1/train-rank0.pid)".
            with open(os.path.join(RUN_DIR, "train-rank0.pid"), "w") as f:
                f.write(f"{os.getpid()}\n")

        try:
            # Monarch imports torch when it starts this process, before RANK/LOCAL_RANK
            # are set, so workers.yaml turns torch's device auto-load off (else all 16
            # processes would initialize Neuron at once, without a rank). Load
            # torch-neuronx now, with this rank's identity, which also registers the
            # "neuron" compile backend.
            import torch_neuronx  # noqa: F401

            # Now TorchTitan (imported by m1_shape's demo modules) sees the Neuron
            # device; see qwen3_pretrain/__init__.py.
            from qwen3_pretrain import m1_shape  # noqa: F401
            from qwen3_pretrain import train as entry

            sys.argv = ["train.py", *args]
            entry.main(list(args))
        except BaseException:
            # The controller only gets a one-line summary, and ranks other than 0 log
            # to files in the pod: write what to debug with to the pod log.
            import traceback

            examples = sys.modules.get("examples")
            os.write(
                pod_log,
                (
                    f"replica {replica} rank {rank} FAILED:\n{traceback.format_exc()}"
                    f"examples module: {examples} {getattr(examples, '__path__', None)}\n"
                    f"sys.path: {sys.path}\n"
                ).encode(),
            )
            # Plain text to the controller: it cannot rebuild the demo trainer's
            # exception classes (it does not have that code).
            error = sys.exc_info()[1]
            raise RuntimeError(
                f"replica {replica} rank {rank}: {type(error).__name__}: {error}"
            ) from None
        if rank == 0:
            open(COMPLETED_MARK, "w").close()
