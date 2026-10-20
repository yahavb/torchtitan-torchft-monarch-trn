"""Monarch controller: the brain. Kubernetes is the hands.

The worker pods (workers.yaml) start idle. For each replica this controller
attaches to its worker pod, spawns the pod's 16 training processes
(ReplicaTrainer) plus a PodHealth actor, and tells it to train.

It watches every replica and decides when one must restart:
- crash: the replica's train call fails (e.g. a killed or crashed rank);
- hang: the replica stops finishing steps for STALE_AFTER_S.
To restart it, the controller marks the pod failed (PodHealth.mark_failed): the
pod's liveness probe fails on that mark, Kubernetes restarts the pod, and the
controller tells the new pod to train. torchft heals the new replica from a
healthy peer while the others keep training.

Training arguments (e.g. --training.steps 5000) are this script's arguments.
"""

from __future__ import annotations

import asyncio
import os
import socket
import sys
import time
import uuid

import monarch.actor
from monarch.actor import attach_to_workers, enable_transport

from qwen3_pretrain.replica_actor import LOCAL_RANKS, PodHealth, ReplicaTrainer

REPLICAS = int(os.environ.get("NNODES", "4"))
WORKER_SERVICE = os.environ.get("WORKER_SERVICE", "qwen3-workers")
WORKER_PORT = int(os.environ.get("MONARCH_WORKER_PORT", "26600"))
RETRY_DELAY_S = 5
# A step is ~5.5 s; a survivor's reconfiguration, or serving a rejoiner's heal,
# takes ~10-30 s. Well above both.
STALE_AFTER_S = float(os.environ.get("STALE_AFTER_S", "120"))
WATCH_EVERY_S = 15


def log(message: str) -> None:
    print(f"[controller {time.strftime('%H:%M:%S')}] {message}", flush=True)


_reported_faults: set[str] = set()


def _fault_hook(failure: object) -> None:
    # Monarch's default exits the controller. A failed replica is handled by its
    # watch loop below (its train call fails), so only log here -- once per failed
    # actor: Monarch re-delivers the events of a dead pod every 10 s.
    report = getattr(failure, "report", None)
    text = str(report() if callable(report) else failure)
    key = text.splitlines()[0] if text else ""
    if key not in _reported_faults:
        _reported_faults.add(key)
        log(f"supervision event: {key}")


monarch.actor.unhandled_fault_hook = _fault_hook


async def _worker_address(replica: int, failed: set[str]) -> str:
    """tcp://IP:port of replica's worker pod once it is listening, skipping
    addresses of pods marked failed: after a failure this waits for the pod
    Kubernetes brings back (a new IP)."""
    host = f"{WORKER_SERVICE}-{replica}.{WORKER_SERVICE}"
    while True:
        try:
            ip = socket.gethostbyname(host)
            if f"tcp://{ip}:{WORKER_PORT}" in failed:
                await asyncio.sleep(RETRY_DELAY_S)
                continue
            _, writer = await asyncio.wait_for(asyncio.open_connection(ip, WORKER_PORT), 5)
            writer.close()
            return f"tcp://{ip}:{WORKER_PORT}"
        except (OSError, asyncio.TimeoutError):
            await asyncio.sleep(RETRY_DELAY_S)


STOP_TIMEOUT_S = 20


async def _stop(mesh: object, what: str) -> None:
    """Best effort, bounded: stopping a mesh on a pod that is already gone can
    hang forever, which once kept a replica from re-attaching."""
    if mesh is None:
        return
    try:
        await asyncio.wait_for(_await(mesh.stop()), STOP_TIMEOUT_S)
    except asyncio.TimeoutError:
        log(f"{what} stop: gave up after {STOP_TIMEOUT_S} s (pod likely gone)")
    except Exception as error:  # already dead is fine
        log(f"{what} stop: {type(error).__name__}: {str(error)[:200]}")


def _stop_in_background(meshes: tuple, replica: int) -> None:
    """After a failure the pod is going away: clean up without making the replica
    wait, so it attaches to the new pod as soon as Kubernetes has it."""

    async def stop_all() -> None:
        for mesh, what in meshes:
            await _stop(mesh, f"replica {replica} {what}")

    asyncio.create_task(stop_all())


async def _await(future: object) -> object:
    return await future


async def _watch(replica: int, train: asyncio.Task, health: PodHealth) -> str | None:
    """Wait for this replica's training to end. None if it completed; otherwise
    the reason the controller decides it failed."""
    stepping = False
    while True:
        done, _ = await asyncio.wait({train}, timeout=WATCH_EVERY_S)
        if done:
            error = train.exception()
            if error is None:
                return None
            return f"crash: {type(error).__name__}: {str(error)[:300]}"
        try:
            age = await asyncio.wait_for(_await(health.heartbeat_age.call_one()), STOP_TIMEOUT_S)
        except Exception as error:
            # The pod's health actor is unreachable: the pod is gone or wedged.
            return f"unreachable: {type(error).__name__}: {str(error)[:200]}"
        if age is None:
            continue  # compiling or healing: not stepping yet
        if age <= STALE_AFTER_S:
            stepping = True
        elif stepping:
            return f"hang: no training step in {age:.0f} s"


async def run_replica(replica: int, args: list[str]) -> None:
    attempt = 0
    failed: set[str] = set()
    while True:
        if attempt:
            log(f"replica {replica}: waiting for Kubernetes to restart its worker pod")
        address = await _worker_address(replica, failed)
        log(f"replica {replica} attempt {attempt}: worker {address}")
        host = procs = health_procs = None
        try:
            host = attach_to_workers(
                # Monarch names: letters, digits and underscores only.
                name=f"replica{replica}_attempt{attempt}",
                ca="trust_all_connections",
                workers=[address],
            )
            await host.initialized
            # Its own process, so it survives a crashed training process.
            health_procs = host.spawn_procs(per_host={"health": 1})
            health = health_procs.spawn("health", PodHealth)
            procs = host.spawn_procs(per_host={"ranks": LOCAL_RANKS})
            trainer = procs.spawn("trainer", ReplicaTrainer)
            worker_inc, process_inc = uuid.uuid4().hex, uuid.uuid4().hex
            await trainer.slice(ranks=0).prepare.call_one(replica, worker_inc, process_inc)
        except Exception as error:
            # Before training (e.g. attach): the pod is fine; try it again.
            log(f"replica {replica} attempt {attempt}: setup failed, retrying: {type(error).__name__}: {str(error)[:300]}")
            await _stop(procs, f"replica {replica} procs")
            await _stop(health_procs, f"replica {replica} health")
            await _stop(host, f"replica {replica} host")
            await asyncio.sleep(RETRY_DELAY_S)
            attempt += 1
            continue

        # A new port per attempt: the old store's port may still be in TIME_WAIT.
        master_port = 29500 + attempt % 100
        log(f"replica {replica} attempt {attempt}: training on {LOCAL_RANKS} processes")
        train = asyncio.create_task(
            _await(trainer.train.call(replica, worker_inc, process_inc, master_port, args))
        )
        reason = await _watch(replica, train, health)
        if reason is None:
            log(f"replica {replica}: training completed")
            for mesh, what in ((procs, "procs"), (health_procs, "health"), (host, "host")):
                await _stop(mesh, f"replica {replica} {what}")
            return

        # The decision: restart this replica. Kubernetes does it.
        log(f"replica {replica} FAILED ({reason}): marking its pod failed for Kubernetes to restart")
        try:
            await asyncio.wait_for(_await(health.mark_failed.call_one(reason)), STOP_TIMEOUT_S)
        except Exception as error:
            log(f"replica {replica}: could not mark the pod ({type(error).__name__}); it is likely gone already")
        failed.add(address)
        train.cancel()
        _stop_in_background(((procs, "procs"), (health_procs, "health"), (host, "host")), replica)
        attempt += 1


async def main(args: list[str]) -> None:
    enable_transport(f"tcp://{os.environ['POD_IP']}:0")
    log(f"supervising {REPLICAS} replicas on {WORKER_SERVICE}-N:{WORKER_PORT}, args {args}")
    await asyncio.gather(*(run_replica(replica, args) for replica in range(REPLICAS)))
    log("all replicas completed")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
