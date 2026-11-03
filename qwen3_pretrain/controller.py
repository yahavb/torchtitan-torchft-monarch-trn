"""Monarch controller: the single controller of the torchft replicas.

As in the PyTorch Monarch blog's fault-tolerance case study (Monarch + torchft):

- the control plane is centralized here, including torchft's Lighthouse, which runs
  as a Monarch actor in this pod (lighthouse_actor.py);
- for each replica it attaches to that replica's worker pod (workers.yaml), spawns
  a replica group -- a process mesh of 16 training processes (ReplicaTrainer) --
  plus a PodHealth actor, and tells it to train;
- it detects failures with Monarch's primitives (a failed train call: crash) and
  the step heartbeat (hang), and recovers per failure type, cheapest first:
    1. fast restart in place: stop the mesh and spawn a new one in the same pod,
       keeping the allocation (devices) -- up to INPLACE_RESTARTS times;
    2. escalate to reallocation: mark the pod failed (its liveness probe fails on
       the mark), Kubernetes restarts it, and a new mesh joins in the new pod.
       Also when the pod itself is gone (eviction, node loss).
- torchft keeps the healthy replicas stepping throughout, and heals each new
  replica group from a healthy peer.

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
from monarch.actor import attach_to_workers, enable_transport, this_host

from qwen3_pretrain.lighthouse_actor import LighthouseActor

from qwen3_pretrain.monarch_worker import worker_address
from qwen3_pretrain.replica_actor import LOCAL_RANKS, PodHealth, ReplicaTrainer

REPLICAS = int(os.environ.get("NNODES", "4"))
WORKER_SERVICE = os.environ.get("WORKER_SERVICE", "qwen3-workers")
WORKER_PORT = int(os.environ.get("MONARCH_WORKER_PORT", "26600"))
RETRY_DELAY_S = 5
# A step is ~5.5 s; a survivor's reconfiguration, or serving a rejoiner's heal,
# takes ~10-30 s. Well above both.
STALE_AFTER_S = float(os.environ.get("STALE_AFTER_S", "120"))
WATCH_EVERY_S = 15
# Recovery strategy per failure type. Fast in-place restarts per pod before
# escalating to a pod restart (reallocation); a hang can be sent straight to
# reallocation with QWEN3_HANG_RECOVERY=reallocate.
INPLACE_RESTARTS = int(os.environ.get("QWEN3_INPLACE_RESTARTS", "2"))
HANG_RECOVERY = os.environ.get("QWEN3_HANG_RECOVERY", "restart")
# torchft Lighthouse (as an actor in this pod), same settings as lighthouse.yaml.
LIGHTHOUSE_BIND = os.environ.get("LIGHTHOUSE_BIND", "[::]:29510")
LIGHTHOUSE_MIN_REPLICAS = int(os.environ.get("LIGHTHOUSE_MIN_REPLICAS", "3"))
LIGHTHOUSE_INITIAL_REPLICAS = int(os.environ.get("LIGHTHOUSE_INITIAL_REPLICAS", str(REPLICAS)))


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
    """The Monarch address of replica's worker pod once it is listening, skipping
    addresses of pods marked failed: after a failure this waits for the pod
    Kubernetes brings back (a new IP)."""
    host = f"{WORKER_SERVICE}-{replica}.{WORKER_SERVICE}"
    while True:
        try:
            ip = socket.gethostbyname(host)
            if worker_address(replica, ip, WORKER_PORT) in failed:
                await asyncio.sleep(RETRY_DELAY_S)
                continue
            _, writer = await asyncio.wait_for(asyncio.open_connection(ip, WORKER_PORT), 5)
            writer.close()
            return worker_address(replica, ip, WORKER_PORT)
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


def _recovery(reason: str, inplace_done: int) -> str:
    """"restart" (fast, in place) or "reallocate" (Kubernetes restarts the pod)."""
    if reason.startswith("unreachable"):
        return "reallocate"  # the pod is gone or wedged: nothing to restart in
    if reason.startswith("hang") and HANG_RECOVERY == "reallocate":
        return "reallocate"
    if inplace_done >= INPLACE_RESTARTS:
        return "reallocate"  # in-place restarts did not stick: escalate
    return "restart"


async def run_replica(replica: int, args: list[str]) -> None:
    attempt = 0
    failed: set[str] = set()
    while True:  # one iteration per allocation (worker pod)
        if attempt:
            log(f"replica {replica}: waiting for Kubernetes to restart its worker pod")
        address = await _worker_address(replica, failed)
        log(f"replica {replica} attempt {attempt}: worker {address}")
        host = health_procs = health = None
        try:
            host = attach_to_workers(
                # Monarch names: letters, digits and underscores only.
                name=f"replica{replica}_attempt{attempt}",
                ca="trust_all_connections",
                workers=[address],
            )
            await host.initialized
            # Its own process, so it survives crashed training processes.
            health_procs = host.spawn_procs(per_host={"health": 1})
            health = health_procs.spawn("health", PodHealth)
        except Exception as error:
            # Before training (e.g. attach): the pod is fine; try it again.
            log(f"replica {replica} attempt {attempt}: attach failed, retrying: {type(error).__name__}: {str(error)[:300]}")
            await _stop(health_procs, f"replica {replica} health")
            await _stop(host, f"replica {replica} host")
            await asyncio.sleep(RETRY_DELAY_S)
            attempt += 1
            continue

        inplace = 0
        while True:  # one iteration per replica group (process mesh) on this pod
            procs = None
            try:
                procs = host.spawn_procs(per_host={"ranks": LOCAL_RANKS})
                trainer = procs.spawn("trainer", ReplicaTrainer)
                worker_inc, process_inc = uuid.uuid4().hex, uuid.uuid4().hex
                await trainer.slice(ranks=0).prepare.call_one(replica, worker_inc, process_inc)
            except Exception as error:
                reason = f"crash: mesh setup failed: {type(error).__name__}: {str(error)[:300]}"
                train = None
            else:
                # A new port per mesh: the old store's port may still be in TIME_WAIT.
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
            if train is not None:
                train.cancel()
            attempt += 1
            strategy = _recovery(reason, inplace)
            if strategy == "restart":
                inplace += 1
                log(f"replica {replica} FAILED ({reason}): fast restart in place "
                    f"({inplace}/{INPLACE_RESTARTS}), same pod and devices")
                await _stop(procs, f"replica {replica} procs")
                continue
            break

        # Escalate: reallocation. Kubernetes restarts the pod on the mark.
        log(f"replica {replica} FAILED ({reason}): escalating to reallocation, "
            f"marking its pod failed for Kubernetes to restart")
        try:
            await asyncio.wait_for(_await(health.mark_failed.call_one(reason)), STOP_TIMEOUT_S)
        except Exception as error:
            log(f"replica {replica}: could not mark the pod ({type(error).__name__}); it is likely gone already")
        failed.add(address)
        _stop_in_background(((procs, "procs"), (health_procs, "health"), (host, "host")), replica)


async def _start_lighthouse() -> object:
    """torchft's Lighthouse as a Monarch actor on a process of this pod."""
    procs = this_host().spawn_procs(per_host={"lighthouse": 1})
    lighthouse = procs.spawn("lighthouse", LighthouseActor)
    address = await lighthouse.start.call_one(
        LIGHTHOUSE_BIND, LIGHTHOUSE_MIN_REPLICAS, LIGHTHOUSE_INITIAL_REPLICAS, 20000, 100
    )
    log(f"torchft Lighthouse actor serving at {address} (min {LIGHTHOUSE_MIN_REPLICAS}, "
        f"initial {LIGHTHOUSE_INITIAL_REPLICAS} replicas)")
    return procs


async def main(args: list[str]) -> None:
    enable_transport(f"tcp://{os.environ['POD_IP']}:0")
    lighthouse = await _start_lighthouse()  # noqa: F841 (keeps the actor's mesh alive)
    log(f"supervising {REPLICAS} replicas on {WORKER_SERVICE}-N:{WORKER_PORT}, args {args}; "
        f"recovery: {INPLACE_RESTARTS} in-place restarts then reallocation, hang -> {HANG_RECOVERY}")
    await asyncio.gather(*(run_replica(replica, args) for replica in range(REPLICAS)))
    log("all replicas completed")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
