"""torchft's Lighthouse as a Monarch actor, in the controller pod.

As in the PyTorch Monarch blog's fault-tolerance case study: the single controller
owns the control plane, including the Lighthouse. The controller spawns this actor
on a process of its own pod and starts torchft's LighthouseServer in it; the
qwen3-lighthouse Service (controller.yaml) points at the controller pod, so the
replicas reach it at the same address as the standalone lighthouse.yaml.
"""

from __future__ import annotations

from monarch.actor import Actor, endpoint


class LighthouseActor(Actor):
    def __init__(self) -> None:
        self._server = None

    @endpoint
    def start(
        self,
        bind: str,
        min_replicas: int,
        initial_replicas: int,
        join_timeout_ms: int,
        quorum_tick_ms: int,
    ) -> str:
        try:
            from torchft.coordination import LighthouseServer
        except ImportError:
            from torchft._torchft import LighthouseServer

        if self._server is None:
            self._server = LighthouseServer(
                bind=bind,
                min_replicas=min_replicas,
                initial_replicas=initial_replicas,
                join_timeout_ms=join_timeout_ms,
                quorum_tick_ms=quorum_tick_ms,
            )
        return self._server.address()
