"""Upstream torchft over NIXL: torchstore's process group and healing transport.

TorchTitan's torchft integration builds ``torchft.ProcessGroupGloo`` for
``--fault_tolerance.process_group gloo`` and gives the Manager no checkpoint
transport. ``install()`` swaps in ``torchstore.integrations.torchft``'s
``ProcessGroupNixl`` (Gloo control plane, device allreduce over NIXL; LIBFABRIC/EFA
on Neuron) and ``NixlCheckpointTransport`` (healing: one NIXL READ of all state, in
place). torchft and TorchTitan stay upstream and unmodified; the transport lives in
the torchstore fork (github.com/yahavb/torchstore, branch nixl-neuron).
"""

from __future__ import annotations

import os
from datetime import timedelta

_installed = False


def install() -> None:
    global _installed
    if _installed or os.environ.get("QWEN3_FT_TRANSPORT", "nixl") != "nixl":
        return
    import torchft
    from torchstore.integrations.torchft import NixlCheckpointTransport, ProcessGroupNixl

    upstream_manager = torchft.Manager
    heal_timeout = timedelta(seconds=float(os.environ.get("QWEN3_HEAL_TIMEOUT_S", "300")))

    def manager(*args, pg, **kwargs):
        transport = NixlCheckpointTransport(pg, timeout=heal_timeout)
        kwargs.setdefault("checkpoint_transport", transport)
        # Advertise an address other pods can reach (the pod IP), not the pod name.
        if os.environ.get("TORCHFT_ADVERTISED_HOST"):
            kwargs.setdefault("hostname", os.environ["TORCHFT_ADVERTISED_HOST"])
        instance = upstream_manager(*args, pg=pg, **kwargs)
        # Receive in place into this replica's live state.
        transport.set_state_dict(instance._manager_state_dict)
        return instance

    torchft.ProcessGroupGloo = ProcessGroupNixl
    torchft.Manager = manager
    _installed = True
