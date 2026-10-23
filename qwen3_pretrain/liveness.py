"""Liveness probe: is THIS replica alive and stepping? Failing it makes Kubernetes
restart the pod (restartPolicy Never: the pod fails and its Job replaces it).

Dead when either:
- torchrun is gone (its PID file names a process that no longer exists);
- hang: the replica had started stepping, but its heartbeat (gradsync.py, one
  per step) is older than LIVE_MAX_AGE_S.

Before the first step (compiling, or healing after a restart) there is no
heartbeat yet, so only the torchrun check applies.

This is the torchrun version's hang detection; in the Monarch version the
controller does it (same 120 s limit) and marks the pod instead.
Run as a script so the probe does not load torch.
"""

from __future__ import annotations

import json
import os
import sys
import time

RUN_DIR = "/run/torchft-neuron-m1"
PID_FILE = os.path.join(RUN_DIR, "torchrun.pid")
HEARTBEAT = os.path.join(RUN_DIR, "qwen3-heartbeat.json")
# A step is ~6.6 s; a survivor's reconfiguration, or serving a rejoiner's heal,
# takes ~10-30 s. Well above both.
MAX_AGE_S = float(os.environ.get("LIVE_MAX_AGE_S", "120"))


def main() -> int:
    try:
        pid = int(open(PID_FILE).read().strip())
        os.kill(pid, 0)
    except (OSError, ValueError) as error:
        print(f"dead: torchrun not running ({error})")
        return 1
    try:
        with open(HEARTBEAT) as f:
            age = time.time() - float(json.load(f)["time"])
    except FileNotFoundError:
        print("alive: no step yet (compiling or healing)")
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(f"alive: heartbeat unreadable ({error})")
        return 0
    if age > MAX_AGE_S:
        print(f"dead: hang, no training step in {age:.0f} s (limit {MAX_AGE_S:.0f} s)")
        return 1
    print(f"alive: last step {age:.0f} s ago")
    return 0


if __name__ == "__main__":
    sys.exit(main())
