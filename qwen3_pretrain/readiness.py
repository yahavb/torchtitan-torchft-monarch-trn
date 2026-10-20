"""Readiness probe: is THIS replica training?

Ready while this pod keeps finishing training steps: gradsync.py's heartbeat
(/run/torchft-neuron-m1/qwen3-heartbeat.json, written by local rank 0 after
every step) exists and is newer than READY_MAX_AGE_S. So:

- a survivor stays ready through another replica's failure (it keeps stepping);
- a replacement is not ready until it has healed and finished its first step.

It depends on no other replica: one pod dying does not make the others not ready.
Under Monarch (workers.yaml) it is also not ready once the controller has marked
this pod failed.
Run as a script so the probe does not load torch and stays within its timeout.
"""

from __future__ import annotations

import json
import os
import sys
import time

HEARTBEAT = "/run/torchft-neuron-m1/qwen3-heartbeat.json"
# Written by the Monarch controller when it decides this replica failed.
FAILED_MARK = "/run/torchft-neuron-m1/qwen3-failed"
# Far above a step (~5.5 s at 4 replicas) plus a reconfiguration after a fault.
MAX_AGE_S = float(os.environ.get("READY_MAX_AGE_S", "300"))


def main() -> int:
    if os.path.exists(FAILED_MARK):
        with open(FAILED_MARK) as f:
            print(json.dumps({"ready": False, "reason": f"marked failed by the controller: {f.read()}"}))
        return 1
    try:
        with open(HEARTBEAT) as f:
            beat = json.load(f)
    except FileNotFoundError:
        print(json.dumps({"ready": False, "reason": "no step finished yet"}))
        return 1
    except (OSError, ValueError) as error:
        print(json.dumps({"ready": False, "reason": f"heartbeat unreadable: {error}"}))
        return 1
    age = time.time() - float(beat["time"])
    status = {"ready": age <= MAX_AGE_S, "step": beat["step"], "age_s": round(age, 1)}
    if not status["ready"]:
        status["reason"] = f"no step in {age:.0f} s (limit {MAX_AGE_S:.0f} s)"
    print(json.dumps(status))
    return 0 if status["ready"] else 1


if __name__ == "__main__":
    sys.exit(main())
