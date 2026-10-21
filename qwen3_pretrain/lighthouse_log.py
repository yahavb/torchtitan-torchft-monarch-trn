"""Readable torchft lighthouse log: stdin = the lighthouse's raw log, stdout = the summary.

The lighthouse prints a status line every time one replica's quorum request
arrives, so a healthy step logs "New quorum not ready, only have N participants"
for the first arrivals, a few milliseconds before the quorum forms. This keeps:

- a QUORUM CHANGE line when the membership changes (quorum id, size, replicas);
- a QUORUM steady line every STEADY_EVERY quorums with unchanged membership;
- any line that is not per-request chatter (startup, warnings, errors).

The full raw log is kept separately (lighthouse.yaml tees it to a file).
"""

from __future__ import annotations

import os
import re
import sys

STEADY_EVERY = int(os.environ.get("LIGHTHOUSE_STEADY_EVERY", "100"))
CHATTER = ("Received quorum request", "New quorum not ready", "Valid quorum with", "Fast quorum found")
QUORUM_ID = re.compile(r"Quorum! Quorum \{ quorum_id: (\d+)")
REPLICA = re.compile(r'replica_id: "[^"]*?-(\d+):')


def main() -> None:
    last_id = None
    since_change = 0
    for line in sys.stdin:
        match = QUORUM_ID.search(line)
        if match:
            quorum_id = int(match.group(1))
            replicas = sorted(int(r) for r in REPLICA.findall(line))
            stamp = line.split(" ", 1)[0]
            if quorum_id != last_id:
                print(f"{stamp} QUORUM CHANGE id={quorum_id} replicas={len(replicas)} {replicas}", flush=True)
                last_id, since_change = quorum_id, 0
            else:
                since_change += 1
                if since_change % STEADY_EVERY == 0:
                    print(f"{stamp} QUORUM steady id={quorum_id} replicas={len(replicas)} "
                          f"({since_change} steps since the last change)", flush=True)
            continue
        if any(text in line for text in CHATTER):
            continue
        sys.stdout.write(line)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
