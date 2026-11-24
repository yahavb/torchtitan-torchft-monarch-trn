#!/bin/bash
# End-to-end failover test of the running Monarch job (workers.yaml + controller.yaml):
# wait for step >= 20 on all 4 replicas, break worker 3's chip (fault/break-device.sh),
# fix it once Kubernetes replaces the pod, then wait for the new pod to heal over NIXL,
# train, and the controller to report a full quorum [0, 1, 2, 3]. Prints the timings
# and PASS/FAIL. Usage: fault/heal-test.sh [log file]
cd "$(dirname "$0")/.."
OUT=${1:-/tmp/heal-test.txt}; : > $OUT
ts() { date -u +%H:%M:%S; }; say() { echo "$(ts) $*" | tee -a $OUT; }
k() { for i in 1 2 3 4 5; do o=$(kubectl "$@" 2>/dev/null) && { echo "$o"; return 0; }; sleep 5; done; return 1; }
running() { k get pods -l app=qwen3-workers --field-selector status.phase=Running -o name | sed 's|pod/||'; }
step_of() { k logs "$1" -c trainer --tail=3000 | grep -oE 'METRICS replica=[0-9] step=[0-9]+' | tail -1 | grep -oE '[0-9]+$'; }
say "waiting for step >= 20 on all 4 replicas"
for i in $(seq 1 200); do n=0; for p in $(running); do s=$(step_of $p); [ -n "$s" ] && [ "$s" -ge 20 ] && n=$((n+1)); done; [ $n -ge 4 ] && break; sleep 15; done
CTRL=$(k get pods -o name | grep qwen3-controller | head -1 | sed 's|pod/||')
P0=$(running | grep workers-0)
say "steady step_time (replica 0): $(k logs $P0 -c trainer --tail=3000 | grep -oE 'step_time=[0-9.]+s' | tail -6 | tr '\n' ' ')"
say "grad_norm per replica: $(for p in $(running); do k logs $p -c trainer --tail=3000 | grep -oE 'METRICS replica=[0-9] step=20 .*grad_norm=[0-9.]+' | grep -oE 'replica=[0-9]|grad_norm=[0-9.]+' | tr '\n' ' '; echo -n '| '; done)"
OLD=$(running | grep workers-3)
say "BREAK chip of $OLD"; bash fault/break-device.sh "$OLD" | grep -E "device:" | tee -a $OUT; TB=$(date -u +%s)
for i in $(seq 1 60); do NEW=$(k get pods -l app=qwen3-workers -o name | sed 's|pod/||' | grep workers-3 | grep -v "^$OLD$" | head -1); [ -n "$NEW" ] && break; sleep 5; done
say "replacement pod $NEW"; bash fault/fix-device.sh "$OLD" >/dev/null 2>&1 && say "FIX: taint removed"
for i in $(seq 1 120); do k logs "$NEW" -c trainer | grep -q "METRICS replica=3" && break; sleep 15; done
for i in $(seq 1 80); do k logs $CTRL -c controller | awk -v t="$(date -u -r $TB +%Y-%m-%dT%H:%M:%S)" '$1>t' | grep -q "QUORUM CHANGE .*replicas=4 \[0, 1, 2, 3\]" && break; sleep 15; done
FULL=$(k logs $CTRL -c controller | awk -v t="$(date -u -r $TB +%Y-%m-%dT%H:%M:%S)" '$1>t' | grep -m1 "QUORUM CHANGE .*replicas=4 \[0, 1, 2, 3\]")
L=$(k logs "$NEW" -c trainer)
g() { echo "$L" | grep -m1 -E "$1" | grep -oE '[0-9]{2}:[0-9]{2}:[0-9]{2}[.,][0-9]{3}' | head -1; }
say "fault at $(date -u -r $TB +%H:%M:%S); new pod created $(k get pod $NEW -o jsonpath='{.metadata.creationTimestamp}' | cut -c12-19)"
say "  training starts     $(g 'Training starts')"
say "  quorum (rejoin)     $(g 'got lighthouse quorum')"
say "  heal received       $(g 'NixlCheckpointTransport: received')"
echo "$L" | grep -m1 -E "NixlCheckpointTransport: received" | sed -E 's/.*(NixlCheckpointTransport)/  \1/' | cut -c1-200 | tee -a $OUT
say "  first step after heal: $(echo "$L" | grep -m1 -oE 'METRICS replica=3 step=[0-9]+ .*step_time=[0-9.]+s')"
say "controller since fault:"; k logs $CTRL -c controller | grep -E "QUORUM CHANGE|replica 3" | grep -v supervision | tail -8 | tee -a $OUT
for p in $(running | grep -v workers-3); do k logs $p -c trainer | grep -m1 -E "NixlCheckpointTransport: sent" | sed "s|^|  source $p: |" | cut -c1-220 | tee -a $OUT; done
sleep 60
say "after recovery: $(for p in $(running); do k logs $p -c trainer --tail=2000 | grep -oE 'METRICS replica=[0-9] step=[0-9]+ .*grad_norm=[0-9.]+ .*step_time=[0-9.]+s' | tail -1 | grep -oE 'replica=[0-9]|step=[0-9]+|grad_norm=[0-9.]+|step_time=[0-9.]+s' | tr '\n' ' '; echo -n '| '; done)"
# Verdict: a full quorum after the fault, the new pod healed over NIXL and trains, and
# all 4 replicas report the same loss and grad_norm at their latest common step.
last=$(for p in $(running); do k logs $p -c trainer --tail=2000 | grep -oE 'METRICS replica=[0-9] step=[0-9]+ loss=[0-9.]+ grad_norm=[0-9.]+'; done)
common=$(echo "$last" | grep -oE 'replica=[0-9] step=[0-9]+' | awk '{split($2,a,"=");s[$1]=a[2]} END{m=1e9;for(r in s) if(s[r]<m)m=s[r]; print m}')
same=$(echo "$last" | grep " step=$common " | grep -oE 'loss=[0-9.]+ grad_norm=[0-9.]+' | sort -u | wc -l | tr -d ' ')
reps=$(echo "$last" | grep " step=$common " | grep -oE 'replica=[0-9]' | sort -u | wc -l | tr -d ' ')
ok=1
[ -n "$FULL" ] && say "check: full quorum after fault: $FULL" || { say "FAIL: no full quorum [0, 1, 2, 3] after the fault"; ok=0; }
echo "$L" | grep -q "NixlCheckpointTransport: received" && say "check: new pod healed over NIXL" || { say "FAIL: new pod did not heal over NIXL"; ok=0; }
echo "$L" | grep -q "METRICS replica=3" && say "check: new pod trains" || { say "FAIL: new pod logged no step"; ok=0; }
[ "$reps" = 4 ] && [ "$same" = 1 ] && say "check: 4 replicas identical at step $common" || { say "FAIL: replicas differ at step $common ($reps replicas, $same distinct)"; ok=0; }
[ $ok = 1 ] && { say "PASS"; exit 0; } || { say "FAIL"; exit 1; }
