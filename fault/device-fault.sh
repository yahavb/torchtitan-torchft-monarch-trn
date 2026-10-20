#!/usr/bin/env bash
# Simulate a Neuron device hardware fault for one training pod, and repair it.
#
#   fault/device-fault.sh inject <pod>   mark one of <pod>'s Neuron devices unhealthy
#   fault/device-fault.sh repair         mark it healthy again
#
# Requires a Neuron DRA driver build whose health poller reads a simulated sysfs
# tree inside the driver pod (SIM_ROOT). Writing a component name into a device's
# unhealthy_components file makes the driver taint that device (NoExecute), and
# Kubernetes evicts the pod using it. Emptying the file makes the driver lift the
# taint. Only devices that expose that file are used.
set -euo pipefail

SIM_ROOT="${SIM_ROOT:-/tmp/fake-neuron}"
DRIVER_NS="${DRIVER_NS:-neuron-dra-driver}"
DRIVER_LABEL="${DRIVER_LABEL:-app=neuron-dra-driver-kubelet-plugin}"
DRIVER_CTR="${DRIVER_CTR:-neuron-dra-driver}"
COMPONENT="${COMPONENT:-hbm}"
RECORD="${RECORD:-$HOME/.device-fault}"
WAIT_S="${WAIT_S:-60}"

fail() { echo "device-fault: $*" >&2; exit 1; }
health_file() { echo "$SIM_ROOT/neuron$1/stats/hardware/health_status/unhealthy_components"; }

in_driver() { # <node> <command...>: run a command in the DRA driver pod on <node>
  local node=$1 pod; shift
  pod=$(kubectl -n "$DRIVER_NS" get pods -l "$DRIVER_LABEL" --field-selector "spec.nodeName=$node" \
    -o jsonpath='{.items[0].metadata.name}')
  [ -n "$pod" ] || fail "no DRA driver pod on $node"
  kubectl -n "$DRIVER_NS" exec "$pod" -c "$DRIVER_CTR" -- "$@"
}

taint_of() { # <node> <device>: the device's taints as published, empty when untainted
  kubectl get resourceslices -o json | python3 -c '
import json, sys
node, dev = sys.argv[1], "neuron-device-" + sys.argv[2]
for s in json.load(sys.stdin)["items"]:
    if s["spec"].get("nodeName") != node:
        continue
    for d in s["spec"].get("devices", []):
        if d["name"] == dev and d.get("taints"):
            print(json.dumps(d["taints"]))
' "$1" "$2"
}

devices_of() { # <pod>: indices of the Neuron devices allocated to <pod>
  local claim
  for claim in $(kubectl get pod "$1" -o jsonpath='{.status.resourceClaimStatuses[*].resourceClaimName}'); do
    kubectl get resourceclaim "$claim" -o json | python3 -c '
import json, sys
for r in json.load(sys.stdin)["status"]["allocation"]["devices"]["results"]:
    if r["driver"] == "neuron.aws.com":
        print(r["device"].rsplit("-", 1)[1])
'
  done
}

wait_until() { # <want: tainted|clean> <node> <device>
  local start=$SECONDS t
  while (( SECONDS - start < WAIT_S )); do
    t=$(taint_of "$2" "$3")
    if [ "$1" = tainted ] && [ -n "$t" ]; then echo "tainted after $((SECONDS - start)) s: $t"; return 0; fi
    if [ "$1" = clean ] && [ -z "$t" ]; then echo "taint lifted after $((SECONDS - start)) s"; return 0; fi
    sleep 1
  done
  fail "device did not become $1 within $WAIT_S s; see the DRA driver log"
}

case "${1:-}" in
inject)
  pod="${2:?usage: device-fault.sh inject <pod>}"
  [ ! -s "$RECORD" ] || fail "a fault is already active ($RECORD); repair it first"
  node=$(kubectl get pod "$pod" -o jsonpath='{.spec.nodeName}')
  [ -n "$node" ] || fail "pod $pod is not scheduled"
  device=""
  for d in $(devices_of "$pod"); do
    if in_driver "$node" test -f "$(health_file "$d")"; then device=$d; break; fi
  done
  [ -n "$device" ] || fail "none of $pod's devices exposes $(health_file N)"
  echo "node=$node device=$device pod=$pod" > "$RECORD"
  echo "inject: $COMPONENT fault on neuron-device-$device ($pod on $node)"
  in_driver "$node" sh -c "echo $COMPONENT > $(health_file "$device")"
  wait_until tainted "$node" "$device"
  echo "now: kubectl get po -w   (the pod is evicted; repair with: fault/device-fault.sh repair)"
  ;;
repair)
  [ -s "$RECORD" ] || fail "no active fault ($RECORD)"
  read -r node_kv device_kv _ < "$RECORD"
  node=${node_kv#node=}; device=${device_kv#device=}
  echo "repair: neuron-device-$device on $node"
  # Empty, not delete: the driver treats a missing file as "no reading" and keeps the taint.
  in_driver "$node" sh -c ": > $(health_file "$device")"
  wait_until clean "$node" "$device"
  rm -f "$RECORD"
  ;;
*)
  sed -n '2,6p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
  ;;
esac
