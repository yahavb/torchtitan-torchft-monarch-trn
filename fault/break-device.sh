#!/usr/bin/env bash
# Break the Neuron device a pod uses: find its node and device, then taint the device NoExecute.
#   fault/break-device.sh <pod>
# Undo with fault/fix-device.sh
set -euo pipefail
P=${1:?usage: fault/break-device.sh <pod>}

NODE=$(kubectl get po "$P" -o jsonpath='{.spec.nodeName}')
CLAIM=$(kubectl get po "$P" -o jsonpath='{.status.resourceClaimStatuses[0].resourceClaimName}')
[ -n "$CLAIM" ] || { echo "pod $P has no ResourceClaim"; exit 1; }
DEVICE=$(kubectl get resourceclaim "$CLAIM" -o jsonpath='{.status.allocation.devices.results[0].device}')

echo "pod:    $P"
echo "node:   $NODE"
echo "device: $DEVICE"
echo "breaking $DEVICE ..."

kubectl apply -f - <<EOF
apiVersion: resource.k8s.io/v1
kind: DeviceTaintRule
metadata:
  name: demo-fault
spec:
  deviceSelector:
    driver: neuron.aws.com
    pool: $NODE
    device: $DEVICE
  taint:
    key: neuron.aws.com/device-unhealthy
    value: simulated-fault
    effect: NoExecute
EOF
echo "done. Watch with: kubectl get po -w"
