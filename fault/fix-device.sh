#!/usr/bin/env bash
# Fix the device broken by fault/break-device.sh: remove the taint.
set -euo pipefail
kubectl delete devicetaintrule demo-fault
echo "device fixed"
