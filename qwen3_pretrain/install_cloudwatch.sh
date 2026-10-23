#!/usr/bin/env bash
# boto3 for qwen3_pretrain/cloudwatch.py, unless the image already has it. Installed
# into /tmp/cw-pkgs, which callers APPEND to PYTHONPATH: any of its dependencies the
# image already has (urllib3, ...) keep the image's version.
set -euo pipefail
PY="${PY:-/usr/local/bin/python3}"
if "$PY" -c 'import boto3' 2>/dev/null; then
  echo "boto3 already in the image"
  exit 0
fi
"$PY" -m pip install --quiet --target /tmp/cw-pkgs 'boto3>=1.34,<2'
PYTHONPATH="${PYTHONPATH:-}:/tmp/cw-pkgs" "$PY" -c 'import boto3' \
  || { echo "WARNING: boto3 did not import; CloudWatch metrics will be off"; exit 0; }
echo "boto3 installed in /tmp/cw-pkgs"
