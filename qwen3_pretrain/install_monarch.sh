#!/usr/bin/env bash
# Install Monarch (torchmonarch 0.4.1, the GRPO pin) into /tmp/monarch-pkgs without
# touching the image's packages. The list is exactly what `pip install
# torchmonarch==0.4.1` added on top of the worker image (monarch-check.yaml,
# 2026-10-19); everything else it needs is already in the image. --no-deps keeps
# pip from pulling second copies of image packages into the target.
# Callers append /tmp/monarch-pkgs to PYTHONPATH.
set -euo pipefail
PY=/opt/torch-neuronx/.venv/bin/python
"$PY" -m pip install --quiet --no-deps --target /tmp/monarch-pkgs \
  torchmonarch==0.4.1 \
  Flask==3.1.3 blinker==1.9.0 click-option-group==0.5.9 cloudpickle==3.1.2 \
  clusterscope==0.0.32 docker==7.2.0 itsdangerous==2.2.0 lark==1.3.1 \
  mypy_extensions==1.1.0 py-spy==0.4.2 pyre-extensions==0.0.32 pyzmq==27.2.0 \
  tabulate==0.10.0 torchx-nightly==2026.10.19 typing-inspect==0.9.0
PYTHONPATH="${PYTHONPATH:-}:/tmp/monarch-pkgs" "$PY" -c 'import monarch.actor' \
  || { echo "FATAL: monarch did not import after install"; exit 1; }
echo "monarch installed in /tmp/monarch-pkgs"
