#!/usr/bin/env bash
# Build what the public Neuron image (torch + torch-neuronx) lacks, into $DEPS
# (an emptyDir shared with the main container). Everything is pinned:
#
#   torchft     $TORCHFT_REPO @ $TORCHFT_REF -- the fork with torchft/neuron.py
#               (ProcessGroupNeuron, over torch-neuronx's Neuron RDMA backend),
#               built with maturin/Rust -> $DEPS/site (+ $DEPS/site/bin/torchft_lighthouse)
#   demo code   the same repo's examples/kubernetes/neuron_m1 (trainer, manager, ...)
#               -> $DEPS/torchft-examples/examples
#   TorchTitan  pytorch/torchtitan @ $TORCHTITAN_REF + the fork's two patches
#               -> $DEPS/torchtitan
#
# usage: build_deps.sh [all|torchft]     (torchft: only what the lighthouse needs)
# Needs GITHUB_TOKEN for the private torchft fork.
set -euo pipefail
WHAT="${1:-all}"
DEPS="${DEPS:-/deps}"
PY="${PY:-/opt/py312/bin/python}"
: "${TORCHFT_REPO:?}" "${TORCHFT_REF:?}" "${GITHUB_TOKEN:?GITHUB_TOKEN (secret github-token) is required}"
export HOME="${HOME:-/tmp/home}" CARGO_HOME="${CARGO_HOME:-/tmp/cargo}" RUSTUP_HOME="${RUSTUP_HOME:-/tmp/rustup}"
mkdir -p "$HOME" "$DEPS"
log() { echo "[build_deps $(date -u +%H:%M:%S)] $*"; }

log "torchft: $TORCHFT_REPO @ $TORCHFT_REF"
rm -rf /tmp/torchft
git clone -q "https://x-access-token:${GITHUB_TOKEN}@github.com/${TORCHFT_REPO}.git" /tmp/torchft
git -C /tmp/torchft checkout -q "$TORCHFT_REF"

if ! command -v cargo >/dev/null; then
  log "installing Rust (minimal)"
  curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --no-modify-path >/tmp/rustup.log 2>&1 \
    || { tail -5 /tmp/rustup.log; exit 1; }
  export PATH="$CARGO_HOME/bin:$PATH"
fi
if ! command -v protoc >/dev/null; then
  log "installing protoc 29.3"
  curl -sSL -o /tmp/protoc.zip \
    https://github.com/protocolbuffers/protobuf/releases/download/v29.3/protoc-29.3-linux-x86_64.zip
  "$PY" -c 'import zipfile; zipfile.ZipFile("/tmp/protoc.zip").extractall("/tmp/protoc")'
  chmod +x /tmp/protoc/bin/protoc
  export PATH="/tmp/protoc/bin:$PATH"
fi
# PyO3 0.24 (torchft's Rust-Python bridge) stops at Python 3.13. Harmless on the
# Python 3.12 image; needed if PY is the DLC's Python 3.14 (public-image-check.yaml).
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
log "building torchft"
# --upgrade: an init container that restarts reuses the same deps volume.
"$PY" -m pip install --quiet --no-deps --upgrade --target "$DEPS/site" /tmp/torchft
# No Neuron device here: keep torch from auto-loading torch-neuronx for this check.
if ! out=$(TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH="$DEPS/site" "$PY" -c 'import torchft.neuron; print("torchft.neuron OK")' 2>&1); then
  echo "$out" | tail -20
  log "torchft import check FAILED"
  exit 1
fi
log "$out; built into $DEPS/site ($(ls "$DEPS/site/bin" | tr '\n' ' '))"

[ "$WHAT" = torchft ] && exit 0

log "demo code: examples/kubernetes/neuron_m1"
rm -rf "$DEPS/torchft-examples"
mkdir -p "$DEPS/torchft-examples"
cp -r /tmp/torchft/examples "$DEPS/torchft-examples/"

: "${TORCHTITAN_REF:?}"
log "TorchTitan @ $TORCHTITAN_REF + the fork's patches"
rm -rf "$DEPS/torchtitan"
git clone -q https://github.com/pytorch/torchtitan.git "$DEPS/torchtitan"
git -C "$DEPS/torchtitan" checkout -q "$TORCHTITAN_REF"
for patch in /tmp/torchft/examples/kubernetes/neuron_m1/torchtitan-*.patch; do
  git -C "$DEPS/torchtitan" apply "$patch"
  log "applied $(basename "$patch")"
done

# TorchTitan's Python dependencies that $PY's environment lacks, into $DEPS/site.
# pip resolves against what is installed (so torch and friends stay the image's) and
# reports only the missing packages; those are installed without further resolution.
# tensorboard and wandb are optional loggers, not used here; triton is added.
log "TorchTitan dependencies"
grep -v -E "^(tensorboard|wandb)" "$DEPS/torchtitan/requirements.txt" > /tmp/tt-req.txt
# TorchTitan imports triton at module level (distributed/minimal_async_ep/kernels.py)
# though Neuron never runs its kernels; the CPU torch wheel does not bring it.
echo "triton" >> /tmp/tt-req.txt
"$PY" -m pip install --quiet --dry-run --report /tmp/tt-report.json -r /tmp/tt-req.txt
MISSING=$("$PY" - <<'EOF'
import json
report = json.load(open("/tmp/tt-report.json"))
print(" ".join(f"{i['metadata']['name']}=={i['metadata']['version']}" for i in report["install"]))
EOF
)
if [ -n "$MISSING" ]; then
  log "installing: $MISSING"
  # shellcheck disable=SC2086
  "$PY" -m pip install --quiet --no-deps --upgrade --target "$DEPS/site" $MISSING
fi
TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH="$DEPS/torchtitan:$DEPS/site${PYTHONPATH:+:$PYTHONPATH}" "$PY" -c \
  'import datasets, tokenizers, torchtitan.trainer; print("torchtitan imports OK")'
log "done"
