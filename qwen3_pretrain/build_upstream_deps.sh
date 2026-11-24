#!/usr/bin/env bash
# Upstream torchft + the torchstore fork's NIXL integration, cached per node
# (like NIXL's build_nixl.sh): the first pod on a node builds, later pods reuse it.
#
#   torchft     meta-pytorch/torchft @ $TORCHFT_UPSTREAM_REF (Rust, maturin)
#               -> $CACHE/torchft-upstream/<ref>/site
#   torchstore  github.com/yahavb/torchstore @ $TORCHSTORE_REF (branch nixl-neuron):
#               torchstore.integrations.torchft (ProcessGroupNixl, NixlCheckpointTransport)
#               -> $CACHE/torchstore/<ref>/src, its pure-Python deps in .../site
#
# Prints the PYTHONPATH entries to use, first:  eval "$(bash build_upstream_deps.sh)"
set -euo pipefail
: "${TORCHFT_UPSTREAM_REF:?}" "${TORCHSTORE_REF:?}"
CACHE="${DEPS_CACHE:-/node-cache/deps}"
PY="${PY:-/opt/torch-neuronx/.venv/bin/python}"
export HOME="${HOME:-/tmp/home}" CARGO_HOME="${CARGO_HOME:-/tmp/cargo}" RUSTUP_HOME="${RUSTUP_HOME:-/tmp/rustup}"
log() { echo "[upstream-deps $(date -u +%H:%M:%S)] $*" >&2; }

TFT="$CACHE/torchft-upstream/$TORCHFT_UPSTREAM_REF"
TS="$CACHE/torchstore/$TORCHSTORE_REF"
mkdir -p "$CACHE"

if [ ! -f "$TFT/.complete" ]; then
  mkdir -p "$TFT"; exec 8>"$TFT.lock"; flock 8
  if [ ! -f "$TFT/.complete" ]; then
    log "building torchft $TORCHFT_UPSTREAM_REF"
    mkdir -p "$HOME"; W=$(mktemp -d)
    git clone -q https://github.com/meta-pytorch/torchft.git "$W/torchft"
    git -C "$W/torchft" checkout -q "$TORCHFT_UPSTREAM_REF"
    command -v cargo >/dev/null || { curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --no-modify-path >/tmp/rustup.log 2>&1 || { tail -5 /tmp/rustup.log >&2; exit 1; }; }
    export PATH="$CARGO_HOME/bin:$PATH"
    if ! command -v protoc >/dev/null; then
      curl -sSL -o "$W/protoc.zip" https://github.com/protocolbuffers/protobuf/releases/download/v29.3/protoc-29.3-linux-x86_64.zip
      "$PY" -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$W/protoc.zip" "$W/protoc"
      chmod +x "$W/protoc/bin/protoc"; export PATH="$W/protoc/bin:$PATH"
    fi
    PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 "$PY" -m pip install --quiet --no-deps --target "$TFT/site" "$W/torchft" >&2
    rm -rf "$W"; touch "$TFT/.complete"
  fi
  exec 8>&-
fi

if [ ! -f "$TS/.complete" ]; then
  mkdir -p "$TS"; exec 9>"$TS.lock"; flock 9
  if [ ! -f "$TS/.complete" ]; then
    log "fetching torchstore $TORCHSTORE_REF"
    rm -rf "$TS/src"
    git clone -q -b nixl-neuron https://github.com/yahavb/torchstore.git "$TS/src"
    git -C "$TS/src" checkout -q "$TORCHSTORE_REF"
    "$PY" -m pip install --quiet --no-deps --target "$TS/site" pygtrie portpicker >&2
    touch "$TS/.complete"
  fi
  exec 9>&-
fi
chmod -R a+rX "$TFT" "$TS" 2>/dev/null || true
echo "export UPSTREAM_DEPS_PATHS=$TFT/site:$TS/src:$TS/site"
