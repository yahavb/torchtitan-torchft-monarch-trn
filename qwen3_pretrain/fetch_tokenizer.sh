#!/usr/bin/env bash
# The Qwen3 tokenizer for C4 training (config_registry.py), from Hugging Face: only
# the tokenizer and config files, a few MB, not the weights (the model trains from
# random init). Skipped for the c4_test sample, which uses TorchTitan's test tokenizer.
set -euo pipefail
[ "${QWEN3_DATASET:-c4}" = c4_test ] && { echo "c4_test: using the test tokenizer"; exit 0; }
DIR="${QWEN3_TOKENIZER_DIR:-/tmp/hf/Qwen3-14B}"
/opt/torch-neuronx/.venv/bin/python - "$DIR" <<'EOF'
import sys
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="Qwen/Qwen3-14B",
    local_dir=sys.argv[1],
    allow_patterns=["tokenizer.json", "tokenizer_config.json", "vocab.json",
                    "merges.txt", "config.json", "generation_config.json"],
)
print(f"Qwen3 tokenizer in {path}")
EOF
