#!/usr/bin/env bash
# Environment setup for the E3 notebook (scripts/evaluate_vllm.py). Separate from
# kaggle_setup.sh: vLLM pins its own torch, so installing it replaces the image's
# torch, which the training notebook should not be exposed to.
set -euo pipefail

# Pinned so a re-run scores with the same engine; results.json records it as well.
# The vLLM docs list compute capability 7.5 (T4) as supported, in float16.
pip install -q "vllm==0.31.0" peft

# peft (used only for the merged-adapter fallback) refused Kaggle's old torchao in
# E2b. Remove torchao only if that problem shows up again.
if ! python -c "from peft.import_utils import is_torchao_available as f; f()" 2>/dev/null; then
    pip uninstall -y -q torchao
    echo "removed torchao (its version broke peft's import)"
fi

python - <<'EOF'
import importlib.metadata as md
for pkg in ("vllm", "torch", "transformers", "peft", "scipy", "scikit-learn"):
    try:
        print(f"{pkg}=={md.version(pkg)}")
    except md.PackageNotFoundError:
        print(f"{pkg}: not installed")
EOF
