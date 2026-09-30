#!/usr/bin/env bash
# Environment setup for the Kaggle notebook. It lives in the repository, not in a
# notebook cell, so a fix here reaches the notebook the next time it re-clones.
set -euo pipefail

# torch, transformers and datasets ship with Kaggle's image; these may not.
pip install -q peft accelerate

# Kaggle's image ships torchao 0.10.0. Current peft refuses any torchao older than
# 0.16 and raises ImportError inside get_peft_model, even for plain LoRA, which does
# not use torchao (seen 30 sep 2026). This project does not use torchao, so remove
# it rather than chase a version compatible with the installed torch.
if pip show torchao >/dev/null 2>&1; then
    pip uninstall -y -q torchao
    echo "removed torchao (unused here; its version broke peft)"
fi

python - <<'EOF'
import importlib.metadata as md
for pkg in ("torch", "transformers", "peft", "accelerate", "datasets", "torchao"):
    try:
        print(f"{pkg}=={md.version(pkg)}")
    except md.PackageNotFoundError:
        print(f"{pkg}: not installed")
EOF
