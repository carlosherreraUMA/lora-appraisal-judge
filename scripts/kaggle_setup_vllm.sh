#!/usr/bin/env bash
# Environment setup for the E3 notebook (scripts/evaluate_vllm.py). Separate from
# kaggle_setup.sh, which the training notebook uses.
set -euo pipefail

# vLLM is pinned to the release whose torch pin matches the torch Kaggle's image
# already has (2.10.0, CUDA 12.8, the one E2b ran on), so pip leaves torch alone.
# vLLM 0.27-0.31 pin torch 2.13 with torchaudio 2.11; from PyPI that is a CUDA 13
# torch next to a CUDA 12.8 torchaudio, and the first E3 smoke run died on exactly
# that mismatch (EXPERIMENTS.md, E3 setup note). The vLLM docs list compute
# capability 7.5 (T4) as supported, in float16.
pip install -q "vllm==0.19.1" peft

# peft (used only for the merged-adapter fallback) refused Kaggle's old torchao in
# E2b. Remove torchao only if that problem shows up again.
if ! python -c "from peft.import_utils import is_torchao_available as f; f()" 2>/dev/null; then
    pip uninstall -y -q torchao
    echo "removed torchao (its version broke peft's import)"
fi

# Fail here, in seconds, if the stack is inconsistent: everything evaluate_vllm.py
# imports, including the transformers class whose lazy import broke in the first run.
python - <<'EOF'
import importlib.metadata as md

for pkg in ("vllm", "torch", "torchvision", "torchaudio", "transformers", "peft"):
    print(f"{pkg}=={md.version(pkg)}")

import torch
import torchaudio  # noqa: F401  (raises on a CUDA-version mismatch with torch)
import torchvision  # noqa: F401  (same)

print("torch CUDA", torch.version.cuda, "| GPU visible:", torch.cuda.is_available())
assert torch.cuda.is_available(), "torch sees no GPU: driver too old for this torch?"

import peft  # noqa: F401
import vllm  # noqa: F401
from transformers import BloomPreTrainedModel  # noqa: F401

print("imports OK")
EOF

# The attention kernels (TRITON_ATTN) and the LoRA kernels are Triton kernels,
# compiled at run time; Triton links its launcher against libcuda. FlashInfer's
# equivalent step failed on this image after 26 minutes (`ld: cannot find -lcuda`).
# Compile and run one trivial Triton kernel now, so the same class of failure shows
# in seconds. Written to a file: @triton.jit reads the function's source, which a
# script on stdin does not have.
cat > /tmp/triton_check.py <<'EOF'
import torch
import triton
import triton.language as tl


@triton.jit
def add_one(x_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(x_ptr + offs, tl.load(x_ptr + offs, mask=mask) + 1, mask=mask)


x = torch.zeros(1000, device="cuda", dtype=torch.float16)
add_one[(triton.cdiv(1000, 256),)](x, 1000, BLOCK=256)
assert torch.all(x == 1), "Triton kernel ran but gave a wrong result"
print(f"triton=={triton.__version__} compiles and runs on", torch.cuda.get_device_name(0))
EOF
python /tmp/triton_check.py

echo "environment OK"
