"""LoRA fine-tuning and scoring helpers — the GPU half of this project.

The model-facing functions here need `torch`, `transformers` and `peft` (the `train`
extra in `pyproject.toml`) and only run on a GPU machine, via
`scripts/train_and_evaluate.py`. Those imports are inside the functions, not at
module level, so this module imports without them — which is what lets the pure
helpers at the top (`choose_dtype_name`, `full_run_steps`, `load_scored`) be
unit-tested on a laptop with no GPU.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

#: Attention and MLP projection layers LoRA adapts, in Qwen2's naming. Adapting all
#: seven rather than only the attention projections costs little extra memory and
#: is the configuration most fine-tuning guides for this model family report as a
#: better tradeoff than attention-only.
DEFAULT_TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


@dataclass(frozen=True)
class LoraTrainingConfig:
    base_model: str = DEFAULT_BASE_MODEL
    target_modules: tuple[str, ...] = DEFAULT_TARGET_MODULES
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_seq_length: int = 1024
    seed: int = 0


# --- Pure helpers (no torch needed; unit-tested) -----------------------------------


def choose_dtype_name(compute_capability: tuple[int, int]) -> str:
    """"bfloat16" on Ampere (8.x) or newer, "float16" otherwise.

    The reason this function exists: the first real run (EXPERIMENTS.md, E2a) used
    bf16 on a Kaggle T4. The T4 is Turing, compute capability 7.5, with no native
    bf16 support. cuBLAS rejected the bf16 matmul (`CUBLAS_STATUS_EXECUTION_FAILED
    ... abType 14`, and 14 is `CUDA_R_16BF`) and every forward pass fell back to a
    slow path, about 10x slower than expected. fp16 is what the T4's tensor cores
    actually run.
    """
    major, _minor = compute_capability
    return "bfloat16" if major >= 8 else "float16"


def full_run_steps(n_train: int, config: LoraTrainingConfig) -> int:
    """Optimizer steps in a full run, as `Trainer` counts them (ceil at each stage).

    Checked against the first real run: 5,594 examples, batch 4, accumulation 4,
    2 epochs → `Trainer` reported 700 steps; so does this.
    """
    micro_batches = math.ceil(n_train / config.per_device_train_batch_size)
    per_epoch = math.ceil(micro_batches / config.gradient_accumulation_steps)
    return per_epoch * config.num_train_epochs


def load_scored(path: Path) -> dict[str, dict]:
    """Already-scored test examples from an append-only JSONL file, keyed by id.

    Evaluation writes one line per example as it goes, so an interrupted evaluation
    resumes where it stopped instead of starting over. A truncated last line (the
    process died mid-write) is skipped, not fatal.
    """
    done: dict[str, dict] = {}
    if not path.exists():
        return done
    with path.open() as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[rec["trajectory_id"]] = rec
    return done


# --- Model-facing functions (need torch/transformers/peft; GPU) --------------------


def load_model_and_tokenizer(config: LoraTrainingConfig, dtype_name: str):
    """The base model in `dtype_name` (no quantization) plus its tokenizer.

    No 4-bit quantization: a 1.5B model is ~3 GB in 16-bit and fits a 16 GB T4 with
    room to spare. Single-device placement (`device_map={"": 0}`): with two GPUs
    visible, `Trainer` wraps the model in `DataParallel`, which crashed the first
    run with a CUDA "illegal memory access". `scripts/train_and_evaluate.py` also
    sets `CUDA_VISIBLE_DEVICES=0` before torch is imported.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, dtype_name)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        # eos as pad is the standard fallback for causal LMs without a pad token.
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model, dtype=dtype, device_map={"": 0}
        )
    except TypeError:  # transformers < 4.56 names it torch_dtype
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model, torch_dtype=dtype, device_map={"": 0}
        )
    if model.dtype != dtype:  # an ignored kwarg would silently load fp32
        model = model.to(dtype)
    model.config.use_cache = False  # incompatible with training; re-enabled for eval
    return model, tokenizer


def build_lora_model(model, config: LoraTrainingConfig):
    """Wrap `model` with a LoRA adapter whose trainable weights are fp32.

    fp32 adapter weights are required for fp16 mixed-precision training: the
    gradient scaler refuses to unscale fp16 gradients. The frozen base stays in
    16-bit. No gradient checkpointing: activations for a 1.5B model at this batch
    size fit on a T4, and recomputing the forward pass would cost speed for memory
    that is not short.
    """
    import torch
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    for param in model.parameters():
        if param.requires_grad and param.dtype != torch.float32:
            param.data = param.data.float()
    return model


def to_chat_text(tokenizer, prompt: str, response: str | None) -> str:
    """One example as chat-template text.

    `response=None` renders only the prompt with a generation cue appended (for
    inference); a real response renders the full turn plus the tokenizer's EOS, so
    the model is trained to stop rather than to keep generating past the label.
    """
    messages = [{"role": "user", "content": prompt}]
    if response is None:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    messages.append({"role": "assistant", "content": response})
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    ).rstrip()
    if not text.endswith(tokenizer.eos_token):
        text += tokenizer.eos_token
    return text


def tokenize_example(tokenizer, prompt: str, response: str, max_seq_length: int) -> dict:
    """One (prompt, response) pair as `input_ids` / `attention_mask` / `labels`.

    Loss is masked to the response tokens (`labels = -100` on every prompt token)
    by hand, with plain `transformers.Trainer`: `trl`'s
    `DataCollatorForCompletionOnlyLM` no longer exists in trl 1.14.1 (the version
    installed on Kaggle, 29 sep 2026), and masking labels directly does not depend
    on trl's API.
    """
    prompt_text = to_chat_text(tokenizer, prompt, None)
    full_text = to_chat_text(tokenizer, prompt, response)

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    n_prompt_tokens = len(prompt_ids)
    labels = [-100] * n_prompt_tokens + full_ids[n_prompt_tokens:]

    # Truncate from the front, not the back: the label tokens are at the end, and
    # losing them would silently train the model on nothing.
    if len(full_ids) > max_seq_length:
        full_ids = full_ids[-max_seq_length:]
        labels = labels[-max_seq_length:]

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def build_tokenized_dataset(tokenizer, examples: list[dict], max_seq_length: int):
    """A `datasets.Dataset` of tokenized, label-masked rows, ready for `Trainer`."""
    from datasets import Dataset

    rows = [
        tokenize_example(tokenizer, ex["prompt"], ex["response"], max_seq_length)
        for ex in examples
    ]
    return Dataset.from_list(rows)


class CausalLMLabelCollator:
    """Pads `input_ids` / `attention_mask` / `labels` to the batch's max length.

    `labels` are padded with -100 (ignored by the loss), not with `pad_token_id`,
    which would otherwise train the model to predict the pad token.
    """

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]):
        import torch

        max_len = max(len(f["input_ids"]) for f in features)

        def pad(seq, value):
            return list(seq) + [value] * (max_len - len(seq))

        return {
            "input_ids": torch.tensor([pad(f["input_ids"], self.pad_token_id) for f in features]),
            "attention_mask": torch.tensor([pad(f["attention_mask"], 0) for f in features]),
            "labels": torch.tensor([pad(f["labels"], -100) for f in features]),
        }


def sequence_logprob(model, tokenizer, prompt: str, completion: str) -> float:
    """Sum of token log-probabilities of `completion` given `prompt`, teacher-forced.

    Used to score RESOLVED vs UNRESOLVED continuously (`metrics.auc` and
    `metrics.paired_separation` need a score, not just a label), rather than relying
    on free generation matching the exact trained wording.
    """
    import torch

    prompt_text = to_chat_text(tokenizer, prompt, None)
    full_text = prompt_text + completion + tokenizer.eos_token

    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
    device = next(model.parameters()).device
    input_ids = full_ids["input_ids"].to(device)

    n_prompt_tokens = prompt_ids["input_ids"].shape[1]
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[0]  # (seq_len, vocab)

    # logits[i] predicts token i+1: the completion's own tokens start at
    # n_prompt_tokens, so their predicting logits start one position earlier.
    completion_ids = input_ids[0, n_prompt_tokens:]
    completion_logits = logits[n_prompt_tokens - 1 : -1]
    log_probs = torch.log_softmax(completion_logits.float(), dim=-1)
    token_log_probs = log_probs[torch.arange(len(completion_ids)), completion_ids]
    return float(token_log_probs.sum().item())


def resolved_score(model, tokenizer, prompt: str) -> float:
    """log P(RESOLVED | prompt) - log P(UNRESOLVED | prompt).

    Positive when the model favours RESOLVED; comparable across examples, and does
    not depend on the model emitting a well-formed one-word answer.
    """
    from lora_appraisal_judge.prompts import RESOLVED_LABEL, UNRESOLVED_LABEL

    return sequence_logprob(model, tokenizer, prompt, RESOLVED_LABEL) - sequence_logprob(
        model, tokenizer, prompt, UNRESOLVED_LABEL
    )


def generate_prediction(model, tokenizer, prompt: str, max_new_tokens: int = 8) -> str:
    """Greedy free-generation continuation, for `prompts.parse_prediction`.

    Reported alongside `resolved_score` rather than instead of it: this is what a
    real deployment would read, `resolved_score` is what makes the metrics honest.
    """
    text = to_chat_text(tokenizer, prompt, None)
    inputs = tokenizer(text, return_tensors="pt").to(next(model.parameters()).device)
    output = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    new_tokens = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)
