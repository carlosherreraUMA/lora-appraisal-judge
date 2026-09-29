"""QLoRA fine-tuning and scoring helpers — the GPU half of this project.

Everything in this module needs `torch`, `transformers`, `peft` and `bitsandbytes`
(the `train` extra in `pyproject.toml`), so it is meant to run on Kaggle
(`notebooks/kaggle_train.ipynb`), not in the same environment as the rest of the
package's tests. Kept separate from `extract.py` / `prompts.py` / `baselines.py` /
`metrics.py` / `splits.py` so those stay importable and testable with no GPU and no
heavy dependencies at all.

Not unit-tested in this repository for that reason — there is no GPU in CI for it to
run against. The honest substitute is the notebook itself, whose output (including
the metrics from `metrics.py`, which *are* tested) is saved and versioned as
`notebooks/kaggle_train_results.md` after each real run, listed in `EXPERIMENTS.md`.
"""
from __future__ import annotations

from dataclasses import dataclass

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

#: Attention and MLP projection layers LoRA adapts, in Qwen2's naming. Adapting all
#: seven rather than only the attention projections costs little extra memory in
#: 4-bit and is the configuration most fine-tuning guides for this model family
#: report as a better tradeoff than attention-only.
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
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 1024
    seed: int = 0


def load_quantized_model_and_tokenizer(config: LoraTrainingConfig):
    """The base model in 4-bit (QLoRA) plus its tokenizer, ready for `peft`.

    Import of `torch`/`transformers`/`bitsandbytes` is inside this function, not at
    module level, so importing `lora_appraisal_judge.training` itself does not
    require a GPU machine — only calling this function does.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        # Qwen2.5's tokenizer ships without a pad token; padding is needed for
        # batched training. eos as pad is the standard fallback for causal LMs.
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=bnb_config,
        # Explicit single-device placement, not "auto": on a machine with more than
        # one GPU visible (e.g. Kaggle's "T4 x2" accelerator), `device_map="auto"`'s
        # sharding combined with `Trainer` wrapping the model in `DataParallel` for
        # the extra GPU corrupts bitsandbytes' 4-bit quantization state, surfacing as
        # a CUDA "illegal memory access" during the forward pass. A 1.5B model in
        # 4-bit fits in one T4's 16 GB with room to spare, so there is no need for
        # more than one device. Restrict visible GPUs to one at the process level too
        # (`CUDA_VISIBLE_DEVICES`, set in the notebook before this import) so
        # `Trainer` never sees a second GPU to wrap around in the first place.
        device_map={"": 0},
    )
    return model, tokenizer


def build_lora_model(model, config: LoraTrainingConfig):
    """Wrap `model` with a LoRA adapter, prepared for k-bit training."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_config)


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

    Loss is masked to the response tokens (`labels = -100` on every prompt token),
    using plain `transformers` rather than `trl`'s `SFTTrainer` /
    `DataCollatorForCompletionOnlyLM`: that class was removed by trl 1.14 (checked
    against the actual installed version on Kaggle, 29 sep 2026 — `trl.__version__ ==
    "1.14.1"`, no `Collator`/`Completion` name left in `trl` or `trl.trainer`), and
    trl's SFT API has moved enough times that pinning to one snapshot of it is more
    fragile than masking labels by hand, which `transformers.Trainer` has supported
    unchanged for years.
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

    Written out directly, rather than reached for from a library, so there is no
    ambiguity about which version's padding defaults apply — the same reasoning as
    dropping the `trl` collator above. `labels` are padded with -100 (ignored by the
    loss), not with `pad_token_id`, which would otherwise train the model to predict
    the pad token.
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
    """Sum of token log-probabilities of `completion` given `prompt`, under teacher
    forcing. Used to score RESOLVED vs UNRESOLVED continuously (see module docstring
    in `metrics.py`: `auc` and `paired_separation` need a score, not just a label),
    rather than relying on free-generation matching the exact trained wording.
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

    A continuous score, positive when the model favours RESOLVED, comparable across
    examples for `metrics.auc` and `metrics.paired_separation` — unlike parsing free
    generation, this does not depend on the model actually emitting a well-formed
    one-word answer.
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
