"""Scoring with vLLM: the same score as `training.resolved_score`, batched.

E2b scored the test split one example at a time with Hugging Face `transformers`
(0.93 s/example on a T4: three forward passes each, no batching). That is fine for
1,717 examples and not for the full-corpus test set of E3. vLLM serves the base
model with the LoRA adapter applied at inference time and schedules many requests
together on the GPU.

The score must be the same number, not a similar one, or E3 would not be comparable
with E2b. So the token ids are built exactly as `training.sequence_logprob` builds
them (same chat template, prompt and full text tokenized separately, completion
starting at `len(prompt_ids)`) and handed to vLLM as token ids, not text. vLLM's
`prompt_logprobs` then gives the log-probability of each of those tokens, which is
what the teacher-forced forward pass computed. `scripts/evaluate_vllm.py` checks the
agreement against E2b's saved scores before trusting anything else.

The pure helpers are tested without vLLM or a GPU; vLLM is imported inside the
model-facing functions only.
"""
from __future__ import annotations

from dataclasses import dataclass

from lora_appraisal_judge.prompts import RESOLVED_LABEL, UNRESOLVED_LABEL, parse_prediction
from lora_appraisal_judge.training import to_chat_text


@dataclass(frozen=True)
class ScoringRequest:
    """Token ids for one example: the generation prompt and both teacher-forced texts."""

    trajectory_id: str
    prompt_ids: list[int]
    resolved_ids: list[int]
    unresolved_ids: list[int]
    #: Where the completion starts in `resolved_ids` / `unresolved_ids`.
    n_prompt_tokens: int


def build_request(tokenizer, example: dict) -> ScoringRequest:
    """The token ids `training.sequence_logprob` would score for this example."""
    prompt_text = to_chat_text(tokenizer, example["prompt"], None)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    def full(label: str) -> list[int]:
        text = prompt_text + label + tokenizer.eos_token
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    return ScoringRequest(
        trajectory_id=example["trajectory_id"],
        prompt_ids=prompt_ids,
        resolved_ids=full(RESOLVED_LABEL),
        unresolved_ids=full(UNRESOLVED_LABEL),
        n_prompt_tokens=len(prompt_ids),
    )


def completion_logprob(prompt_logprobs: list, token_ids: list[int], start: int) -> float:
    """Sum of the log-probabilities of `token_ids[start:]`, from vLLM `prompt_logprobs`.

    `prompt_logprobs[i]` is vLLM's per-position dict {token_id: Logprob} for prompt
    token i (None at position 0, which has no context). A missing entry for the
    actual token raises instead of being skipped: a silently partial sum would be a
    wrong score that still looks like a number.
    """
    total = 0.0
    for i in range(start, len(token_ids)):
        entry = prompt_logprobs[i]
        tok = token_ids[i]
        if entry is None or tok not in entry:
            raise KeyError(f"no logprob for prompt token {tok} at position {i}")
        lp = entry[tok]
        total += float(getattr(lp, "logprob", lp))
    return total


def agreement(reference: dict[str, dict], new: dict[str, dict], ids: list[str]) -> dict:
    """How closely `new` reproduces `reference` on the same examples.

    Both map trajectory_id -> {score, generated}, the format of `test_scores.jsonl`.
    Rank correlation is what matters for AUC and paired separation; the absolute
    differences and sign agreement say how far fp16 rounding moved individual scores.
    """
    import numpy as np
    from scipy.stats import pearsonr, spearmanr

    ref = np.array([reference[i]["score"] for i in ids], dtype=float)
    cur = np.array([new[i]["score"] for i in ids], dtype=float)
    diff = np.abs(ref - cur)
    same_label = [
        parse_prediction(reference[i]["generated"]) == parse_prediction(new[i]["generated"])
        for i in ids
    ]
    return {
        "spearman": float(spearmanr(ref, cur).statistic),
        "pearson": float(pearsonr(ref, cur).statistic),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "sign_agreement": float(np.mean(np.sign(ref) == np.sign(cur))),
        "generated_agreement": float(np.mean(same_label)),
    }


def label_distribution(examples: list[dict], preds: list[bool | None]) -> dict:
    """Generated labels by true label: where does the model's own decision fall?

    E2b's reading (point 2) left this open: the generated label was at chance while
    the score ranked, which points at a badly placed decision threshold.
    """
    def name(p):
        return {True: RESOLVED_LABEL, False: UNRESOLVED_LABEL, None: "unparsed"}[p]

    out: dict[str, dict[str, int]] = {"resolved": {}, "unresolved": {}}
    for ex, p in zip(examples, preds):
        row = out["resolved" if ex["resolved"] else "unresolved"]
        row[name(p)] = row.get(name(p), 0) + 1
    return out


# --- Model-facing (needs vllm and a GPU) ------------------------------------------


def load_llm(model_path: str, dtype_name: str, lora_rank: int | None, max_model_len: int):
    """A vLLM engine for `model_path`, with LoRA enabled when `lora_rank` is given.

    `dtype_name` comes from `training.choose_dtype_name`: float16 on a T4, which has
    no native bf16 (EXPERIMENTS.md, E2a), whatever the model's config says.
    """
    from vllm import LLM

    kwargs = dict(
        model=model_path,
        dtype=dtype_name,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.85,
        seed=0,
    )
    if lora_rank is not None:
        kwargs.update(enable_lora=True, max_lora_rank=lora_rank, max_loras=1)
    return LLM(**kwargs)


def generation_repetition_penalty(model_path: str) -> float:
    """The repetition penalty Hugging Face `generate` applied in E2b.

    `model.generate(do_sample=False)` still applies the model's `generation_config`,
    and Qwen2.5-1.5B-Instruct's sets `repetition_penalty=1.1` (checked 6 oct 2026;
    temperature, top-k and top-p do not apply to greedy decoding; the penalty does). vLLM applies none unless asked,
    so the generated labels would not be comparable without this.
    """
    from transformers import GenerationConfig

    try:
        return float(GenerationConfig.from_pretrained(model_path).repetition_penalty or 1.0)
    except OSError:  # no generation_config.json
        return 1.0


def score_batch(llm, requests: list[ScoringRequest], lora_request=None,
                max_new_tokens: int = 8, repetition_penalty: float = 1.0) -> list[dict]:
    """Score and greedily generate for a batch, in one call to the engine.

    Three requests per example: the two teacher-forced texts (one token generated,
    only the prompt log-probabilities are read; they come from the raw logits, so
    sampling settings do not touch the score) and the generation prompt (greedy, as
    `training.generate_prediction`, with the same repetition penalty). Returns
    records shaped like E2b's `test_scores.jsonl`: {trajectory_id, score, generated}.
    """
    from vllm import SamplingParams

    teacher = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1)
    greedy = SamplingParams(max_tokens=max_new_tokens, temperature=0.0,
                            repetition_penalty=repetition_penalty)

    prompts, params = [], []
    for r in requests:
        prompts += [{"prompt_token_ids": r.resolved_ids},
                    {"prompt_token_ids": r.unresolved_ids},
                    {"prompt_token_ids": r.prompt_ids}]
        params += [teacher, teacher, greedy]

    outputs = llm.generate(prompts, params, lora_request=lora_request, use_tqdm=False)

    records = []
    for k, r in enumerate(requests):
        out_r, out_u, out_g = outputs[3 * k: 3 * k + 3]
        lp_r = completion_logprob(out_r.prompt_logprobs, r.resolved_ids, r.n_prompt_tokens)
        lp_u = completion_logprob(out_u.prompt_logprobs, r.unresolved_ids, r.n_prompt_tokens)
        records.append({
            "trajectory_id": r.trajectory_id,
            "score": lp_r - lp_u,
            "generated": out_g.outputs[0].text,
        })
    return records


def merge_adapter(base_model: str, adapter_dir: str, out_dir: str, dtype_name: str) -> None:
    """Fallback if vLLM's LoRA path fails on this GPU: bake the adapter into the weights.

    Done on CPU, before the engine starts, so it does not compete for GPU memory.
    The merged weights are rounded to 16-bit once, so scores can differ slightly from
    the unmerged adapter; the agreement check against E2b measures by how much.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, dtype_name)
    try:
        base = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype)
    except TypeError:  # transformers < 4.56 names it torch_dtype
        base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype)
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    merged.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(base_model).save_pretrained(out_dir)
