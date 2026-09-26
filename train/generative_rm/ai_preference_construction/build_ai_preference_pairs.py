#!/usr/bin/env python3
"""Build generative RM preference pairs from max-margin candidates."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_IN_PATH = os.environ.get("PAIR_INPUT_PATH", os.environ.get("MAX_MARGIN_PAIR_OUTPUT", ""))
DEFAULT_OUT_PATH = os.environ.get("PREF_PAIR_OUTPUT_PATH", "")
DEFAULT_REVIEW_MODEL_PATH = os.environ.get("REVIEW_MODEL_PATH", "")

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def require_arg(value: str, cli_name: str, env_name: str) -> None:
    if str(value or "").strip():
        return
    raise SystemExit(
        f"Missing --{cli_name}. Set it explicitly or via ${env_name}."
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Build AI-labeled preference pairs for generative RM training.")
    p.add_argument("--input_path", type=str, default=DEFAULT_IN_PATH)
    p.add_argument("--output_path", type=str, default=DEFAULT_OUT_PATH)
    p.add_argument("--review_model_path", type=str, default=DEFAULT_REVIEW_MODEL_PATH)
    p.add_argument("--max_length", type=int, default=8192)
    p.add_argument("--device_map", type=str, default="auto")
    p.add_argument("--local_files_only", action="store_true", default=True)
    p.add_argument("--trust_remote_code", action="store_true", default=True)
    return p


def _pick_attn_impl() -> str:
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except Exception:
        return "eager"


def _pick_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def _build_max_memory() -> Optional[Dict[int, str]]:
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return None
    max_mem: Dict[int, str] = {}
    for i in range(torch.cuda.device_count()):
        total = torch.cuda.get_device_properties(i).total_memory
        reserve = 2 * 1024**3
        allowed = max(int(total * 0.85), int(total - reserve))
        gib = max(1, allowed // (1024**3))
        max_mem[i] = f"{gib}GiB"
    return max_mem


def _first_device_of_model(model: torch.nn.Module) -> torch.device:
    for param in model.parameters():
        return param.device
    return torch.device("cpu")


def load_judge_model(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(
        args.review_model_path,
        trust_remote_code=bool(args.trust_remote_code),
        local_files_only=bool(args.local_files_only),
        use_fast=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.review_model_path,
        torch_dtype=_pick_dtype(),
        trust_remote_code=bool(args.trust_remote_code),
        local_files_only=bool(args.local_files_only),
        low_cpu_mem_usage=True,
        device_map=str(args.device_map),
        max_memory=_build_max_memory(),
        attn_implementation=_pick_attn_impl(),
    )
    model.eval()
    first_device = _first_device_of_model(model)
    print(f"[Device Map] {getattr(model, 'hf_device_map', 'single device: ' + str(first_device))}")
    return tokenizer, model, first_device


def render_prompt(prompt_text: str, cand1: str, cand2: str) -> str:
    return (
        "You are a careful evaluator. Given a TASK and two CANDIDATES, "
        "decide which candidate better satisfies the task. Consider correctness, completeness, "
        "faithfulness to the task requirements, clarity, and overall quality.\n"
        "Reply with a single character: 1 if CANDIDATE 1 is better, otherwise 2.\n\n"
        f"TASK:\n{prompt_text}\n\n"
        f"CANDIDATE 1:\n{cand1}\n\n"
        f"CANDIDATE 2:\n{cand2}\n\n"
        "Answer: "
    )


def _candidate_ids_for_digit(tokenizer: Any, digit: str) -> List[int]:
    variants = [digit, " " + digit, "\n" + digit, digit + ".", " " + digit + ".", "\n" + digit + "."]
    ids = set()
    for variant in variants:
        token_ids = tokenizer.encode(variant, add_special_tokens=False)
        if token_ids:
            ids.add(token_ids[-1])
    return sorted(ids)


@torch.inference_mode()
def next_token_choice_probs(
    tokenizer: Any,
    model: torch.nn.Module,
    first_device: torch.device,
    rendered_text: str,
    max_length: int,
) -> Tuple[float, float]:
    inputs = tokenizer(
        rendered_text,
        return_tensors="pt",
        truncation=True,
        max_length=int(max_length),
    )
    inputs = {k: v.to(first_device) for k, v in inputs.items()}
    out = model(**inputs)

    logits = out.logits[:, -1, :].to(torch.float32)
    log_probs = torch.log_softmax(logits, dim=-1)[0]

    ids_1 = _candidate_ids_for_digit(tokenizer, "1")
    ids_2 = _candidate_ids_for_digit(tokenizer, "2")
    p1 = torch.logsumexp(log_probs[ids_1], dim=0).exp().item() if ids_1 else 0.0
    p2 = torch.logsumexp(log_probs[ids_2], dim=0).exp().item() if ids_2 else 0.0

    total = p1 + p2
    if total <= 0:
        return 0.5, 0.5
    return p1 / total, p2 / total


def rlaif_pair_preference(
    tokenizer: Any,
    model: torch.nn.Module,
    first_device: torch.device,
    prompt_text: str,
    cand_a: str,
    cand_b: str,
    max_length: int,
) -> Tuple[str, str]:
    p1_ab, p2_ab = next_token_choice_probs(
        tokenizer, model, first_device, render_prompt(prompt_text, cand_a, cand_b), max_length
    )
    p1_ba, p2_ba = next_token_choice_probs(
        tokenizer, model, first_device, render_prompt(prompt_text, cand_b, cand_a), max_length
    )
    p_a = 0.5 * (p1_ab + p2_ba)
    p_b = 1.0 - p_a
    return (cand_a, cand_b) if p_a >= p_b else (cand_b, cand_a)


def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        if first == "[":
            f.seek(0)
            for obj in json.load(f):
                yield obj
            return

        f.seek(0)
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    args = build_arg_parser().parse_args()
    require_arg(args.input_path, "input_path", "PAIR_INPUT_PATH")
    require_arg(args.output_path, "output_path", "PREF_PAIR_OUTPUT_PATH")
    require_arg(args.review_model_path, "review_model_path", "REVIEW_MODEL_PATH")

    tokenizer, model, first_device = load_judge_model(args)
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    total, ok = 0, 0
    with open(args.output_path, "w", encoding="utf-8") as writer:
        for obj in iter_jsonl(args.input_path):
            total += 1
            try:
                prompt = str(obj["prompt"])
                response_1, response_2 = obj["responses"]
                chosen, rejected = rlaif_pair_preference(
                    tokenizer,
                    model,
                    first_device,
                    prompt,
                    str(response_1),
                    str(response_2),
                    int(args.max_length),
                )
                writer.write(
                    json.dumps({"prompt": prompt, "chosen": chosen, "rejected": rejected}, ensure_ascii=False) + "\n"
                )
                ok += 1
            except Exception as exc:
                print(f"[ERROR] line {total} failed: {exc}", file=sys.stderr)

            if total % 50 == 0:
                print(f"[Progress] processed={total}, ok={ok}")

    print(f"[DONE] Total={total}, Success={ok}, Output={args.output_path}")


if __name__ == "__main__":
    main()
