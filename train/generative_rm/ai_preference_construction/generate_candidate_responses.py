#!/usr/bin/env python3
"""Generate SFT-policy candidates for generative RM preference data.

The script samples prompts, generates several candidate responses with the SFT
policy, and records an SFT-policy score for each candidate so the next step can
select the pair with maximum margin distance.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_DATA_PATH = os.environ.get("SFT_PROMPT_DATA", "")
DEFAULT_BASE_MODEL_DIR = os.environ.get("BASE_MODEL_DIR", os.environ.get("MODEL_ID", ""))
DEFAULT_OUT_PATH = os.environ.get("CANDIDATE_OUT_PATH", "")


def require_arg(value: str, cli_name: str, env_name: str) -> None:
    if str(value or "").strip():
        return
    raise SystemExit(
        f"Missing --{cli_name}. Set it explicitly or via ${env_name}."
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Generate candidate responses with the trained SFT policy.")
    p.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH)
    p.add_argument("--base_model_dir", type=str, default=DEFAULT_BASE_MODEL_DIR)
    p.add_argument("--sft_lora_dir", type=str, default=os.environ.get("SFT_LORA_DIR", ""))
    p.add_argument("--out_path", type=str, default=DEFAULT_OUT_PATH)
    p.add_argument("--num_prompts", type=int, default=7000)
    p.add_argument("--num_responses", type=int, default=5)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.92)
    p.add_argument("--max_prompt_tokens", type=int, default=1024)
    p.add_argument("--device_map", type=str, default="auto")
    p.add_argument("--trust_remote_code", action="store_true", default=True)
    p.add_argument("--local_files_only", action="store_true", default=True)
    p.add_argument("--allow_base_model_generation", action="store_true", default=False)
    return p


def load_prompts(path: str) -> List[str]:
    prompts: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            prompt = obj.get("prompt") or obj.get("instruction") or obj.get("question")
            if prompt is not None and str(prompt) != "":
                prompts.append(str(prompt))
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def resolve_device_map(raw: str) -> Any:
    raw = str(raw or "").strip()
    if raw.lower() in {"", "none", "null", "cpu"}:
        return None
    return raw


def load_sft_policy(args: argparse.Namespace):
    if not args.sft_lora_dir and not args.allow_base_model_generation:
        raise SystemExit(
            "Candidate construction requires --sft_lora_dir or SFT_LORA_DIR. "
            "Use --allow_base_model_generation only when the base model already contains the SFT policy."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_dir,
        use_fast=True,
        trust_remote_code=bool(args.trust_remote_code),
        local_files_only=bool(args.local_files_only),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device_map = resolve_device_map(args.device_map) if torch.cuda.is_available() else None
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_dir,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=bool(args.trust_remote_code),
        local_files_only=bool(args.local_files_only),
    )

    if args.sft_lora_dir:
        if not os.path.exists(args.sft_lora_dir):
            raise FileNotFoundError(f"SFT LoRA adapter not found: {args.sft_lora_dir}")
        model = PeftModel.from_pretrained(
            base_model,
            args.sft_lora_dir,
            torch_dtype=dtype,
            is_trainable=False,
        )
    else:
        model = base_model

    if device_map is None and torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    return model, tokenizer


@torch.no_grad()
def average_response_logprobs(
    model,
    sequences: torch.Tensor,
    *,
    context_length: int,
    pad_token_id: int,
) -> torch.Tensor:
    attention_mask = sequences.ne(int(pad_token_id)).long()
    out = model(input_ids=sequences, attention_mask=attention_mask, use_cache=False, return_dict=True)
    logits = out.logits[:, :-1, :]
    labels = sequences[:, 1:]
    selected = torch.gather(logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    normalizer = torch.logsumexp(logits, dim=-1)
    token_logp = (selected - normalizer).float()

    start = max(int(context_length) - 1, 0)
    response_logp = token_logp[:, start:]
    response_labels = labels[:, start:]
    response_mask = response_labels.ne(int(pad_token_id)).to(dtype=response_logp.dtype)
    return (response_logp * response_mask).sum(dim=1) / response_mask.sum(dim=1).clamp(min=1.0)


def generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    args: argparse.Namespace,
) -> Tuple[List[List[str]], List[List[float]]]:
    enc = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=int(args.max_prompt_tokens),
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}
    context_length = int(enc["input_ids"].size(1))

    with torch.no_grad():
        generated = model.generate(
            **enc,
            max_new_tokens=int(args.max_new_tokens),
            do_sample=True,
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            num_return_sequences=int(args.num_responses),
            pad_token_id=int(tokenizer.pad_token_id),
            eos_token_id=tokenizer.eos_token_id,
        )
        scores = average_response_logprobs(
            model,
            generated,
            context_length=context_length,
            pad_token_id=int(tokenizer.pad_token_id),
        )

    batch_size = len(prompts)
    generated = generated.view(batch_size, int(args.num_responses), -1)
    scores = scores.view(batch_size, int(args.num_responses))

    batch_responses: List[List[str]] = []
    batch_scores: List[List[float]] = []
    for i in range(batch_size):
        responses: List[str] = []
        for j in range(int(args.num_responses)):
            response_ids = generated[i, j, context_length:].tolist()
            responses.append(tokenizer.decode(response_ids, skip_special_tokens=True).strip())
        batch_responses.append(responses)
        batch_scores.append([float(x) for x in scores[i].detach().cpu().tolist()])
    return batch_responses, batch_scores


def main() -> None:
    args = build_arg_parser().parse_args()
    require_arg(args.data_path, "data_path", "SFT_PROMPT_DATA")
    require_arg(args.base_model_dir, "base_model_dir", "BASE_MODEL_DIR")
    require_arg(args.out_path, "out_path", "CANDIDATE_OUT_PATH")
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    torch.backends.cuda.matmul.allow_tf32 = True

    prompts = load_prompts(args.data_path)
    if int(args.num_prompts) > len(prompts):
        raise ValueError(f"num_prompts={args.num_prompts} exceeds dataset size={len(prompts)}")
    sampled = random.sample(prompts, int(args.num_prompts))

    model, tokenizer = load_sft_policy(args)
    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)

    with open(args.out_path, "w", encoding="utf-8") as out_f:
        for start in tqdm(range(0, len(sampled), int(args.batch_size)), desc="Generating candidates"):
            batch_prompts = sampled[start : start + int(args.batch_size)]
            responses_batch, scores_batch = generate_batch(model, tokenizer, batch_prompts, args)
            for prompt, responses, scores in zip(batch_prompts, responses_batch, scores_batch):
                record: Dict[str, Any] = {
                    "prompt": prompt,
                    "responses": responses,
                    "sft_logprob_scores": scores,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()

    print(f"Finished generating SFT-policy candidates to {args.out_path}")


if __name__ == "__main__":
    main()
