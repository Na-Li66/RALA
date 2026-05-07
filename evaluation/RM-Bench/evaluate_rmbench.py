#!/usr/bin/env python3
"""RM-Bench evaluator for pairwise preference accuracy."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from peft import PeftModel
except Exception:
    PeftModel = None  # type: ignore


DEFAULT_VERIFICATION_PROMPT = """Based on the question and the response provided, is the response correct and complete?
Answer with only "Yes" or "No".

Answer:"""

STYLE_LABELS = ("concise", "detailed_plain", "detailed_markdown")
TABLE_DOMAINS = ("Chat", "Math", "Code", "Safety")


def set_offline_env(offline: bool) -> None:
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def model_device(model: Any) -> torch.device:
    return next(model.parameters()).device


def read_rows(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        data = json.loads(text)
        if isinstance(data, dict):
            data = data.get("data", [])
        return [dict(x) for x in data]
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_records(args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows = read_rows(args.data_path) if args.data_path else [dict(x) for x in load_dataset(args.dataset_name, args.dataset_config, split=args.split)]
    return rows[: args.max_examples] if args.max_examples > 0 else rows


def ensure_pad_token(tokenizer: Any) -> None:
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id


def as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    return [str(value)]


def record_responses(record: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    chosen = as_list(record.get("chosen", record.get("positive", record.get("winner"))))
    rejected = as_list(record.get("rejected", record.get("negative", record.get("loser"))))
    return chosen, rejected


def style_difficulty(chosen_index: int, rejected_index: int) -> str:
    if chosen_index < rejected_index:
        return "hard"
    if chosen_index > rejected_index:
        return "easy"
    return "normal"


def style_label(index: int) -> str:
    return STYLE_LABELS[index] if 0 <= index < len(STYLE_LABELS) else f"style_{index}"


def normalized_domain(record: Dict[str, Any]) -> str:
    value = record.get("domain", record.get("category", record.get("subset", "")))
    text = str(value).strip()
    lower = text.lower()
    if "safety" in lower:
        return "Safety"
    if "chat" in lower:
        return "Chat"
    if "math" in lower:
        return "Math"
    if "code" in lower:
        return "Code"
    return text or "Unknown"


def record_pairs(chosen: Sequence[str], rejected: Sequence[str], pairing: str) -> List[Dict[str, Any]]:
    if not chosen or not rejected:
        return []
    if pairing == "official":
        if len(chosen) != 3 or len(rejected) != 3:
            raise ValueError(
                "RM-Bench official protocol expects exactly three chosen and three rejected responses per record."
            )
        index_pairs = [(i, j) for i in range(3) for j in range(3)]
    elif pairing == "all":
        index_pairs = [(i, j) for i in range(len(chosen)) for j in range(len(rejected))]
    else:
        index_pairs = [(i, i) for i in range(min(len(chosen), len(rejected)))]

    return [
        {
            "chosen_index": i,
            "rejected_index": j,
            "chosen_style": style_label(i),
            "rejected_style": style_label(j),
            "difficulty": style_difficulty(i, j),
        }
        for i, j in index_pairs
    ]


def prompt_response_text(tokenizer: Any, prompt: str, response: str, use_chat_template: bool) -> Tuple[str, str]:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        prompt_text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        full_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
            tokenize=False,
            add_generation_prompt=False,
        )
        return prompt_text, full_text
    prompt_text = prompt.rstrip() + "\n"
    return prompt_text, prompt_text + response.lstrip()


def load_causal_model(args: argparse.Namespace, model_id: str, adapter_dir: str = "") -> tuple[Any, Any]:
    quant_config = None
    if args.load_in_4bit or args.load_in_8bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=args.trust_remote_code)
    ensure_pad_token(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    if adapter_dir:
        if PeftModel is None:
            raise RuntimeError("peft is required to load LoRA adapters.")
        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
    model.eval()
    return model, tokenizer


class EndoScorer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.model, self.tokenizer = load_causal_model(args, args.model_id, args.adapter_dir)
        self.use_chat_template = args.use_chat_template
        self.length_normalize = args.length_normalize

    @torch.no_grad()
    def score(self, prompt: str, response: str) -> float:
        prompt_text, full_text = prompt_response_text(self.tokenizer, prompt, response, self.use_chat_template)
        prompt_len = len(self.tokenizer(prompt_text, add_special_tokens=False).input_ids)
        enc = self.tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
        dev = model_device(self.model)
        input_ids = enc["input_ids"].to(dev)
        attention_mask = enc["attention_mask"].to(dev)
        if input_ids.shape[1] <= prompt_len:
            return float("-inf")
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits[:, :-1, :]
        labels = input_ids[:, 1:]
        logp = F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        response_logp = logp[:, max(0, prompt_len - 1) :]
        value = response_logp.sum().item()
        return float(value / max(1, response_logp.numel())) if self.length_normalize else float(value)


class GenScorer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.model, self.tokenizer = load_causal_model(args, args.gen_model_id or args.model_id, args.reward_gen_lora)
        self.use_chat_template = args.use_chat_template
        self.verification_prompt = args.verification_prompt
        self.yes_ids = self.one_token_ids(["Yes", " Yes", "YES"])
        self.no_ids = self.one_token_ids(["No", " No", "NO"])

    def one_token_ids(self, texts: Sequence[str]) -> List[int]:
        ids: List[int] = []
        for text in texts:
            toks = self.tokenizer(text, add_special_tokens=False).input_ids
            if len(toks) == 1:
                ids.append(int(toks[0]))
        return sorted(set(ids))

    def build_text(self, prompt: str, response: str) -> str:
        content = f"Question:\n{prompt}\n\nResponse:\n{response}\n\n{self.verification_prompt}"
        if self.use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        return content

    @torch.no_grad()
    def score(self, prompt: str, response: str) -> float:
        enc = self.tokenizer(self.build_text(prompt, response), return_tensors="pt", truncation=True)
        dev = model_device(self.model)
        logits = self.model(input_ids=enc["input_ids"].to(dev), attention_mask=enc["attention_mask"].to(dev), use_cache=False).logits[0, -1]
        probs = torch.softmax(logits.float(), dim=-1)
        yes = probs[self.yes_ids].sum() if self.yes_ids else torch.tensor(0.0, device=probs.device)
        no = probs[self.no_ids].sum() if self.no_ids else torch.tensor(0.0, device=probs.device)
        return float((yes / torch.clamp(yes + no, min=1e-12)).item())


class DiscScorer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.model, self.tokenizer = load_causal_model(args, args.disc_model_id or args.model_id, args.disc_adapter_dir)
        self.use_chat_template = args.use_chat_template
        self.head = self.load_head(args.disc_head).to(model_device(self.model)).eval()

    def load_head(self, path: str) -> nn.Linear:
        if not path:
            raise ValueError("--disc_head is required for --method disc or rala.")
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict) and "reward_head" in state:
            state = state["reward_head"]
        clean = {k.replace("module.", ""): v for k, v in state.items()}
        weight_key = "weight" if "weight" in clean else next(k for k in clean if k.endswith(".weight"))
        bias_key = "bias" if "bias" in clean else next((k for k in clean if k.endswith(".bias")), "")
        weight = clean[weight_key]
        if weight.ndim == 1:
            weight = weight.unsqueeze(0)
        head = nn.Linear(weight.shape[1], weight.shape[0], bias=bool(bias_key))
        head.weight.data.copy_(weight.float())
        if bias_key:
            head.bias.data.copy_(clean[bias_key].float())
        return head

    @torch.no_grad()
    def score(self, prompt: str, response: str) -> float:
        _, full_text = prompt_response_text(self.tokenizer, prompt, response, self.use_chat_template)
        enc = self.tokenizer(full_text, return_tensors="pt", truncation=True)
        dev = model_device(self.model)
        input_ids = enc["input_ids"].to(dev)
        attention_mask = enc["attention_mask"].to(dev)
        base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        inner = getattr(base, "model", base)
        hidden = inner(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True).last_hidden_state
        last = attention_mask.long().sum(dim=1).clamp(min=1) - 1
        pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
        return float(self.head(pooled.float()).squeeze(-1).item())


def zscore(values: Sequence[float]) -> List[float]:
    mean = sum(values) / len(values)
    std = math.sqrt(max(sum((x - mean) ** 2 for x in values) / len(values), 1e-12))
    return [(x - mean) / std for x in values]


def mean_present(values: Sequence[Any]) -> Any:
    present = [float(v) for v in values if v is not None]
    return sum(present) / len(present) if present else None


def bucket_summary(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(items)
    correct = sum(1 for r in items if r["correct"])
    out: Dict[str, Any] = {
        "total_pairs": total,
        "correct_pairs": correct,
        "accuracy": correct / total if total else None,
    }

    matrix_counts = [[0 for _ in STYLE_LABELS] for _ in STYLE_LABELS]
    matrix_correct = [[0 for _ in STYLE_LABELS] for _ in STYLE_LABELS]
    for item in items:
        i = item.get("chosen_index")
        j = item.get("rejected_index")
        if isinstance(i, int) and isinstance(j, int) and 0 <= i < len(STYLE_LABELS) and 0 <= j < len(STYLE_LABELS):
            matrix_counts[i][j] += 1
            matrix_correct[i][j] += int(bool(item["correct"]))

    matrix = []
    for i in range(len(STYLE_LABELS)):
        row = []
        for j in range(len(STYLE_LABELS)):
            row.append(matrix_correct[i][j] / matrix_counts[i][j] if matrix_counts[i][j] else None)
        matrix.append(row)

    out["style_order"] = list(STYLE_LABELS)
    out["accuracy_matrix"] = matrix
    out["hard"] = mean_present(matrix[i][j] for i in range(len(STYLE_LABELS)) for j in range(len(STYLE_LABELS)) if i < j)
    out["normal"] = mean_present(matrix[i][i] for i in range(len(STYLE_LABELS)))
    out["easy"] = mean_present(matrix[i][j] for i in range(len(STYLE_LABELS)) for j in range(len(STYLE_LABELS)) if i > j)
    return out


def summarize(results: List[Dict[str, Any]], group_fields: Sequence[str], *, total_records: int, skipped_records: int) -> Dict[str, Any]:
    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / len(results) if results else 0.0
    out: Dict[str, Any] = {
        "metric": "pairwise_preference_accuracy",
        "total_records": total_records,
        "evaluated_records": total_records - skipped_records,
        "skipped_records": skipped_records,
        "total_pairs": len(results),
        "correct_pairs": correct,
        "pairwise_preference_accuracy": accuracy,
        "accuracy": accuracy,
        "groups": {},
    }
    for field in group_fields:
        buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in results:
            if field in r["meta"]:
                buckets[str(r["meta"][field])].append(r)
        if buckets:
            out["groups"][field] = {
                k: {"total_pairs": len(v), "correct_pairs": sum(x["correct"] for x in v), "accuracy": sum(x["correct"] for x in v) / len(v)}
                for k, v in sorted(buckets.items())
            }

    domain_items: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for result in results:
        domain_items[result["domain"]].append(result)
    domain_scores = {domain: bucket_summary(domain_items[domain]) for domain in sorted(domain_items)}
    table = {domain: domain_scores[domain]["accuracy"] for domain in TABLE_DOMAINS if domain in domain_scores}
    table["Easy"] = mean_present(domain_scores[d].get("easy") for d in TABLE_DOMAINS if d in domain_scores)
    table["Normal"] = mean_present(domain_scores[d].get("normal") for d in TABLE_DOMAINS if d in domain_scores)
    table["Hard"] = mean_present(domain_scores[d].get("hard") for d in TABLE_DOMAINS if d in domain_scores)
    table["Avg"] = mean_present(domain_scores[d].get("accuracy") for d in TABLE_DOMAINS if d in domain_scores)
    out["rmbench_official"] = {
        "style_order": list(STYLE_LABELS),
        "all": bucket_summary(results),
        "domains": domain_scores,
        "table": table,
    }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate reward models on RM-Bench.")
    p.add_argument("--data_path", default="")
    p.add_argument("--dataset_name", default="THU-KEG/RM-Bench")
    p.add_argument("--dataset_config", default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--method", choices=["endo", "gen", "disc", "rala"], default="rala")
    p.add_argument("--model_id", required=True)
    p.add_argument("--adapter_dir", default="")
    p.add_argument("--disc_model_id", default="")
    p.add_argument("--disc_adapter_dir", default="")
    p.add_argument("--disc_head", default="")
    p.add_argument("--gen_model_id", default="")
    p.add_argument("--reward_gen_lora", default="")
    p.add_argument("--verification_prompt", default=DEFAULT_VERIFICATION_PROMPT)
    p.add_argument(
        "--pairing",
        choices=["official", "same_index", "all"],
        default="official",
        help="RM-Bench official uses all 3x3 style comparisons; other modes are for custom pair files.",
    )
    p.add_argument("--use_chat_template", action="store_true")
    p.add_argument("--length_normalize", action="store_true", default=True)
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--load_in_4bit", action="store_true", default=True)
    p.add_argument("--load_in_8bit", action="store_true")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--max_examples", type=int, default=0)
    p.add_argument("--group_fields", default="category,domain,difficulty,level,subset")
    p.add_argument("--output_json", default="")
    args = p.parse_args()
    if args.load_in_8bit:
        args.load_in_4bit = False
    set_offline_env(args.offline)

    scorers: Dict[str, Any] = {}
    if args.method in {"endo", "rala"}:
        scorers["endo"] = EndoScorer(args)
    if args.method in {"gen", "rala"}:
        scorers["gen"] = GenScorer(args)
    if args.method in {"disc", "rala"}:
        scorers["disc"] = DiscScorer(args)

    group_fields = [x.strip() for x in args.group_fields.split(",") if x.strip()]
    results: List[Dict[str, Any]] = []
    record_outputs: List[Dict[str, Any]] = []
    records = load_records(args)
    skipped_records = 0
    for ridx, record in enumerate(tqdm(records, desc="RM-Bench")):
        prompt = str(record.get("prompt", record.get("instruction", "")))
        chosen_responses, rejected_responses = record_responses(record)
        pairs = record_pairs(chosen_responses, rejected_responses, args.pairing)
        if not pairs:
            skipped_records += 1
            continue
        record_domain = normalized_domain(record)
        scored = {
            name: {
                "chosen": [scorer.score(prompt, response) for response in chosen_responses],
                "rejected": [scorer.score(prompt, response) for response in rejected_responses],
            }
            for name, scorer in scorers.items()
        }
        record_outputs.append(
            {
                "record_index": ridx,
                "id": record.get("id", record.get("ID", ridx)),
                "domain": record_domain,
                "component_scores": scored,
            }
        )
        for pidx, pair in enumerate(pairs):
            chosen_index = pair["chosen_index"]
            rejected_index = pair["rejected_index"]
            raw = {
                name: (scores["chosen"][chosen_index], scores["rejected"][rejected_index])
                for name, scores in scored.items()
            }
            if args.method == "rala":
                fused = [0.0, 0.0]
                for scores in raw.values():
                    z = zscore([scores[0], scores[1]])
                    fused[0] += z[0]
                    fused[1] += z[1]
                chosen_score, rejected_score = fused
            else:
                chosen_score, rejected_score = raw[args.method]
            results.append(
                {
                    "record_index": ridx,
                    "pair_index": pidx,
                    "chosen_index": chosen_index,
                    "rejected_index": rejected_index,
                    "chosen_style": pair["chosen_style"],
                    "rejected_style": pair["rejected_style"],
                    "difficulty": pair["difficulty"],
                    "domain": record_domain,
                    "correct": bool(chosen_score > rejected_score),
                    "score_chosen": chosen_score,
                    "score_rejected": rejected_score,
                    "component_scores": raw,
                    "meta": {
                        **{k: record.get(k) for k in group_fields if k in record},
                        "rmbench_domain": record_domain,
                        "rmbench_difficulty": pair["difficulty"],
                    },
                }
            )
    if not results:
        raise RuntimeError("No RM-Bench pairs were evaluated. Check the dataset schema and --pairing mode.")
    payload = {
        "summary": summarize(results, group_fields, total_records=len(records), skipped_records=skipped_records),
        "record_scores": record_outputs,
        "results": results,
    }
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
