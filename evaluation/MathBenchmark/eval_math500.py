#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MATH-500 evaluation script for Qwen2.5-Math(-Instruct) style models.

Required CLI args:
  --model_id   (local model path)
  --lora_dir   (optional; can be empty string)
  --output_json (path to write results json)

Default dataset path:
  data/math_500/test.jsonl

Expected dataset format (JSONL): each line contains at least:
  - problem (or question/Problem)
  - answer  (or Answer)

This script evaluates MATH-500 pass@1 using symbolic equivalence grading.
"""

import argparse
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception as e:
    raise RuntimeError(
        "transformers is required. Please install the exact versions recommended by Qwen2.5-Math evaluation "
        "(see QwenLM/Qwen2.5-Math repo). Original error: %s" % (e,)
    )

try:
    from peft import PeftModel  # type: ignore
except Exception:
    PeftModel = None  # type: ignore

# Sympy is used for equivalence checking (MATH-500)
try:
    import sympy as sp
except Exception as e:
    raise RuntimeError("sympy is required for MATH-500 grading. Original error: %s" % (e,))

# latex2sympy is used by the official Qwen2.5-Math evaluation (adapted from math-evaluation-harness)
# Qwen repo provides a local latex2sympy package; you can also install latex2sympy2 from pip.
latex2sympy_fn = None
try:
    from latex2sympy2 import latex2sympy as _latex2sympy  # type: ignore
    latex2sympy_fn = _latex2sympy
except Exception:
    try:
        from latex2sympy import latex2sympy as _latex2sympy  # type: ignore
        latex2sympy_fn = _latex2sympy
    except Exception:
        latex2sympy_fn = None


QWEN25_MATH_COT_SYSTEM = "Please reason step by step, and put your final answer within \\\\boxed{}."
QWEN25_MATH_TIR_SYSTEM = (
    "Please integrate natural language reasoning with programs to solve the problem above, "
    "and put your final answer within \\\\boxed{}."
)


@dataclass
class GenConfig:
    max_new_tokens: int = 3072
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0


def _first_param_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def load_model_and_tokenizer(model_id: str, lora_dir: str = ""):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_fast=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    ).eval()

    if lora_dir:
        if PeftModel is None:
            raise RuntimeError("peft is required when lora_dir is provided. Please `pip install peft`.")
        if not os.path.isdir(lora_dir):
            raise RuntimeError(f"lora_dir does not exist or is not a directory: {lora_dir}")
        model = PeftModel.from_pretrained(model, lora_dir).eval()

    return model, tokenizer


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def normalize_latex(s: str) -> str:
    # Normalization aligned with common math-eval harness logic + known Qwen issues:
    # - strip spaces
    # - remove $...$
    # - remove \left \right
    # - remove common latex spacing commands
    # - strip trailing punctuation
    if s is None:
        return ""
    s = str(s).strip()
    if len(s) >= 2 and s[0] == "$" and s[-1] == "$":
        s = s[1:-1].strip()
    for tok in ["\\left", "\\right", "\\!", "\\,", "\\;", "\\:", "\\qquad", "\\quad"]:
        s = s.replace(tok, "")
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    # Remove trailing period/comma
    s = re.sub(r"[\\.,;:]$", "", s).strip()
    return s


def extract_boxed(text: str) -> Optional[str]:
    """
    Extract the content of the *last* \\boxed{...} in the text.
    Handles nested braces.
    """
    if not text:
        return None
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None

    # Find first '{' after \boxed
    brace_start = text.find("{", idx)
    if brace_start == -1:
        return None

    depth = 0
    for i in range(brace_start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start + 1 : i].strip()
    return None


def extract_final_answer(text: str) -> Optional[str]:
    """
    Prefer \\boxed{...}. Fallback to common patterns; finally return the last non-empty line.
    """
    if not text:
        return None

    boxed = extract_boxed(text)
    if boxed is not None and boxed != "":
        return boxed

    # Common patterns
    patterns = [
        r"Final answer\s*[:：]\s*(.+)",
        r"Answer\s*[:：]\s*(.+)",
        r"答案\s*[:：]\s*(.+)",
    ]
    for pat in patterns:
        m = re.findall(pat, text, flags=re.IGNORECASE)
        if m:
            cand = m[-1].strip()
            # take until end of line
            cand = cand.splitlines()[0].strip()
            return cand

    # last non-empty line
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line:
            return line
    return None


def _sympy_equals(a: Any, b: Any) -> bool:
    # Recursive structural equality + symbolic equivalence
    if a == b:
        return True

    # Handle tuples
    if isinstance(a, (tuple, list)) or isinstance(b, (tuple, list)):
        if not (isinstance(a, (tuple, list)) and isinstance(b, (tuple, list))):
            return False
        if len(a) != len(b):
            return False
        return all(_sympy_equals(x, y) for x, y in zip(a, b))

    # Handle Sympy Matrix
    if isinstance(a, sp.MatrixBase) and isinstance(b, sp.MatrixBase):
        try:
            return (a - b).is_zero_matrix
        except Exception:
            return False

    # Handle Sympy Set
    if isinstance(a, sp.Set) and isinstance(b, sp.Set):
        try:
            return sp.simplify(a.symmetric_difference(b)) == sp.EmptySet
        except Exception:
            return a == b

    if isinstance(a, sp.Basic) and isinstance(b, sp.Basic):
        # Try .equals first
        try:
            r = a.equals(b)
            if r is True:
                return True
        except Exception:
            pass

        # Try simplify difference
        try:
            diff = sp.simplify(a - b)
            if diff == 0:
                return True
        except Exception:
            pass

        # Try simplify ratio (for multiplicative)
        try:
            ratio = sp.simplify(a / b)
            if ratio == 1:
                return True
        except Exception:
            pass

        # Fallback: numeric evaluation at random points for expressions with symbols
        try:
            syms = list((a.free_symbols | b.free_symbols))
            if syms:
                import random
                for _ in range(5):
                    subs = {s: random.randint(1, 5) for s in syms}
                    av = a.subs(subs)
                    bv = b.subs(subs)
                    if sp.simplify(av - bv) != 0:
                        return False
                return True
        except Exception:
            pass

    return False


def math_equal(pred: str, gt: str) -> bool:
    """
    Math equivalence check aligned with the style used in math-evaluation-harness.
    """
    if pred is None or gt is None:
        return False

    pred_n = normalize_latex(pred)
    gt_n = normalize_latex(gt)

    if pred_n == gt_n:
        return True

    # If both look like simple numbers (possibly with sign), compare numerically
    num_pat = r"^[+-]?\d+(\.\d+)?$"
    if re.match(num_pat, pred_n) and re.match(num_pat, gt_n):
        try:
            return float(pred_n) == float(gt_n)
        except Exception:
            pass

    # latex2sympy route
    if latex2sympy_fn is None:
        # Fallback to normalized string match only
        return pred_n == gt_n

    try:
        a = latex2sympy_fn(pred_n)
        b = latex2sympy_fn(gt_n)
        # Some implementations may return a list
        if isinstance(a, list) and len(a) == 1:
            a = a[0]
        if isinstance(b, list) and len(b) == 1:
            b = b[0]
        return _sympy_equals(a, b)
    except Exception:
        # Fallback: try sympy parsing after crude conversions
        try:
            def crude(s: str) -> str:
                s = s.replace("\\pi", "pi")
                s = s.replace("^", "**")
                s = re.sub(r"\\frac\s*{([^}]+)}{([^}]+)}", r"(\1)/(\2)", s)
                return s
            a = sp.sympify(crude(pred_n))
            b = sp.sympify(crude(gt_n))
            return _sympy_equals(a, b)
        except Exception:
            return False


def build_prompt(problem: str, tokenizer, prompt_type: str) -> str:
    if prompt_type not in {"qwen25-math-cot", "qwen25-math-tir"}:
        raise ValueError(f"Unsupported prompt_type: {prompt_type}")

    system = QWEN25_MATH_COT_SYSTEM if prompt_type == "qwen25-math-cot" else QWEN25_MATH_TIR_SYSTEM

    if getattr(tokenizer, "chat_template", None) and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": problem},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Fallback for base models without chat template
    return f"{system}\n\n{problem}\n\n"


def generate_one(model, tokenizer, prompt: str, gen_cfg: GenConfig) -> str:
    device = _first_param_device(model)
    inputs = tokenizer([prompt], return_tensors="pt").to(device)

    gen_kwargs = dict(
        max_new_tokens=gen_cfg.max_new_tokens,
        do_sample=gen_cfg.do_sample,
        temperature=gen_cfg.temperature,
        top_p=gen_cfg.top_p,
    )
    if gen_cfg.top_k and gen_cfg.top_k > 0:
        gen_kwargs["top_k"] = gen_cfg.top_k

    # Ensure pad token id
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id

    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)

    gen_ids = out[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def dump_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--lora_dir", type=str, default="")
    parser.add_argument("--output_json", type=str, required=True)

    # Optional knobs (defaults follow the official greedy evaluation setting)
    parser.add_argument("--data_path", type=str, default=os.path.join(os.path.dirname(__file__), "data", "math_500", "test.jsonl"))
    parser.add_argument("--prompt_type", type=str, default="qwen25-math-cot", choices=["qwen25-math-cot", "qwen25-math-tir"])
    parser.add_argument("--max_new_tokens", type=int, default=3072)

    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model_id, args.lora_dir)

    gen_cfg = GenConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
    )

    data = load_jsonl(args.data_path)

    results = []
    correct = 0

    t0 = time.time()
    t0 = time.time()
    from tqdm import tqdm
    pbar = tqdm(data, desc="Evaluating", unit="img")
    for i, ex in enumerate(pbar):
        ex_id = ex.get("id", ex.get("ID", ex.get("idx", str(i))))
        problem = ex.get("problem", ex.get("Problem", ex.get("question", ex.get("prompt"))))
        gt = ex.get("answer", ex.get("Answer", ex.get("target")))
        if problem is None:
            raise RuntimeError(f"Missing problem field in example {i}: keys={list(ex.keys())}")

        prompt = build_prompt(problem, tokenizer, args.prompt_type)
        resp = generate_one(model, tokenizer, prompt, gen_cfg)

        pred_ans_raw = extract_final_answer(resp)

        is_correct = math_equal(pred_ans_raw, gt) if pred_ans_raw is not None else False
        if is_correct:
            correct += 1

        results.append({
            "id": ex_id,
            "problem": problem,
            "gold_answer": gt,
            "response": resp,
            "pred_answer": pred_ans_raw,
            "correct": is_correct,
        })

        
        # update progress bar
        pbar.set_postfix({"acc": f"{correct/(i+1):.4f}"})

    total = len(data)
    acc = correct / total if total else 0.0

    out = {
        "benchmark": "MATH-500",
        "metric": "pass@1",
        "pass@1": acc,
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "model_id": args.model_id,
        "lora_dir": args.lora_dir,
        "prompt_type": args.prompt_type,
        "gen_config": asdict(gen_cfg),
        "time_sec": time.time() - t0,
        "results": results,
    }
    dump_json(args.output_json, out)
    print(f"Done. pass@1={acc:.4f} ({correct}/{total}) -> {args.output_json}")


if __name__ == "__main__":
    main()
