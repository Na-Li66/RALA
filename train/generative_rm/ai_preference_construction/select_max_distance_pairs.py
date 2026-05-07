#!/usr/bin/env python3
"""Select candidate pairs with maximum SFT-policy margin distance."""

from __future__ import annotations

import argparse
import json
import os
from itertools import combinations
from typing import Any, Dict, List, Tuple


DEFAULT_INPUT_PATH = os.environ.get("CANDIDATE_INPUT_PATH", "")
DEFAULT_OUTPUT_PATH = os.environ.get("MAX_MARGIN_PAIR_OUTPUT", "")


def require_arg(value: str, cli_name: str, env_name: str) -> None:
    if str(value or "").strip():
        return
    raise SystemExit(
        f"Missing --{cli_name}. Set it explicitly or via ${env_name}."
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Select max-margin response pairs for generative RM data construction.")
    p.add_argument("--input_path", type=str, default=DEFAULT_INPUT_PATH)
    p.add_argument("--output_path", type=str, default=DEFAULT_OUTPUT_PATH)
    p.add_argument("--score_field", type=str, default="sft_logprob_scores")
    return p


def pick_max_margin_pair(responses: List[str], scores: List[float]) -> Tuple[str, str, float]:
    if len(responses) < 2:
        raise ValueError("Need at least two candidate responses.")
    if len(scores) != len(responses):
        raise ValueError(f"scores length {len(scores)} does not match responses length {len(responses)}")

    best_key = (-1.0, -1, -1)
    best_pair = (0, 1)
    for i, j in combinations(range(len(responses)), 2):
        margin = abs(float(scores[i]) - float(scores[j]))
        key = (margin, -i, -j)
        if key > best_key:
            best_key = key
            best_pair = (i, j)

    i, j = best_pair
    return responses[i], responses[j], float(best_key[0])


def process_record(obj: Dict[str, Any], score_field: str) -> Dict[str, Any]:
    responses = obj.get("responses", [])
    scores = obj.get(score_field, None)
    if not isinstance(responses, list):
        raise ValueError("record field 'responses' must be a list")
    if not isinstance(scores, list):
        raise ValueError(f"record field '{score_field}' must be a list")

    a, b, margin = pick_max_margin_pair([str(x) for x in responses], [float(x) for x in scores])
    out = dict(obj)
    out["responses"] = [a, b]
    out["margin_distance"] = margin
    out["margin_score_field"] = score_field
    return out


def process_file(input_path: str, output_path: str, score_field: str) -> None:
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    total = 0
    ok = 0
    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            obj = json.loads(line)
            try:
                selected = process_record(obj, score_field)
            except Exception as e:
                raise RuntimeError(f"Failed at line {total}: {e}") from e
            fout.write(json.dumps(selected, ensure_ascii=False, separators=(",", ":")) + "\n")
            ok += 1
    print(f"Done. Selected {ok}/{total} max-margin pairs -> {output_path}")


def main() -> None:
    args = build_arg_parser().parse_args()
    require_arg(args.input_path, "input_path", "CANDIDATE_INPUT_PATH")
    require_arg(args.output_path, "output_path", "MAX_MARGIN_PAIR_OUTPUT")
    process_file(args.input_path, args.output_path, args.score_field)


if __name__ == "__main__":
    main()
