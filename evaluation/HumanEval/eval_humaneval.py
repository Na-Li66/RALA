"""
HumanEval runner for RALA RLHF checkpoints.

This script takes a checkpoint directory that saves LoRA weights under:

  checkpoint_xxx/policy_adapter/policy/

Usage:
  python eval_humaneval.py --checkpoint_dir /path/to/checkpoint_100

If `adapter_config.json` does not contain a resolvable base model path, pass:
  --base_model /path/to/base_model
"""

from __future__ import annotations

import json
import os
from argparse import ArgumentParser
from typing import Optional

# Some environments have a broken tensorflow install (e.g., missing distutils on Py3.12),
# which can break Transformers import through optional TF paths.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")


# --------------------------- Runtime environment --------------------------- #

def _set_offline_env() -> None:
    # HuggingFace offline knobs
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    # Common logging knobs
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


_set_offline_env()

def _script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _abspath(path: str) -> str:
    if not path:
        return ""
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(_script_dir(), path))


def _resolve_model_ref(model_ref: str) -> str:
    """
    Resolve a model reference which can be:
    - local absolute path
    - local relative path (resolved relative to this script dir *if it exists*)
    - HF Hub model id / other string (left as-is)
    """
    model_ref = (model_ref or "").strip()
    if not model_ref:
        return ""
    if os.path.isabs(model_ref):
        return model_ref
    candidate = os.path.normpath(os.path.join(_script_dir(), model_ref))
    if os.path.exists(candidate):
        return candidate
    return model_ref


def adapter_weight_file(adapter_dir: str) -> Optional[str]:
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        path = os.path.join(adapter_dir, name)
        if os.path.isfile(path):
            return path
    return None


def assert_policy_adapter_dir(adapter_dir: str, *, allow_root_policy_adapter: bool) -> str:
    adapter_dir = os.path.abspath(adapter_dir)
    adapter_file = adapter_weight_file(adapter_dir)
    if adapter_file is None:
        raise RuntimeError(f"LoRA adapter weights not found under {adapter_dir}")
    is_checkpoint_root_adapter = (
        os.path.basename(os.path.normpath(adapter_dir)) == "policy_adapter"
        and os.path.basename(os.path.dirname(os.path.normpath(adapter_dir))).startswith("checkpoint_")
    )
    if is_checkpoint_root_adapter and not allow_root_policy_adapter:
        raise RuntimeError(
            "Refusing to load checkpoint_xxx/policy_adapter root. "
            "The default trained policy adapter is checkpoint_xxx/policy_adapter/policy."
        )
    return adapter_file


def _set_common_env(*, offline: bool) -> None:
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def resolve_adapter_dir(checkpoint_dir: str, *, adapter_subdir: str) -> Optional[str]:
    """
    Resolve adapter path from a checkpoint dir.
    Priority:
      1) checkpoint_dir/<adapter_subdir> (default: policy_adapter/policy)
      2) checkpoint_dir (if user passed adapter dir directly)
    """
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    candidate = os.path.join(checkpoint_dir, adapter_subdir)
    if os.path.isfile(os.path.join(candidate, "adapter_config.json")):
        return candidate
    if os.path.isfile(os.path.join(checkpoint_dir, "adapter_config.json")):
        return checkpoint_dir
    return None


def is_full_model_dir(model_dir: str) -> bool:
    model_dir = os.path.abspath(model_dir)
    if not os.path.isfile(os.path.join(model_dir, "config.json")):
        return False
    return any(
        os.path.isfile(os.path.join(model_dir, fname))
        for fname in ("model.safetensors", "pytorch_model.bin", "pytorch_model.bin.index.json")
    )


def infer_base_model_from_adapter(adapter_dir: str) -> Optional[str]:
    cfg_path = os.path.join(adapter_dir, "adapter_config.json")
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return None

    base = cfg.get("base_model_name_or_path")
    if not isinstance(base, str) or not base.strip():
        return None

    base = base.strip()
    if os.path.isabs(base) and os.path.exists(base):
        return base

    if not os.path.isabs(base):
        rel_to_adapter = os.path.normpath(os.path.join(adapter_dir, base))
        if os.path.exists(rel_to_adapter):
            return rel_to_adapter

        rel_to_script = _abspath(base)
        if os.path.exists(rel_to_script):
            return rel_to_script

    # Could be a HF Hub id; return it as-is (works if online cache is available)
    return base


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="",
        help="Path to the RLHF checkpoint dir (e.g. .../checkpoint_100) or adapter dir.",
    )
    parser.add_argument(
        "--adapter_subdir",
        type=str,
        default="policy_adapter/policy",
        help="Adapter subdir name inside checkpoint_dir (default: policy_adapter/policy).",
    )
    parser.add_argument(
        "--allow_root_policy_adapter",
        action="store_true",
        default=False,
        help="Allow loading checkpoint_xxx/policy_adapter root.",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="",
        help="Base model path or HF id. If empty, will try to infer from adapter_config.json.",
    )
    parser.add_argument("--logdir", type=str, default="humaneval_log", help="Directory to save logs")
    parser.add_argument("--language", type=str, default="python", help="Language to evaluate")
    parser.add_argument("--dataroot", type=str, default="data", help="Path to data directory")
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--max_gen_len", type=int, default=512)
    parser.add_argument("--max_prompt_tokens", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_sample", type=int, default=1, help="Number of generations per prompt")
    parser.add_argument("--k_sample", type=int, default=1, help="Compute pass@k (k_sample)")
    parser.add_argument("--limit", type=int, default=None, help="Optional prompt/sample limit")
    parser.add_argument("--skip_score", action="store_true", help="Generate samples but skip functional scoring")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--use_chat_template", action="store_true", help="Use the training chat template for prompts")
    parser.add_argument(
        "--chat_system_prompt",
        type=str,
        default="You are a helpful coding assistant. Provide a concise, correct answer. Stop after the final answer.",
    )
    parser.add_argument("--use_pad_token_as_eos", action="store_true")
    parser.add_argument(
        "--extra_eos_token_ids",
        type=str,
        default="",
        help="Comma-separated token ids to treat as EOS in addition to tokenizer.eos_token_id.",
    )
    parser.add_argument("--no_unsloth", action="store_true", help="Force Transformers+PEFT loading")
    parser.add_argument("--offline", action="store_true", help="Force HF offline mode (no downloads)")
    parser.add_argument("--load_in_4bit", dest="load_in_4bit", action="store_true", help="Enable 4bit (Unsloth)")
    parser.add_argument(
        "--no_load_in_4bit",
        dest="load_in_4bit",
        action="store_false",
        help="Disable 4bit (Unsloth)",
    )
    parser.add_argument("--load_in_8bit", action="store_true", default=False, help="Enable 8bit (Unsloth)")
    parser.set_defaults(load_in_4bit=True)
    args = parser.parse_args()
    if args.load_in_8bit:
        args.load_in_4bit = False

    _set_common_env(offline=args.offline)

    # Heavy imports after argparse, so `--help` works even if ML deps are broken.
    import torch
    from accelerate import Accelerator
    from accelerate import DistributedDataParallelKwargs

    try:
        # Unsloth should be imported before model loading.
        from unsloth import FastLanguageModel  # type: ignore

        has_unsloth = True
    except Exception:
        FastLanguageModel = None  # type: ignore
        has_unsloth = False

    from transformers import AutoTokenizer, AutoModelForCausalLM

    try:
        from peft import PeftModel
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("peft is required to load RLHF adapters.") from exc

    from humaneval import HumanEval as evaltor

    kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters=True)]
    accelerator = Accelerator(mixed_precision="bf16", kwargs_handlers=kwargs_handlers)

    base_model = _resolve_model_ref(args.base_model)
    checkpoint_dir = _resolve_model_ref(args.checkpoint_dir)
    if not checkpoint_dir:
        if not base_model:
            raise RuntimeError("Both --checkpoint_dir and --base_model are empty; at least one is required.")
        # If checkpoint_dir is empty, treat base_model as the model to evaluate.
        checkpoint_dir = base_model

    logdir = _abspath(args.logdir)
    dataroot = _abspath(args.dataroot)
    os.makedirs(logdir, exist_ok=True)

    adapter_dir = None
    full_model = True
    if os.path.isdir(checkpoint_dir):
        adapter_dir = resolve_adapter_dir(checkpoint_dir, adapter_subdir=args.adapter_subdir)
        full_model = is_full_model_dir(checkpoint_dir)

    if not base_model and adapter_dir is not None:
        inferred = infer_base_model_from_adapter(adapter_dir)
        if inferred:
            base_model = _resolve_model_ref(inferred)

    adapter_file = None
    if adapter_dir is not None:
        adapter_file = assert_policy_adapter_dir(
            adapter_dir,
            allow_root_policy_adapter=bool(args.allow_root_policy_adapter),
        )

    if accelerator.is_main_process:
        print("==== HumanEval (Unsloth RLHF) ====")
        print(f"checkpoint_dir: {checkpoint_dir}")
        print(f"adapter_dir: {adapter_dir}")
        print(f"adapter_file: {adapter_file if adapter_file else '(none)'}")
        print(f"full_model_dir: {full_model}")
        print(f"base_model: {base_model if base_model else '(not set)'}")
        print(f"language: {args.language}")
        print(f"dataroot: {dataroot}")
        print(f"logdir: {logdir}")
        print(
            f"unsloth: {has_unsloth and (not args.no_unsloth)} | "
            f"load_in_4bit={args.load_in_4bit} load_in_8bit={args.load_in_8bit}"
        )
        print(
            "generation: "
            f"use_chat_template={args.use_chat_template} "
            f"max_prompt_tokens={args.max_prompt_tokens} "
            f"max_gen_len={args.max_gen_len} "
            f"use_pad_token_as_eos={args.use_pad_token_as_eos} "
            f"extra_eos_token_ids={args.extra_eos_token_ids or '(none)'}"
        )

    if os.path.isdir(checkpoint_dir) and adapter_dir is None and not full_model:
        raise RuntimeError(
            "checkpoint_dir is neither a full model dir nor a LoRA adapter dir. "
            "Expected 'config.json' (full model) or 'adapter_config.json' (adapter)."
        )

    def load_model():
        device_map = None
        if torch.cuda.is_available():
            device_map = {"": accelerator.local_process_index}

        if adapter_dir is None:
            if has_unsloth and not args.no_unsloth:
                model, _tokenizer = FastLanguageModel.from_pretrained(  # type: ignore[misc]
                    model_name=checkpoint_dir,
                    max_seq_length=args.max_seq_len,
                    dtype=None,
                    load_in_4bit=args.load_in_4bit,
                    load_in_8bit=args.load_in_8bit,
                )
                FastLanguageModel.for_inference(model)  # type: ignore[union-attr]
                model.eval()
                return model
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_dir,
                device_map=device_map,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
            model.eval()
            return model

        if has_unsloth and not args.no_unsloth:
            try:
                model, _tokenizer = FastLanguageModel.from_pretrained(  # type: ignore[misc]
                    model_name=adapter_dir,
                    max_seq_length=args.max_seq_len,
                    dtype=None,
                    load_in_4bit=args.load_in_4bit,
                    load_in_8bit=args.load_in_8bit,
                )
                FastLanguageModel.for_inference(model)  # type: ignore[union-attr]
                model.eval()
                return model
            except Exception as e:
                if accelerator.is_main_process:
                    print(f"[warn] Unsloth load(adapter) failed: {e}")

            if not base_model:
                raise RuntimeError(
                    "Unsloth adapter load failed and no --base_model was provided."
                )

            if accelerator.is_main_process:
                print("[info] Falling back to Unsloth load(base) + load_adapter(...)")
            model, _tokenizer = FastLanguageModel.from_pretrained(  # type: ignore[misc]
                model_name=base_model,
                max_seq_length=args.max_seq_len,
                dtype=None,
                load_in_4bit=args.load_in_4bit,
                load_in_8bit=args.load_in_8bit,
            )
            if not hasattr(model, "load_adapter"):
                raise RuntimeError("Loaded Unsloth model does not support load_adapter().")
            model.load_adapter(adapter_dir)
            FastLanguageModel.for_inference(model)  # type: ignore[union-attr]
            model.eval()
            return model

        if not base_model:
            raise RuntimeError(
                "Adapter checkpoint detected, but base model path is unknown. "
                "Pass --base_model or ensure adapter_config.json has base_model_name_or_path."
            )

        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        model = PeftModel.from_pretrained(model, adapter_dir)
        model.eval()
        return model

    model = load_model()

    # HumanEval class re-instantiates tokenizer from tokenizer_cfg["model_path"]
    # Prefer base_model for tokenizer if available; otherwise fall back to checkpoint dir.
    tokenizer_model_path = base_model if base_model else checkpoint_dir
    if adapter_dir is not None:
        adapter_root = os.path.dirname(adapter_dir)
        if os.path.isfile(os.path.join(adapter_root, "tokenizer_config.json")):
            tokenizer_model_path = adapter_root
    tokenizer_cfg = dict(cls=AutoTokenizer, model_path=tokenizer_model_path)
    extra_eos_token_ids = [
        int(item.strip())
        for item in args.extra_eos_token_ids.split(",")
        if item.strip()
    ]

    evaluator = evaltor(
        data_root=dataroot,
        max_seq_len=args.max_seq_len,
        tokenizer_cfg=tokenizer_cfg,
        log_dir=logdir,
        n_sample=args.n_sample,
        batch_size=args.batch_size,
        language=args.language,
        max_gen_len=args.max_gen_len,
        temperature=args.temperature,
        top_p=args.top_p,
        k_sample=args.k_sample,
        limit=args.limit,
        skip_score=args.skip_score,
        use_chat_template=args.use_chat_template,
        chat_system_prompt=args.chat_system_prompt,
        max_prompt_tokens=args.max_prompt_tokens,
        use_pad_token_as_eos=args.use_pad_token_as_eos,
        extra_eos_token_ids=extra_eos_token_ids,
    )

    evaluator.eval_model(model, accelerator)


if __name__ == "__main__":
    main()
