# -*- coding: utf-8 -*-
"""
RALA Step-3: RLHF (PPO / DPO) with RALA reward fusion.
Unsloth + QLoRA single-GPU friendly version.

Core algorithm:
  - r_d(x,y): discriminative latent reward (reward_head on frozen *ref* features)
  - r_g(x,y): generative verifier reward = p_phi("Yes" | x,y,P)
  - r_e(x,y): endogenous reward = negative mean response-token entropy under the SFT/ref policy
  - fusion: batch Z-score + inverse-variance weighting with EMA sigma

Implementation notes:
  - initialize policy/ref from the dual-head SFT LoRA adapter by default
  - switch policy/ref adapters on one shared model to reduce VRAM use
  - load reward_gen as an additional frozen adapter for scoring
"""
from __future__ import annotations

# ---------------------------
# Import Unsloth first so its patches are installed before Transformers/PEFT.
# ---------------------------
try:
    from unsloth import FastLanguageModel  # type: ignore
except Exception:
    FastLanguageModel = None  # noqa: N816

import argparse
import copy
import hashlib
import json
import logging
import math
import os
import random
import time
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel
try:
    # Transformers uses BitsAndBytesConfig for 4-bit loading.
    from transformers import BitsAndBytesConfig
except Exception:  # pragma: no cover
    BitsAndBytesConfig = None  # type: ignore

try:
    from peft import PeftModel
    from peft import prepare_model_for_kbit_training
except Exception as exc:  # pragma: no cover
    raise RuntimeError("peft is required. Please ensure your environment matches the SFT/RM training scripts.") from exc


# CLI arguments override these optional environment defaults.
DEFAULT_BASE_MODEL_ID = os.environ.get("BASE_MODEL", "")
DEFAULT_REWARD_DISC_HEAD_PATH = os.environ.get("REWARD_DISC_HEAD", "")
DEFAULT_REWARD_GEN_DIR = os.environ.get("REWARD_GEN_LORA", "")
DEFAULT_RLHF_DATA_PATH = os.environ.get("RLHF_DATA", "")
DEFAULT_OUTPUT_DIR = os.environ.get("RLHF_OUTPUT_DIR", os.environ.get("OUTPUT_DIR", ""))

# --------------------------- Training and generation defaults. --------------------------- #
DEFAULT_SEED = 3407

# Generation
DEFAULT_MAX_PROMPT_TOKENS = 768
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_MAX_SEQ_LENGTH = 1056
DEFAULT_DO_SAMPLE = True
DEFAULT_TOP_P = 0.85
DEFAULT_TEMPERATURE = 0.55

# PPO
DEFAULT_TOTAL_UPDATES = 270
# Effective rollout batch = batch_size * rollout_accum_steps (default: 8)
DEFAULT_BATCH_SIZE = 1
DEFAULT_ROLLOUT_ACCUM_STEPS = 8
DEFAULT_PPO_EPOCHS = 1
DEFAULT_MINI_BATCH_SIZE = 1
DEFAULT_PPO_GRAD_ACCUM_STEPS = 4
DEFAULT_LR = 1.5e-7
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_MAX_GRAD_NORM = 0.12

DEFAULT_GAMMA = 1.0
DEFAULT_LAMBDA_GAE = 0.95
DEFAULT_CLIP_RANGE = 0.02
DEFAULT_CLIP_RANGE_VF = 0.1
DEFAULT_VF_COEF = 0.1
DEFAULT_ENT_COEF = 0.0
DEFAULT_KL_COEF = 0.12
DEFAULT_KL_TOKEN_CLIP = 3.0
DEFAULT_LOG_RATIO_CLIP = 3.0

# reward fusion
DEFAULT_FUSION_EMA_BETA = 0.95
DEFAULT_FUSION_EPS = 1e-6
DEFAULT_REWARD_CLIP = 10.0

# reward_gen
VERIFICATION_PROMPT = (
    "Based on the question and the response provided, is the response correct and complete?\n"
    "Answer with only \"Yes\" or \"No\".\n\n"
    "Answer:"
)
DEFAULT_MAX_LEN_REWARD_MODELS = 1056

# DPO
DEFAULT_DPO_UPDATES = 2000
DEFAULT_DPO_BETA = 0.1
DEFAULT_DPO_CANDIDATES = 2  # online candidates per prompt

# Quantized LoRA loading for code RLHF.
DEFAULT_LOAD_IN_4BIT = False
DEFAULT_LOAD_IN_8BIT = True

# LoRA defaults used for explicit base/merged-model initialization.
DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.0
DEFAULT_LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# --------------------------- Runtime environment --------------------------- #

def _set_offline_env() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


_set_offline_env()
# --------------------------- utils --------------------------- #

def _script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _abspath(rel_or_abs: str) -> str:
    """Resolve a path relative to this script; preserve empty strings."""
    if not rel_or_abs or not rel_or_abs.strip():
        return ""
    if os.path.isabs(rel_or_abs):
        return rel_or_abs
    return os.path.normpath(os.path.join(_script_dir(), rel_or_abs))


def _require_arg(value: str, cli_name: str, env_name: str) -> None:
    if str(value or "").strip():
        return
    raise SystemExit(
        f"Missing --{cli_name}. Set it explicitly or via ${env_name}."
    )


def find_latest_checkpoint(output_dir: str) -> Tuple[Optional[str], int]:
    """Find the latest fully written checkpoint directory in output_dir."""
    if not os.path.exists(output_dir):
        return None, 0

    max_step = 0
    best_dir = None
    for name in os.listdir(output_dir):
        # match checkpoint_{step}
        m = re.match(r"checkpoint_(\d+)", name)
        if m:
            step = int(m.group(1))
            path = os.path.join(output_dir, name)
            if not os.path.isdir(path):
                continue
            if not os.path.exists(os.path.join(path, "checkpoint_complete.json")):
                continue
            if step > max_step:
                max_step = step
                best_dir = path
    return best_dir, max_step


def adapter_weight_file(adapter_dir: str) -> Optional[str]:
    """Return the LoRA weight file in an adapter directory, if present."""
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        path = os.path.join(adapter_dir, name)
        if os.path.exists(path):
            return path
    return None


def resolve_policy_adapter_for_resume(
    checkpoint_dir: str,
    *,
    allow_root_adapter_resume: bool = False,
    logger: Optional[logging.Logger] = None,
    prefix: str = "[resume]",
) -> Tuple[str, str]:
    """Resolve the trained policy adapter for a saved checkpoint.

    PEFT saves the adapter named "policy" under
    checkpoint_xxx/policy_adapter/policy/. The policy_adapter root can also
    contain a stale no-op/reference adapter when multiple adapters are present,
    so root resume is refused by default.
    """
    root_adapter_dir = os.path.join(checkpoint_dir, "policy_adapter")
    nested_policy_dir = os.path.join(root_adapter_dir, "policy")

    nested_file = adapter_weight_file(nested_policy_dir)
    if nested_file:
        config_file = os.path.join(nested_policy_dir, "adapter_config.json")
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"{prefix} missing adapter_config.json under {nested_policy_dir}")
        if logger is not None:
            logger.info(
                "%s Resolved trained policy adapter: dir=%s file=%s",
                prefix,
                nested_policy_dir,
                nested_file,
            )
        return nested_policy_dir, nested_file

    root_file = adapter_weight_file(root_adapter_dir)
    if root_file and not allow_root_adapter_resume:
        raise RuntimeError(
            f"{prefix} Refusing to resume from root adapter '{root_adapter_dir}'. "
            "This checkpoint does not contain policy_adapter/policy/adapter_model.*. "
            "In this project the root policy_adapter adapter may be a no-op/reference "
            "adapter; resume/eval must use checkpoint_xxx/policy_adapter/policy. "
            "Pass --allow_root_adapter_resume only when this root adapter is intended."
        )
    if root_file:
        if logger is not None:
            logger.warning(
                "%s Using root adapter because --allow_root_adapter_resume was set: file=%s",
                prefix,
                root_file,
            )
        return root_adapter_dir, root_file

    raise FileNotFoundError(
        f"{prefix} No policy adapter weights found. Expected "
        f"'{nested_policy_dir}/adapter_model.safetensors' (or .bin)."
    )


def save_policy_adapter(
    policy_model: nn.Module,
    tokenizer,
    adapter_dir: str,
    adapter_name: str,
    *,
    logger: Optional[logging.Logger] = None,
    prefix: str = "[checkpoint]",
) -> Tuple[str, str]:
    os.makedirs(adapter_dir, exist_ok=True)
    policy_model.set_adapter(adapter_name)
    policy_model.save_pretrained(
        adapter_dir,
        selected_adapters=[adapter_name],
        safe_serialization=True,
    )
    tokenizer.save_pretrained(adapter_dir)

    trained_adapter_dir = os.path.join(adapter_dir, adapter_name)
    trained_file = adapter_weight_file(trained_adapter_dir)
    if trained_file is None:
        raise RuntimeError(
            f"{prefix} PEFT did not save selected adapter '{adapter_name}' under "
            f"{trained_adapter_dir}. Root adapter weights were not written."
        )

    config_file = os.path.join(trained_adapter_dir, "adapter_config.json")
    if not os.path.exists(config_file):
        raise RuntimeError(f"{prefix} missing adapter_config.json under {trained_adapter_dir}")

    root_file = adapter_weight_file(adapter_dir)
    if root_file is not None:
        raise RuntimeError(
            f"{prefix} unexpected root adapter weights at {root_file}; "
            "expected checkpoints store the trained policy under policy_adapter/policy."
        )

    if logger is not None:
        logger.info(
            "%s Saved trained policy adapter: dir=%s file=%s",
            prefix,
            trained_adapter_dir,
            trained_file,
        )
    return trained_adapter_dir, trained_file


def write_checkpoint_manifest(
    ckpt_dir: str,
    adapter_dir: str,
    adapter_name: str,
    trained_adapter_dir: str,
    trained_adapter_file: str,
    *,
    extra: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
    prefix: str = "[checkpoint]",
) -> None:
    root_file = adapter_weight_file(adapter_dir)
    manifest = {
        "checkpoint_dir": ckpt_dir,
        "adapter_name": adapter_name,
        "policy_adapter_root_dir": adapter_dir,
        "trained_policy_adapter_dir": trained_adapter_dir,
        "trained_policy_adapter_file": trained_adapter_file,
        "root_adapter_file": root_file,
        "resume_or_eval_adapter_dir": trained_adapter_dir,
        "resume_or_eval_note": "Use checkpoint_xxx/policy_adapter/policy for the trained policy adapter.",
    }
    if extra:
        manifest.update(extra)
    manifest_path = os.path.join(ckpt_dir, "policy_adapter_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    if logger is not None:
        logger.info("%s Wrote policy adapter metadata to %s", prefix, manifest_path)


def write_checkpoint_complete(
    ckpt_dir: str,
    *,
    update_idx: int,
    trained_adapter_dir: str,
    trained_adapter_file: str,
    value_head_path: Optional[str] = None,
    fusion_state_path: Optional[str] = None,
    optimizer_path: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    prefix: str = "[checkpoint]",
) -> None:
    required_paths = {
        "trained_policy_adapter_file": trained_adapter_file,
        "trained_policy_adapter_config": os.path.join(trained_adapter_dir, "adapter_config.json"),
        "policy_adapter_manifest": os.path.join(ckpt_dir, "policy_adapter_manifest.json"),
    }
    if value_head_path:
        required_paths["value_head"] = value_head_path
    if fusion_state_path:
        required_paths["fusion_state"] = fusion_state_path
    if optimizer_path:
        required_paths["optimizer"] = optimizer_path

    missing = [label for label, path in required_paths.items() if not os.path.exists(path)]
    if missing:
        raise RuntimeError(f"{prefix} refusing to mark checkpoint complete; missing files: {missing}")

    complete = {
        "complete": True,
        "update": int(update_idx),
        "trained_policy_adapter_dir": trained_adapter_dir,
        "trained_policy_adapter_file": trained_adapter_file,
        "files": required_paths,
    }
    complete_path = os.path.join(ckpt_dir, "checkpoint_complete.json")
    with open(complete_path, "w", encoding="utf-8") as f:
        json.dump(complete, f, ensure_ascii=False, indent=2)
    if logger is not None:
        logger.info("%s Wrote checkpoint completion marker to %s", prefix, complete_path)


def require_validated_checkpoint_for_resume(
    checkpoint_dir: str,
    *,
    logger: Optional[logging.Logger] = None,
    prefix: str = "[resume]",
    seen: Optional[set] = None,
    require_value_head: bool = True,
) -> None:
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    if seen is None:
        seen = set()
    if checkpoint_dir in seen:
        raise RuntimeError(f"{prefix} checkpoint parent cycle detected at {checkpoint_dir}")
    seen.add(checkpoint_dir)

    adapter_root = os.path.join(checkpoint_dir, "policy_adapter")
    if adapter_weight_file(adapter_root) is not None:
        raise RuntimeError(
            f"{prefix} checkpoint validation refuses root adapter weights under {adapter_root}."
        )
    adapter_dir, adapter_file = resolve_policy_adapter_for_resume(
        checkpoint_dir,
        allow_root_adapter_resume=False,
        logger=logger,
        prefix=prefix,
    )
    manifest_path = os.path.join(checkpoint_dir, "policy_adapter_manifest.json")
    complete_path = os.path.join(checkpoint_dir, "checkpoint_complete.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"{prefix} missing policy_adapter_manifest.json under {checkpoint_dir}")
    if not os.path.exists(complete_path):
        raise FileNotFoundError(f"{prefix} missing checkpoint_complete.json under {checkpoint_dir}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(complete_path, "r", encoding="utf-8") as f:
        complete = json.load(f)
    if manifest.get("validated_checkpoint") is not True:
        raise RuntimeError(f"{prefix} checkpoint validation requires validated_checkpoint=true in {checkpoint_dir}")
    if manifest.get("unvalidated_parent_allowed") is not False:
        raise RuntimeError(f"{prefix} checkpoint validation requires unvalidated_parent_allowed=false in {checkpoint_dir}")
    if manifest.get("root_adapter_resume_allowed") is not False:
        raise RuntimeError(f"{prefix} checkpoint validation requires root_adapter_resume_allowed=false in {checkpoint_dir}")
    if manifest.get("root_adapter_file") not in (None, "", False):
        raise RuntimeError(f"{prefix} manifest records root adapter weights in {checkpoint_dir}")
    if complete.get("complete") is not True:
        raise RuntimeError(f"{prefix} checkpoint_complete.json does not have complete=true")
    if os.path.abspath(str(manifest.get("resume_or_eval_adapter_dir", ""))) != os.path.abspath(adapter_dir):
        raise RuntimeError(f"{prefix} manifest resume_or_eval_adapter_dir does not point to nested policy adapter")

    files = complete.get("files", {})
    if not isinstance(files, dict):
        raise RuntimeError(f"{prefix} checkpoint_complete files field is not a dict")
    expected_files = {
        "trained_policy_adapter_file": adapter_file,
        "trained_policy_adapter_config": os.path.join(adapter_dir, "adapter_config.json"),
        "policy_adapter_manifest": manifest_path,
        "fusion_state": os.path.join(checkpoint_dir, "fusion_state.json"),
        "optimizer": os.path.join(checkpoint_dir, "optimizer.pt"),
    }
    if require_value_head:
        expected_files["value_head"] = os.path.join(checkpoint_dir, "value_head.pt")
    missing_labels = [label for label in expected_files if label not in files]
    if missing_labels:
        raise RuntimeError(f"{prefix} checkpoint_complete missing required file labels: {missing_labels}")
    for label, expected_path in expected_files.items():
        actual_path = os.path.abspath(str(files[label]))
        if actual_path != os.path.abspath(expected_path):
            raise RuntimeError(
                f"{prefix} checkpoint_complete file path for {label} points to {actual_path}, "
                f"expected {os.path.abspath(expected_path)}"
            )
    for label, path in files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"{prefix} complete file missing for {label}: {path}")

    if re.match(r"checkpoint_(\d+)$", os.path.basename(checkpoint_dir)):
        expected_update = int(re.match(r"checkpoint_(\d+)$", os.path.basename(checkpoint_dir)).group(1))
        if int(complete.get("update", -1)) != expected_update:
            raise RuntimeError(
                f"{prefix} checkpoint_complete update={complete.get('update')} does not match {checkpoint_dir}"
            )

    checkpoint_root = manifest.get("checkpoint_root")
    parent_checkpoint = str(manifest.get("resume_checkpoint", "") or "")
    if checkpoint_root == "start_from_base":
        if parent_checkpoint:
            raise RuntimeError(f"{prefix} checkpoint_root=start_from_base must not also set resume_checkpoint")
        if logger is not None:
            logger.info("%s Checkpoint root reached at %s", prefix, checkpoint_dir)
        return
    if not parent_checkpoint:
        raise RuntimeError(
            f"{prefix} checkpoint validation requires either checkpoint_root=start_from_base or resume_checkpoint"
        )
    if os.path.isabs(parent_checkpoint):
        parent_dir = os.path.abspath(parent_checkpoint)
    else:
        parent_dir = os.path.abspath(os.path.join(checkpoint_dir, parent_checkpoint))
    if logger is not None:
        logger.info("%s Verifying parent checkpoint: %s", prefix, parent_dir)
    parent_adapter_dir, parent_adapter_file = resolve_policy_adapter_for_resume(
        parent_dir,
        allow_root_adapter_resume=False,
        logger=logger,
        prefix=f"{prefix}[parent-bind]",
    )
    if os.path.abspath(str(manifest.get("resume_policy_adapter_dir", ""))) != os.path.abspath(parent_adapter_dir):
        raise RuntimeError(f"{prefix} child manifest resume_policy_adapter_dir does not point to actual parent")
    if os.path.abspath(str(manifest.get("resume_policy_adapter_file", ""))) != os.path.abspath(parent_adapter_file):
        raise RuntimeError(f"{prefix} child manifest resume_policy_adapter_file does not point to actual parent")
    require_validated_checkpoint_for_resume(
        parent_dir,
        logger=logger,
        prefix=f"{prefix}[parent]",
        seen=seen,
        require_value_head=require_value_head,
    )


def adapter_load_blockers(missing: List[str], unexpected: List[str], adapter_name: str) -> Tuple[List[str], List[str]]:
    critical_slots = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B", "modules_to_save")
    critical_missing = [
        key
        for key in missing
        if any(slot in str(key) for slot in critical_slots) and f".{adapter_name}." in str(key)
    ]
    critical_unexpected = [str(key) for key in unexpected]
    return critical_missing, critical_unexpected



def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(log_path: str) -> logging.Logger:
    logger = logging.getLogger("rala_rlhf")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    fh = logging.FileHandler(log_path, encoding="utf-8", mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def _append_jsonl(path: str, records: List[Dict[str, object]]) -> None:
    if not path or not records:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _tok_len(tokenizer) -> int:
    try:
        return int(len(tokenizer))
    except Exception:
        return -1


def _tok_vocab_size(tokenizer) -> int:
    v = getattr(tokenizer, "vocab_size", None)
    if v is None:
        return _tok_len(tokenizer)
    try:
        return int(v)
    except Exception:
        return _tok_len(tokenizer)


def _tok_added_vocab_size(tokenizer) -> int:
    try:
        added = tokenizer.get_added_vocab()
        return int(len(added)) if isinstance(added, dict) else 0
    except Exception:
        try:
            added = getattr(tokenizer, "added_tokens_encoder", None)
            return int(len(added)) if isinstance(added, dict) else 0
        except Exception:
            return 0


def _model_vocab_size(model) -> int:
    try:
        v = int(getattr(getattr(model, "config", object()), "vocab_size", -1))
        if v > 0:
            return v
    except Exception:
        pass
    try:
        emb = model.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight"):
            return int(emb.weight.shape[0])
    except Exception:
        pass
    return -1


def log_tokenizer_model_diagnostics(
    logger: logging.Logger,
    *,
    tokenizer,
    model,
    prefix: str,
    sample_texts: Optional[List[str]] = None,
    max_samples: int = 2,
) -> None:
    """Log high-signal tokenizer/model alignment diagnostics.

    This is intentionally lightweight so it can be left enabled in training logs.
    """
    try:
        tok_cls = type(tokenizer).__name__
    except Exception:
        tok_cls = "<unknown>"
    try:
        tok_path = getattr(tokenizer, "name_or_path", None)
    except Exception:
        tok_path = None
    try:
        tok_fast = bool(getattr(tokenizer, "is_fast", False))
    except Exception:
        tok_fast = False
    try:
        chat_template = getattr(tokenizer, "chat_template", None)
    except Exception:
        chat_template = None
    try:
        model_cls = type(model).__name__
    except Exception:
        model_cls = "<unknown>"

    mv = _model_vocab_size(model)
    tl = _tok_len(tokenizer)
    tv = _tok_vocab_size(tokenizer)
    tav = _tok_added_vocab_size(tokenizer)

    eos = getattr(tokenizer, "eos_token_id", None)
    pad = getattr(tokenizer, "pad_token_id", None)
    bos = getattr(tokenizer, "bos_token_id", None)

    try:
        eos_tok = getattr(tokenizer, "eos_token", None)
        pad_tok = getattr(tokenizer, "pad_token", None)
        bos_tok = getattr(tokenizer, "bos_token", None)
    except Exception:
        eos_tok = pad_tok = bos_tok = None

    try:
        emb = model.get_input_embeddings()
        emb_n = int(emb.weight.shape[0]) if emb is not None and hasattr(emb, "weight") else -1
    except Exception:
        emb_n = -1

    logger.info(
        "%s tokenizer=%s (is_fast=%s name_or_path=%r) | len=%d vocab_size=%d added_vocab=%d | eos=%s pad=%s bos=%s | eos_tok=%r pad_tok=%r bos_tok=%r",
        prefix,
        tok_cls,
        str(tok_fast),
        tok_path,
        tl,
        tv,
        tav,
        str(eos),
        str(pad),
        str(bos),
        eos_tok,
        pad_tok,
        bos_tok,
    )
    if chat_template:
        # Keep it short to avoid bloating logs
        logger.info("%s chat_template(head)=%r", prefix, str(chat_template)[:200])
    logger.info("%s model=%s | model_vocab_size=%d | input_embedding_n=%d", prefix, model_cls, mv, emb_n)

    # Basic consistency warnings
    if mv > 0 and tl > 0 and tl != mv:
        logger.warning("%s tokenizer len (%d) != model vocab_size (%d). This is a strong sign of mismatch.", prefix, tl, mv)
    if mv > 0 and tv > 0 and tv != mv and tl != mv:
        logger.warning("%s tokenizer vocab_size (%d) != model vocab_size (%d).", prefix, tv, mv)
    if eos is None or (mv > 0 and int(eos) >= mv):
        logger.warning("%s eos_token_id=%s is invalid for model_vocab_size=%d; generation may never stop.", prefix, str(eos), mv)
    if pad is None or (mv > 0 and int(pad) >= mv):
        logger.warning("%s pad_token_id=%s is invalid for model_vocab_size=%d.", prefix, str(pad), mv)

    # Boundary decode sanity check
    if mv > 0 and tl > 0:
        try:
            hi = max(mv - 1, 0)
            probe_ids = [max(0, hi - 3), max(0, hi - 2), max(0, hi - 1), hi]
            dec = tokenizer.decode(probe_ids, skip_special_tokens=False)
            logger.info("%s decode(model_vocab tail ids=%s) -> %r", prefix, str(probe_ids), dec[:120])
        except Exception as e:
            logger.warning("%s decode(model_vocab tail ids) failed: %s", prefix, str(e))

    # Encode/decode sample prompts: detect OOV token ids relative to model vocab
    if not sample_texts:
        return
    if mv <= 0:
        return

    for i, text in enumerate(sample_texts[:max_samples]):
        try:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if not ids:
                logger.warning("%s sample[%d] encoded to empty ids", prefix, i)
                continue
            mx = int(max(ids))
            mn = int(min(ids))
            oov = [x for x in ids if int(x) >= mv]
            logger.info(
                "%s sample[%d] encode: n_ids=%d min=%d max=%d oov_count=%d (model_vocab=%d)",
                prefix,
                i,
                int(len(ids)),
                mn,
                mx,
                int(len(oov)),
                mv,
            )
            # Round-trip sanity (partial, to keep logs short)
            rt = tokenizer.decode(ids[:64], skip_special_tokens=False)
            logger.info("%s sample[%d] decode(first64) -> %r", prefix, i, rt[:160])
        except Exception as e:
            logger.warning("%s sample[%d] encode/decode failed: %s", prefix, i, str(e))


def log_generation_diagnostics(
    logger: logging.Logger,
    *,
    tokenizer,
    model_vocab: int,
    gen_ids: torch.Tensor,
    ctx_len: int,
    prefix: str,
    stop_token_ids: Optional[List[int]] = None,
    pad_stop_id: Optional[int] = None,
    max_samples: int = 2,
) -> None:
    """Log token-level stats for a generated batch.

    gen_ids: [B, ctx_len + new_len]
    """
    if gen_ids is None or not hasattr(gen_ids, "shape"):
        return
    try:
        B = int(gen_ids.size(0))
        new = gen_ids[:, ctx_len:]
        new_len = int(new.size(1))
    except Exception:
        return

    eos = getattr(tokenizer, "eos_token_id", None)
    pad = getattr(tokenizer, "pad_token_id", None)
    eos_ids = normalize_token_id_list(eos)
    stop_ids = normalize_token_id_list(stop_token_ids if stop_token_ids is not None else eos_ids)

    # Token ids for newline and "and" (often a degenerate attractor)
    nl_id = None
    and_id = None
    try:
        nls = tokenizer.encode("\n", add_special_tokens=False)
        if len(nls) == 1:
            nl_id = int(nls[0])
    except Exception:
        nl_id = None
    try:
        ands = tokenizer.encode("and", add_special_tokens=False)
        if len(ands) == 1:
            and_id = int(ands[0])
    except Exception:
        and_id = None

    logger.info(
        "%s gen_ids: B=%d ctx_len=%d new_len=%d | eos_id=%s pad_id=%s stop_ids=%s pad_stop_id=%s | nl_id=%s and_id=%s",
        prefix,
        B,
        int(ctx_len),
        new_len,
        str(eos),
        str(pad),
        str(stop_ids),
        str(pad_stop_id),
        str(nl_id),
        str(and_id),
    )

    for i in range(min(B, max_samples)):
        ids = new[i].detach().cpu().tolist()
        stop_flags = response_stop_flags(
            new[i],
            eos_token_ids=eos_ids,
            stop_token_ids=stop_ids,
            pad_stop_id=pad_stop_id,
        )
        uniq = len(set(int(x) for x in ids))
        oov = sum(1 for x in ids if model_vocab > 0 and int(x) >= model_vocab)
        nl_cnt = sum(1 for x in ids if nl_id is not None and int(x) == int(nl_id))
        and_cnt = sum(1 for x in ids if and_id is not None and int(x) == int(and_id))
        head_ids = ids[:24]
        tail_ids = ids[-24:] if len(ids) >= 24 else ids

        try:
            dec_head = tokenizer.decode(head_ids, skip_special_tokens=False)
        except Exception:
            dec_head = "<decode_failed>"

        logger.info(
            "%s sample[%d] new_tokens: uniq=%d oov=%d hit_eos=%s hit_pad_stop=%s hit_any_stop=%s stop_token_id=%s nl_cnt=%d and_cnt=%d | head_ids=%s | head_dec=%r",
            prefix,
            i,
            uniq,
            int(oov),
            str(bool(stop_flags["hit_eos"])),
            str(bool(stop_flags["hit_pad_stop"])),
            str(bool(stop_flags["hit_any_stop"])),
            str(stop_flags["stop_token_id"]),
            int(nl_cnt),
            int(and_cnt),
            str(head_ids),
            dec_head[:160],
        )
        logger.info("%s sample[%d] tail_ids=%s", prefix, i, str(tail_ids))


def format_prompts_for_generation(
    tokenizer,
    prompts: List[str],
    *,
    use_chat_template: bool,
    chat_system_prompt: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> List[str]:
    if not use_chat_template:
        return prompts
    if not hasattr(tokenizer, "apply_chat_template"):
        if logger:
            logger.warning("Tokenizer has no apply_chat_template; falling back to raw prompts.")
        return prompts
    chat_template = getattr(tokenizer, "chat_template", None)
    if not chat_template:
        if logger:
            logger.warning("Tokenizer has no chat_template; falling back to raw prompts.")
        return prompts

    sys_msg = str(chat_system_prompt) if chat_system_prompt else ""
    out: List[str] = []
    for p in prompts:
        messages = []
        if sys_msg:
            messages.append({"role": "system", "content": sys_msg})
        messages.append({"role": "user", "content": p})
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except TypeError:
            # older signature
            text = tokenizer.apply_chat_template(messages, tokenize=False)
        out.append(text)
    return out


PROMPT_GENERATION_MARKERS = (
    "### Response",
    "Response:",
    "Assistant:",
    "assistant",
    "<|assistant",
)


def prompt_tail_has_generation_marker(text: str) -> bool:
    return any(marker in str(text) for marker in PROMPT_GENERATION_MARKERS)


def prompt_tail_diag_record(
    *,
    tokenizer,
    update: int,
    rollout: int,
    sample_idx: int,
    global_prompt_idx: int,
    prompt_ids: torch.Tensor,
    prompt_token_len_raw: int,
    prompt_token_len_used: int,
    max_prompt_tokens: int,
    use_chat_template: bool,
) -> Dict[str, object]:
    ids = [int(x) for x in prompt_ids.detach().cpu().reshape(-1).tolist()]
    tail_ids = ids[-96:] if len(ids) > 96 else ids
    tail_dec = tokenizer.decode(tail_ids, skip_special_tokens=False)
    marker_present = prompt_tail_has_generation_marker(tail_dec)
    marker_expected = bool(use_chat_template)
    return {
        "type": "prompt_tail",
        "update": int(update),
        "rollout": int(rollout),
        "sample_idx": int(sample_idx),
        "global_prompt_idx": int(global_prompt_idx),
        "prompt_token_len_raw": int(prompt_token_len_raw),
        "prompt_token_len_used": int(prompt_token_len_used),
        "prompt_truncated": bool(int(prompt_token_len_raw) > int(max_prompt_tokens)),
        "padding_side": str(getattr(tokenizer, "padding_side", "")),
        "truncation_side": str(getattr(tokenizer, "truncation_side", "")),
        "tail_has_generation_marker": bool(marker_present),
        "tail_marker_expected": bool(marker_expected),
        "tail_marker_missing": bool(marker_expected and not marker_present),
        "tail_decoded": str(tail_dec)[-240:],
    }


def ensure_pad_token(tokenizer) -> None:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def _pick_prompt_field(obj: Dict[str, object], *, prompt_style: str, prompt_field: Optional[str]) -> Optional[str]:
    if prompt_field:
        val = obj.get(prompt_field)
        return val if isinstance(val, str) else None
    if prompt_style == "completion":
        for k in ("prefix", "original_prompt", "prompt"):
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                return v
        return None
    # instruction/raw: default to prompt
    v = obj.get("prompt")
    return v if isinstance(v, str) else None


def load_prompts(path: str, *, prompt_style: str = "instruction", prompt_field: Optional[str] = None) -> List[str]:
    """Load prompts from JSON or JSONL with optional field selection."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"RLHF dataset not found: {path}")

    prompts: List[str] = []
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    p = _pick_prompt_field(obj, prompt_style=prompt_style, prompt_field=prompt_field)
                    if isinstance(p, str):
                        prompts.append(p)
    else:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            for ex in obj:
                if isinstance(ex, dict):
                    p = _pick_prompt_field(ex, prompt_style=prompt_style, prompt_field=prompt_field)
                    if isinstance(p, str):
                        prompts.append(p)
        elif isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
            for ex in obj["data"]:
                if isinstance(ex, dict):
                    p = _pick_prompt_field(ex, prompt_style=prompt_style, prompt_field=prompt_field)
                    if isinstance(p, str):
                        prompts.append(p)

    if not prompts:
        raise ValueError("No prompts loaded from dataset (prompt field missing?).")
    return prompts


def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mask = mask.to(dtype=x.dtype)
    return (x * mask).sum() / (mask.sum() + eps)


def masked_std(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mask_f = mask.to(dtype=x.dtype)
    mean = (x * mask_f).sum() / (mask_f.sum() + eps)
    var = ((x - mean) ** 2 * mask_f).sum() / (mask_f.sum() + eps)
    return torch.sqrt(var + eps)


def pad_and_cat_2d(tensors: List[torch.Tensor], pad_value, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    if not tensors:
        raise ValueError("pad_and_cat_2d() got empty tensors")
    max_len = max(int(t.size(1)) for t in tensors)
    outs = []
    for t in tensors:
        if dtype is not None and t.dtype != dtype:
            t = t.to(dtype)
        if int(t.size(1)) < max_len:
            pad = torch.full((t.size(0), max_len - int(t.size(1))), pad_value, dtype=t.dtype, device=t.device)
            t = torch.cat([t, pad], dim=1)
        outs.append(t)
    return torch.cat(outs, dim=0)


def pad_and_cat_1d(tensors: List[torch.Tensor], pad_value, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    rows = []
    for t in tensors:
        if t.dim() != 1:
            t = t.reshape(-1)
        rows.append(t.unsqueeze(0))
    return pad_and_cat_2d(rows, pad_value, dtype=dtype)


def short_hash(text: str) -> str:
    return hashlib.sha1(str(text).encode("utf-8", errors="ignore")).hexdigest()[:12]


def _float_or_none(value) -> Optional[float]:
    try:
        x = float(value)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return x


def tensor_stats(values: torch.Tensor) -> Dict[str, Optional[float]]:
    if values is None:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "p01": None,
            "p05": None,
            "p95": None,
            "p99": None,
        }
    v = values.detach().float().reshape(-1).cpu()
    v = v[torch.isfinite(v)]
    if int(v.numel()) == 0:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "p01": None,
            "p05": None,
            "p95": None,
            "p99": None,
        }
    if int(v.numel()) == 1:
        one = _float_or_none(v[0].item())
        return {
            "mean": one,
            "std": 0.0,
            "min": one,
            "max": one,
            "p01": one,
            "p05": one,
            "p95": one,
            "p99": one,
        }
    qs = torch.quantile(v, torch.tensor([0.01, 0.05, 0.95, 0.99], dtype=v.dtype))
    return {
        "mean": _float_or_none(v.mean().item()),
        "std": _float_or_none(v.std(unbiased=False).item()),
        "min": _float_or_none(v.min().item()),
        "max": _float_or_none(v.max().item()),
        "p01": _float_or_none(qs[0].item()),
        "p05": _float_or_none(qs[1].item()),
        "p95": _float_or_none(qs[2].item()),
        "p99": _float_or_none(qs[3].item()),
    }


def masked_values(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return x.detach().float().cpu()[mask.detach().bool().cpu()]


def max_ngram_repeat(token_ids: List[int], n: int) -> int:
    if n <= 0 or len(token_ids) < n:
        return 0
    counts: Dict[Tuple[int, ...], int] = {}
    best = 0
    for i in range(0, len(token_ids) - n + 1):
        key = tuple(int(x) for x in token_ids[i : i + n])
        c = counts.get(key, 0) + 1
        counts[key] = c
        if c > best:
            best = c
    return best


def unique_token_ratio(token_ids: List[int]) -> Optional[float]:
    if not token_ids:
        return None
    return float(len(set(int(x) for x in token_ids))) / float(len(token_ids))


def prompt_response_overlap_ratio(prompt: str, response: str) -> Optional[float]:
    resp_words = re.findall(r"\w+", str(response).lower())
    if not resp_words:
        return None
    prompt_words = set(re.findall(r"\w+", str(prompt).lower()))
    if not prompt_words:
        return 0.0
    overlap = sum(1 for w in resp_words if w in prompt_words)
    return float(overlap) / float(len(resp_words))


def parse_int_list(raw_list: Optional[List[str]]) -> List[int]:
    if raw_list is None or len(raw_list) == 0:
        return []
    if len(raw_list) == 1:
        s = str(raw_list[0]).strip()
        if s.startswith("["):
            try:
                xs = json.loads(s)
                if isinstance(xs, list):
                    raw_list = [str(x) for x in xs]
            except Exception:
                pass
        elif "," in s:
            raw_list = [x.strip() for x in s.split(",") if x.strip()]
    out: List[int] = []
    for x in raw_list:
        sx = str(x).strip()
        if not sx:
            continue
        out.append(int(sx))
    return out


def normalize_token_id_list(raw_ids) -> List[int]:
    if raw_ids is None:
        return []
    if isinstance(raw_ids, (list, tuple, set)):
        xs = list(raw_ids)
    else:
        xs = [raw_ids]
    out: List[int] = []
    for x in xs:
        try:
            tid = int(x)
        except Exception:
            continue
        if tid < 0 or tid in out:
            continue
        out.append(tid)
    return out


def build_stop_token_config(tokenizer, args) -> Dict[str, object]:
    eos_token_ids = normalize_token_id_list(getattr(tokenizer, "eos_token_id", None))
    stop_token_ids = list(eos_token_ids)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    pad_stop_id = None
    if bool(getattr(args, "use_pad_token_as_eos", False)) and pad_token_id is not None:
        pad_stop_id = int(pad_token_id)
        if pad_stop_id not in stop_token_ids:
            stop_token_ids.append(pad_stop_id)
    for tid in normalize_token_id_list(getattr(args, "extra_eos_token_ids", [])):
        if tid not in stop_token_ids:
            stop_token_ids.append(tid)
    return {
        "eos_token_ids": eos_token_ids,
        "stop_token_ids": stop_token_ids,
        "pad_stop_id": pad_stop_id,
    }


def generation_eos_token_arg(stop_token_ids: List[int]):
    if not stop_token_ids:
        return None
    if len(stop_token_ids) == 1:
        return int(stop_token_ids[0])
    return [int(x) for x in stop_token_ids]


def truncate_response_ids(
    response_ids: torch.Tensor,
    *,
    pad_token_id: Optional[int],
    stop_token_ids: List[int],
) -> torch.Tensor:
    ids = response_ids.detach().cpu().reshape(-1)
    stop_set = set(int(x) for x in stop_token_ids)
    if stop_set:
        for pos, token_id in enumerate(ids.tolist()):
            if int(token_id) in stop_set:
                return ids[: pos + 1]
    if pad_token_id is not None:
        while ids.numel() > 0 and int(ids[-1].item()) == int(pad_token_id):
            ids = ids[:-1]
    return ids


def response_stop_flags(
    response_ids: torch.Tensor,
    *,
    eos_token_ids: List[int],
    stop_token_ids: List[int],
    pad_stop_id: Optional[int],
) -> Dict[str, object]:
    ids = [int(x) for x in response_ids.detach().cpu().reshape(-1).tolist()]
    stop_set = set(int(x) for x in stop_token_ids)
    eos_set = set(int(x) for x in eos_token_ids)
    first_stop_id = None
    for token_id in ids:
        if token_id in stop_set:
            first_stop_id = int(token_id)
            break
    hit_any_stop = first_stop_id is not None
    hit_eos = bool(first_stop_id in eos_set) if hit_any_stop else False
    hit_pad_stop = bool(pad_stop_id is not None and first_stop_id == int(pad_stop_id))
    return {
        "hit_eos": hit_eos,
        "hit_pad_stop": hit_pad_stop,
        "hit_any_stop": hit_any_stop,
        "stop_token_id": first_stop_id,
    }


def response_diag_record(
    *,
    tokenizer,
    eos_token_ids: List[int],
    stop_token_ids: List[int],
    pad_stop_id: Optional[int],
    update: int,
    rollout: int,
    sample_idx: int,
    global_prompt_idx: int,
    prompt: str,
    response: str,
    prompt_token_len_raw: int,
    prompt_token_len_used: int,
    response_ids: torch.Tensor,
    max_prompt_tokens: int,
    max_new_tokens: int,
) -> Dict[str, object]:
    ids = [int(x) for x in response_ids.detach().cpu().reshape(-1).tolist()]
    stop_flags = response_stop_flags(
        response_ids,
        eos_token_ids=eos_token_ids,
        stop_token_ids=stop_token_ids,
        pad_stop_id=pad_stop_id,
    )
    return {
        "type": "rollout_sample",
        "update": int(update),
        "rollout": int(rollout),
        "sample_idx": int(sample_idx),
        "global_prompt_idx": int(global_prompt_idx),
        "prompt_hash": short_hash(prompt),
        "prompt_token_len_raw": int(prompt_token_len_raw),
        "prompt_token_len_used": int(prompt_token_len_used),
        "prompt_truncated": bool(int(prompt_token_len_raw) > int(max_prompt_tokens)),
        "response_token_len": int(len(ids)),
        "hit_eos": bool(stop_flags["hit_eos"]),
        "hit_pad_stop": bool(stop_flags["hit_pad_stop"]),
        "hit_any_stop": bool(stop_flags["hit_any_stop"]),
        "stop_token_id": stop_flags["stop_token_id"],
        "truncated_by_max_new_tokens": bool((not bool(stop_flags["hit_any_stop"])) and int(len(ids)) >= int(max_new_tokens)),
        "char_len": int(len(str(response))),
        "line_count": int(str(response).count("\n") + 1) if response else 0,
        "unique_token_ratio": unique_token_ratio(ids),
        "repeat_2gram_max": int(max_ngram_repeat(ids, 2)),
        "repeat_3gram_max": int(max_ngram_repeat(ids, 3)),
        "repeat_4gram_max": int(max_ngram_repeat(ids, 4)),
        "prompt_response_overlap_ratio": prompt_response_overlap_ratio(prompt, response),
        "starts_with_prompt_continuation": bool(str(response).strip() and str(prompt).rstrip().endswith(str(response).strip()[:24])),
        "code_fence_balance": int(str(response).count("```")),
        "code_fence_closed": bool(str(response).count("```") % 2 == 0),
    }


def logprob_delta_diag(
    *,
    update: int,
    old_single: torch.Tensor,
    old_padded: torch.Tensor,
    mask: torch.Tensor,
) -> Dict[str, object]:
    m = mask.detach().bool().cpu()
    delta = (old_single.detach().float().cpu() - old_padded.detach().float().cpu()).abs()
    vals = delta[m]
    return {
        "type": "logprob_accounting",
        "update": int(update),
        "abs_delta": tensor_stats(vals),
        "n_abs_delta_gt_0_1": int((vals > 0.1).sum().item()) if vals.numel() else 0,
        "n_abs_delta_gt_1": int((vals > 1.0).sum().item()) if vals.numel() else 0,
        "n_abs_delta_gt_5": int((vals > 5.0).sum().item()) if vals.numel() else 0,
    }


def post_update_diag_record(
    *,
    update: int,
    post_logp: torch.Tensor,
    old_logp: torch.Tensor,
    post_ref_logp: torch.Tensor,
    mask: torch.Tensor,
    clip_range: float,
) -> Dict[str, object]:
    m = mask.detach().bool().cpu()
    log_ratio = (post_logp.detach().float().cpu() - old_logp.detach().float().cpu())
    valid_lr = log_ratio[m]
    ratio = torch.exp(valid_lr.clamp(min=-20.0, max=20.0)) if valid_lr.numel() else valid_lr
    post_kl_to_ref = (post_logp.detach().float().cpu() - post_ref_logp.detach().float().cpu())
    return {
        "type": "post_update_policy",
        "update": int(update),
        "post_logratio_old_new": tensor_stats(valid_lr),
        "post_ratio_old_new": tensor_stats(ratio),
        "post_approx_kl_old_new": _float_or_none((0.5 * valid_lr.pow(2)).mean().item()) if valid_lr.numel() else None,
        "post_clipfrac_old_new_clip_range": _float_or_none(((ratio - 1.0).abs() > float(clip_range)).float().mean().item()) if ratio.numel() else None,
        "post_clipfrac_old_new_0_02": _float_or_none(((ratio - 1.0).abs() > 0.02).float().mean().item()) if ratio.numel() else None,
        "post_clipfrac_old_new_0_03": _float_or_none(((ratio - 1.0).abs() > 0.03).float().mean().item()) if ratio.numel() else None,
        "post_clipfrac_old_new_0_2": _float_or_none(((ratio - 1.0).abs() > 0.2).float().mean().item()) if ratio.numel() else None,
        "post_kl_to_ref_seq": tensor_stats(((post_kl_to_ref * m.float()).sum(dim=1) / m.float().sum(dim=1).clamp(min=1.0))),
    }


def ratio_outlier_records(
    *,
    tokenizer,
    update: int,
    old_logp: torch.Tensor,
    new_logp: torch.Tensor,
    ref_logp: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    mask: torch.Tensor,
    response_token_ids: torch.Tensor,
    prompt_records: List[Dict[str, object]],
    threshold: float,
    topk: int,
    low_threshold: Optional[float] = None,
) -> List[Dict[str, object]]:
    m = mask.detach().bool().cpu()
    if not m.any():
        return []
    lr = (new_logp.detach().float().cpu() - old_logp.detach().float().cpu())
    ratio = torch.exp(lr.clamp(min=-20.0, max=20.0))
    score = ratio.masked_fill(~m, -1.0)
    flat = score.reshape(-1)
    k = min(int(topk), int(m.sum().item()))
    if k <= 0:
        return []
    vals, inds = torch.topk(flat, k=k)
    records: List[Dict[str, object]] = []
    width = int(score.size(1))
    advantages_cpu = advantages.detach().float().cpu()
    returns_cpu = returns.detach().float().cpu()
    mask_cpu = mask.detach().cpu()

    def build_record(sample_i: int, token_pos: int, *, ratio_value: float, direction: str, cutoff: float) -> Dict[str, object]:
        token_id = None
        token_text = ""
        try:
            token_id = int(response_token_ids[sample_i, token_pos].item())
            token_text = tokenizer.decode([token_id], skip_special_tokens=False)
        except Exception:
            token_id = None
            token_text = ""
        base = prompt_records[sample_i] if sample_i < len(prompt_records) else {}
        return {
            "type": "ratio_outlier",
            "direction": direction,
            "update": int(update),
            "sample_i": sample_i,
            "rollout": int(base.get("rollout", -1)),
            "sample_idx": int(base.get("sample_idx", -1)),
            "global_prompt_idx": int(base.get("global_prompt_idx", -1)),
            "prompt_hash": str(base.get("prompt_hash", "")),
            "response_len": int(base.get("response_token_len", -1)),
            "token_pos": token_pos,
            "token_id": token_id,
            "token_text": token_text,
            "ratio_cutoff": _float_or_none(cutoff),
            "old_logp": _float_or_none(old_logp[sample_i, token_pos].item()),
            "new_logp": _float_or_none(new_logp[sample_i, token_pos].item()),
            "ref_logp": _float_or_none(ref_logp[sample_i, token_pos].item()),
            "log_ratio_raw": _float_or_none(lr[sample_i, token_pos].item()),
            "ratio_raw": _float_or_none(ratio_value),
            "advantage": _float_or_none(advantages_cpu[sample_i, token_pos].item()),
            "return": _float_or_none(returns_cpu[sample_i, token_pos].item()),
            "action_mask": int(mask_cpu[sample_i, token_pos].item()),
        }

    for val, flat_idx in zip(vals.tolist(), inds.tolist()):
        if float(val) < float(threshold):
            continue
        sample_i = int(flat_idx // width)
        token_pos = int(flat_idx % width)
        records.append(
            build_record(
                sample_i,
                token_pos,
                ratio_value=float(ratio[sample_i, token_pos].item()),
                direction="high",
                cutoff=float(threshold),
            )
        )

    if low_threshold is not None and float(low_threshold) > 0.0:
        low_cutoff = float(low_threshold)
        if low_cutoff > 1.0:
            low_cutoff = 1.0 / low_cutoff
        low_score = ratio.masked_fill(~m, float("inf"))
        low_flat = low_score.reshape(-1)
        low_vals, low_inds = torch.topk(low_flat, k=k, largest=False)
        for val, flat_idx in zip(low_vals.tolist(), low_inds.tolist()):
            if float(val) > float(low_cutoff):
                continue
            sample_i = int(flat_idx // width)
            token_pos = int(flat_idx % width)
            records.append(
                build_record(
                    sample_i,
                    token_pos,
                    ratio_value=float(ratio[sample_i, token_pos].item()),
                    direction="low",
                    cutoff=float(low_cutoff),
                )
            )
    return records


def parse_reward_list(raw_list: Optional[List[str]]) -> List[str]:
    """Parse reward stream selections.

    Supported forms:
      - --reward disc gen endo
      - --reward '["disc","endo"]'
      - --reward disc,endo
    """
    if raw_list is None or len(raw_list) == 0:
        return ["disc", "gen", "endo"]
    if len(raw_list) == 1:
        s = raw_list[0].strip()
        if s.startswith("["):
            try:
                xs = json.loads(s)
                if isinstance(xs, list):
                    raw_list = [str(x) for x in xs]
            except Exception:
                pass
        elif "," in s:
            raw_list = [x.strip() for x in s.split(",") if x.strip()]
    out = []
    for x in raw_list:
        x = str(x).strip().lower()
        if not x:
            continue
        if x not in ("disc", "gen", "endo"):
            raise ValueError(f"--reward contains invalid value: {x} (allowed: disc/gen/endo)")
        out.append(x)
    if not out:
        out = ["disc", "gen", "endo"]
    # Deduplicate while preserving order.
    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


def choose_compute_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def unsloth_for_inference(model) -> None:
    if FastLanguageModel is None:
        return
    try:
        FastLanguageModel.for_inference(model)
    except Exception:
        return


def unsloth_for_training(model, *, use_gradient_checkpointing: bool = True) -> None:
    if FastLanguageModel is None:
        return
    try:
        FastLanguageModel.for_training(model, use_gradient_checkpointing=use_gradient_checkpointing)
    except TypeError:
        try:
            FastLanguageModel.for_training(model)
        except Exception:
            return
    except Exception:
        return


def load_base_model_qlora(
    *,
    model_name_or_path: str,
    max_seq_length: int,
    load_in_4bit: bool,
    load_in_8bit: bool,
    logger: logging.Logger,
):
    """Load the base model through Unsloth."""
    if FastLanguageModel is None:
        raise RuntimeError("Unsloth is not installed but is required for this script.")

    logger.info(
        "[load] Using Unsloth FastLanguageModel.from_pretrained(load_in_4bit=%s, load_in_8bit=%s)",
        str(load_in_4bit),
        str(load_in_8bit),
    )
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name_or_path,
        max_seq_length=int(max_seq_length),
        dtype=None,  # Let Unsloth choose fp16 or bf16.
        load_in_4bit=bool(load_in_4bit),
        load_in_8bit=bool(load_in_8bit),
    )
    return model, tokenizer


# --------------------------- reward_disc head --------------------------- #

def _select_last_token(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    lengths = attention_mask.long().sum(dim=1) - 1
    lengths = torch.clamp(lengths, min=0)
    batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
    return hidden_states[batch_idx, lengths]


def resolve_reward_disc_head_path(path: str) -> str:
    if os.path.isdir(path):
        return os.path.join(path, "reward_head.pt")
    return path


def resolve_reward_disc_level(path: str, requested: str, logger: Optional[logging.Logger] = None) -> str:
    requested = str(requested or "auto").strip().lower()
    if requested in {"embedding", "embedding-level"}:
        requested = "embedding_level"
    if requested in {"token", "token-level"}:
        requested = "token_level"
    if requested in {"embedding_level", "token_level"}:
        return requested
    if requested != "auto":
        raise ValueError(
            f"Invalid reward_disc_level={requested!r}. Expected auto, embedding_level, or token_level."
        )

    head_path = resolve_reward_disc_head_path(path)
    meta_path = os.path.join(os.path.dirname(head_path), "training_meta.json")
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        level = str(meta.get("reward_level", "")).strip().lower()
        if level in {"embedding_level", "token_level"}:
            if logger is not None:
                logger.info("[policy] resolved reward_disc_level=%s from %s", level, meta_path)
            return level
        if logger is not None:
            logger.warning("[policy] training_meta.json has invalid reward_level=%r; using embedding_level", level)
    except FileNotFoundError:
        if logger is not None:
            logger.info("[policy] no reward_disc training_meta.json at %s; using embedding_level", meta_path)
    except Exception as e:
        if logger is not None:
            logger.warning("[policy] failed to read reward_disc training_meta.json at %s: %s; using embedding_level", meta_path, e)
    return "embedding_level"


class RewardHead(nn.Module):
    """Linear reward head for embedding-level or token-level discriminative rewards."""
    def __init__(self, hidden_size: int, *, reward_level: str = "embedding_level") -> None:
        super().__init__()
        if reward_level not in {"embedding_level", "token_level"}:
            raise ValueError(
                f"Invalid reward_level={reward_level!r}. Expected 'embedding_level' or 'token_level'."
            )
        self.reward_level = reward_level
        self.proj = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if attention_mask.device != hidden_states.device:
            attention_mask = attention_mask.to(hidden_states.device)
        if self.reward_level == "embedding_level":
            pooled = _select_last_token(hidden_states, attention_mask)
            if pooled.dtype != self.proj.weight.dtype:
                pooled = pooled.to(self.proj.weight.dtype)
            return self.proj(pooled).squeeze(-1)

        if hidden_states.dtype != self.proj.weight.dtype:
            hidden_states = hidden_states.to(self.proj.weight.dtype)
        token_scores = self.proj(hidden_states).squeeze(-1)
        mask = attention_mask.to(dtype=token_scores.dtype)
        return (token_scores * mask).sum(dim=1)


# --------------------------- reward_gen scorer --------------------------- #

def _get_yes_no_token_ids(tokenizer) -> Tuple[int, int]:
    yes_candidates = ["Yes", "yes", " Yes", " yes"]
    no_candidates = ["No", "no", " No", " no"]

    yes_id = None
    for candidate in yes_candidates:
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(ids) == 1:
            yes_id = ids[0]
            break
    if yes_id is None:
        yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]

    no_id = None
    for candidate in no_candidates:
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(ids) == 1:
            no_id = ids[0]
            break
    if no_id is None:
        no_id = tokenizer.encode("No", add_special_tokens=False)[0]

    return int(yes_id), int(no_id)


def _encode_verification_format(
    tokenizer,
    prompt: str,
    response: str,
    verification_prompt: str,
    max_length: int,
) -> Tuple[List[int], List[int]]:
    """Encode prompt head, response tail, and verification prompt."""
    bos: List[int] = [int(tokenizer.bos_token_id)] if tokenizer.bos_token_id is not None else []

    q_prefix_ids = tokenizer.encode("Question: ", add_special_tokens=False)
    r_prefix_ids = tokenizer.encode("\n\nResponse: ", add_special_tokens=False)
    v_ids = tokenizer.encode("\n\n" + verification_prompt, add_special_tokens=False)

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    response_ids = tokenizer.encode(response, add_special_tokens=False)

    base_len = len(bos) + len(q_prefix_ids) + len(r_prefix_ids) + len(v_ids)

    if max_length is None or max_length <= 0:
        input_ids = bos + q_prefix_ids + prompt_ids + r_prefix_ids + response_ids + v_ids
        return input_ids, [1] * len(input_ids)

    if base_len >= max_length:
        keep = bos + q_prefix_ids + r_prefix_ids + v_ids
        keep = keep[-max_length:]
        return keep, [1] * len(keep)

    budget = max_length - base_len

    MIN_PROMPT_TOKENS = 64
    MIN_RESPONSE_TOKENS = 256

    desired_resp = min(len(response_ids), min(MIN_RESPONSE_TOKENS, budget))
    prompt_budget = min(len(prompt_ids), max(0, budget - desired_resp))
    prompt_budget = min(prompt_budget, len(prompt_ids))

    if len(prompt_ids) > 0 and budget > 0:
        min_prompt = min(MIN_PROMPT_TOKENS, budget)
        if prompt_budget < min_prompt:
            steal = min(min_prompt - prompt_budget, max(0, desired_resp - 1))
            desired_resp -= steal
            prompt_budget = min(len(prompt_ids), budget - desired_resp)

    resp_budget = max(0, budget - prompt_budget)
    resp_budget = min(resp_budget, len(response_ids))

    prompt_keep = prompt_ids[:prompt_budget]
    resp_keep = response_ids[-resp_budget:] if resp_budget > 0 else []

    input_ids = bos + q_prefix_ids + prompt_keep + r_prefix_ids + resp_keep + v_ids
    if len(input_ids) > max_length:
        input_ids = input_ids[-max_length:]
    attention_mask = [1] * len(input_ids)
    return input_ids, attention_mask


def _pad_2d(seqs: List[List[int]], pad_value: int) -> torch.Tensor:
    max_len = max(len(s) for s in seqs) if seqs else 0
    out = torch.full((len(seqs), max_len), pad_value, dtype=torch.long)
    for i, s in enumerate(seqs):
        if not s:
            continue
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out


def _last_token_logits(logits: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    attn = attention_mask.long()
    idx = attn.sum(dim=1) - 1
    idx = idx.clamp(min=0, max=logits.size(1) - 1)
    batch = torch.arange(logits.size(0), device=logits.device)
    return logits[batch, idx, :]


def _p_yes_from_logits(
    logits: torch.Tensor,
    attention_mask: torch.Tensor,
    yes_token_id: int,
    no_token_id: int,
) -> torch.Tensor:
    last = _last_token_logits(logits, attention_mask)
    yn = torch.stack([last[:, yes_token_id], last[:, no_token_id]], dim=-1).float()  # [B,2]
    return F.softmax(yn, dim=-1)[:, 0]


def _ensure_lora_adapter_coverage(
    model: nn.Module,
    adapter_name: str,
    *,
    reference_adapter: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Ensure `adapter_name` exists on **every** LoRA layer in `model`.

    Why this is needed:
    - In practice, an adapter (e.g. reward_gen) might be trained with only a subset of target_modules.
    - When you later *combine* adapters (e.g. `set_adapter(["ref", "reward_gen"])`) and your model
      contains LoRA layers on other modules (e.g. `gate_proj` / `up_proj` / `down_proj`),
      Unsloth's fast LoRA kernels may do direct dict lookups like `lora_A[adapter]` and raise:
          KeyError: 'reward_gen'

    Fix:
    - For LoRA layers where `adapter_name` is missing, we create a zero-initialized LoRA (A,B) pair
      and set its scaling to 0.0, so it contributes nothing but prevents KeyError.

    This keeps behavior identical to "adapter has no weights on that module" while making the runtime robust.
    """
    patched = 0

    for module in model.modules():
        if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
            continue

        lora_A = getattr(module, "lora_A", None)
        lora_B = getattr(module, "lora_B", None)
        if not isinstance(lora_A, nn.ModuleDict) or not isinstance(lora_B, nn.ModuleDict):
            continue

        if adapter_name in lora_A and adapter_name in lora_B:
            # Ensure scaling key exists to avoid KeyError in some kernels
            if hasattr(module, "scaling") and isinstance(getattr(module, "scaling"), dict):
                if adapter_name not in module.scaling:
                    try:
                        any_key = next(iter(module.scaling.keys()))
                        module.scaling[adapter_name] = float(module.scaling.get(any_key, 1.0))
                    except Exception:
                        module.scaling[adapter_name] = 1.0
            continue

        # Pick a template adapter to infer shapes / dtype / device
        template = None
        if reference_adapter and reference_adapter in lora_A and reference_adapter in lora_B:
            template = reference_adapter
        elif len(lora_A) > 0 and len(lora_B) > 0:
            # Try to pick a key that exists in both
            for k in lora_A.keys():
                if k in lora_B:
                    template = k
                    break
            if template is None:
                template = next(iter(lora_A.keys()))

        if template is None or template not in lora_A or template not in lora_B:
            continue

        A_ref = lora_A[template]
        B_ref = lora_B[template]

        # Infer (in_features, r, out_features)
        try:
            in_features = int(getattr(A_ref, "in_features"))
            r = int(getattr(A_ref, "out_features"))
        except Exception:
            # weight shape: [r, in_features]
            r, in_features = [int(x) for x in A_ref.weight.shape]

        try:
            out_features = int(getattr(B_ref, "out_features"))
        except Exception:
            # weight shape: [out_features, r]
            out_features = int(B_ref.weight.shape[0])

        device = A_ref.weight.device
        dtype = A_ref.weight.dtype

        A_new = nn.Linear(in_features, r, bias=False).to(device=device, dtype=dtype)
        B_new = nn.Linear(r, out_features, bias=False).to(device=device, dtype=dtype)
        nn.init.zeros_(A_new.weight)
        nn.init.zeros_(B_new.weight)

        lora_A[adapter_name] = A_new
        lora_B[adapter_name] = B_new

        # dropout dict (if present)
        if hasattr(module, "lora_dropout") and isinstance(getattr(module, "lora_dropout"), nn.ModuleDict):
            p = 0.0
            try:
                if template in module.lora_dropout:
                    p = float(getattr(module.lora_dropout[template], "p", 0.0))
            except Exception:
                p = 0.0
            module.lora_dropout[adapter_name] = nn.Dropout(p)

        # scaling / meta dicts
        if hasattr(module, "scaling") and isinstance(getattr(module, "scaling"), dict):
            # IMPORTANT: patched adapter contributes nothing
            module.scaling[adapter_name] = 0.0
        if hasattr(module, "r") and isinstance(getattr(module, "r"), dict):
            module.r[adapter_name] = int(r)
        if hasattr(module, "lora_alpha") and isinstance(getattr(module, "lora_alpha"), dict):
            try:
                module.lora_alpha[adapter_name] = int(module.lora_alpha.get(template, r))
            except Exception:
                module.lora_alpha[adapter_name] = int(r)

        # Freeze new params
        for p_ in A_new.parameters():
            p_.requires_grad_(False)
        for p_ in B_new.parameters():
            p_.requires_grad_(False)

        patched += 1

    if logger is not None and patched > 0:
        logger.info(
            "[reward_gen] Patched %d LoRA layers to include missing adapter '%s' (zero init, scaling=0).",
            patched,
            adapter_name,
        )

class RewardGenScorer:
    """Generative verifier reward scorer.

    Mode: Shared mode (Memory efficient).
    Reuses the Policy model and loads reward_gen as an extra adapter.
    Temporarily switches adapter during scoring.
    """
    def __init__(
        self,
        *,
        base_model_id: str,
        reward_gen_dir: str,
        tokenizer,
        max_seq_length: int,
        load_in_4bit: bool,
        logger: logging.Logger,
        shared_model: Optional[PeftModel] = None,
        restore_adapter_name: Optional[str] = None,
    ) -> None:
        self.logger = logger
        self.tokenizer = tokenizer
        self.yes_id, self.no_id = _get_yes_no_token_ids(tokenizer)
        self.reward_gen_adapter = "reward_gen"
        self.restore_adapter = restore_adapter_name

        if shared_model is None:
            raise ValueError("[reward_gen] shared_model is required. Separate mode is removed.")

        # ---------------- Shared Mode ---------------- #
        self.logger.info("[reward_gen] Using SHARED model mode (saving VRAM)...")
        model = shared_model

        # Load reward_gen adapter if not present
        if not hasattr(model, "load_adapter"):
                raise RuntimeError("Shared model must support load_adapter")

        if not os.path.exists(reward_gen_dir):
            raise FileNotFoundError(f"[reward_gen] reward_gen adapter not found at: {reward_gen_dir}. "
                                    f"Please check --reward_gen_lora argument.")

        self.logger.info("[reward_gen] loading reward_gen adapter (frozen) into shared model from: %s", reward_gen_dir)
        try:
            model.load_adapter(reward_gen_dir, adapter_name=self.reward_gen_adapter, is_trainable=False)
        except TypeError:
            model.load_adapter(reward_gen_dir, adapter_name=self.reward_gen_adapter)

        # Freeze it
        for n, p in model.named_parameters():
            if self.reward_gen_adapter in n:
                p.requires_grad_(False)

        # Patch missing layers to prevent KeyError in Unsloth
        _ensure_lora_adapter_coverage(
            model,
            self.reward_gen_adapter,
            reference_adapter=self.restore_adapter, # use policy adapter as a shape/template reference
            logger=self.logger
        )

        self.model = model
        self.device = next(model.parameters()).device
        self.logger.info("[reward_gen] yes_id=%d no_id=%d device=%s", self.yes_id, self.no_id, str(self.device))

    @torch.no_grad()
    def score_batch(self, prompts: List[str], responses: List[str], *, max_length: int) -> torch.Tensor:
        assert len(prompts) == len(responses)
        pad_id = int(self.tokenizer.pad_token_id)

        ids_list: List[List[int]] = []
        attn_list: List[List[int]] = []
        for p, r in zip(prompts, responses):
            ids, attn = _encode_verification_format(
                self.tokenizer,
                p,
                r,
                VERIFICATION_PROMPT,
                max_length,
            )
            ids_list.append(ids)
            attn_list.append(attn)

        input_ids = _pad_2d(ids_list, pad_id).to(self.device)
        attention_mask = _pad_2d(attn_list, 0).to(self.device)

        # Switch adapter if shared
        prev_adapter = None
        try:
            # peft model usually has 'active_adapter' property or '_active_adapter'
            if hasattr(self.model, "active_adapter"):
                prev_adapter = self.model.active_adapter
                if isinstance(prev_adapter, (list, tuple)) and len(prev_adapter) == 1:
                    prev_adapter = prev_adapter[0]
            elif hasattr(self.model, "active_adapters"):
                prev_adapter = self.model.active_adapters
                if isinstance(prev_adapter, (list, tuple)) and len(prev_adapter) == 1:
                    prev_adapter = prev_adapter[0]
        except Exception:
            pass

        # Activate reward_gen
        self.model.set_adapter(self.reward_gen_adapter)
        self.model.eval()

        try:
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
            logits = out.logits
            p_yes = _p_yes_from_logits(logits, attention_mask, self.yes_id, self.no_id)  # [B]
            return p_yes.detach().float().cpu()
        finally:
            if prev_adapter:
                try:
                    self.model.set_adapter(prev_adapter)
                except Exception:
                    pass


# --------------------------- reward fusion --------------------------- #

class MultiRewardFusion:
    """Fuse selected reward streams with normalized inverse-variance weights."""

    def __init__(self, ema_beta: float = 0.95, eps: float = 1e-6) -> None:
        self.ema_beta = float(ema_beta)
        self.eps = float(eps)
        self.sigma: Dict[str, float] = {"disc": 1.0, "gen": 1.0, "endo": 1.0}

    def state_dict(self):
        return {"sigma": dict(self.sigma), "ema_beta": self.ema_beta, "eps": self.eps}

    def load_state_dict(self, state):
        if not isinstance(state, dict):
            return
        if "sigma" in state and isinstance(state["sigma"], dict):
            for k, v in state["sigma"].items():
                if k in self.sigma:
                    try:
                        self.sigma[k] = float(v)
                    except Exception:
                        pass

        # Load params
        if "ema_beta" in state:
            try:
                self.ema_beta = float(state["ema_beta"])
            except Exception:
                pass
        if "eps" in state:
            try:
                self.eps = float(state["eps"])
            except Exception:
                pass

    def fuse(
        self,
        rewards: Dict[str, torch.Tensor],
        active: List[str],
        logger: Optional[logging.Logger] = None,
        verbose: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """rewards: each [B] CPU tensor. active: subset of keys.
        return fused [B] and info.

        Args:
            rewards: reward streams keyed by "disc", "gen", and "endo".
            active: reward streams included in this run.
            logger: optional logger for diagnostics.
            verbose: whether to log per-stream statistics.
        """
        if not active:
            raise ValueError("active reward streams is empty")
        # ensure keys exist
        for k in active:
            if k not in rewards:
                raise KeyError(f"Missing reward stream: {k}")

        # Normalize each stream and keep diagnostics for logging.
        zscores: Dict[str, torch.Tensor] = {}
        stats: Dict[str, Dict[str, float]] = {}

        for k in active:
            x = rewards[k].float()
            mu = float(x.mean().item())
            range_val = float((x.max() - x.min()).clamp(min=self.eps).item())
            raw_std = float(x.std(unbiased=False).clamp(min=self.eps).item())

            zscores[k] = (x - mu) / range_val

            # Standard deviation after normalization.
            z_std = float(zscores[k].std(unbiased=False).clamp(min=self.eps).item())

            stats[k] = {
                "raw_values": x.tolist(),
                "mean": mu,
                "range": range_val,
                "raw_std": raw_std,
                "z_std": z_std,
            }

        # Log detailed per-stream statistics when requested.
        if verbose and logger is not None:
            for k in active:
                s = stats[k]
                raw_str = ", ".join([f"{v:.4f}" for v in s["raw_values"]])
                logger.info(
                    "[fuse] %s: raw=[%s] | mean=%.4f | range=%.4f | raw_std=%.4f | z_std=%.4f",
                    k, raw_str, s["mean"], s["range"], s["raw_std"], s["z_std"]
                )

        # update sigma on normalized streams
        for k in active:
            std_k = stats[k]["z_std"]
            self.sigma[k] = self.ema_beta * self.sigma[k] + (1.0 - self.ema_beta) * std_k

        # inverse variance weights
        inv: Dict[str, float] = {}
        for k in active:
            inv[k] = 1.0 / (self.sigma[k] * self.sigma[k] + self.eps)
        denom = sum(inv.values())
        weights: Dict[str, float] = {k: inv[k] / denom for k in active}

        fused = None
        for k in active:
            part = weights[k] * zscores[k]
            fused = part if fused is None else (fused + part)
        assert fused is not None

        info: Dict[str, float] = {}
        for k in ["disc", "gen", "endo"]:
            info[f"alpha_{k}"] = float(weights.get(k, 0.0))
            info[f"sigma_{k}"] = float(self.sigma.get(k, 1.0))
            # Keep raw statistics for downstream diagnostics.
            if k in stats:
                info[f"mean_{k}"] = stats[k]["mean"]
                info[f"range_{k}"] = stats[k]["range"]
                info[f"raw_std_{k}"] = stats[k]["raw_std"]

        if verbose and logger is not None:
            weight_str = ", ".join([f"{k}={weights.get(k, 0.0):.4f}" for k in active])
            sigma_str = ", ".join([f"{k}={self.sigma.get(k, 1.0):.4f}" for k in active])
            logger.info("[fuse] weights: %s | sigma_ema: %s", weight_str, sigma_str)

        return fused, info


# --------------------------- PPO helpers --------------------------- #

def _get_base(model):
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def forward_hidden_and_logits(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return last hidden states and logits from a decoder-only model."""
    base = _get_base(model)

    # Most decoder-only models have base.model
    if hasattr(base, "model") and isinstance(getattr(base, "model"), nn.Module):
        out = base.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden = out.last_hidden_state
    else:
        out = base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden = getattr(out, "last_hidden_state", None)
        if hidden is None:
            raise RuntimeError("Cannot get last_hidden_state from model output.")

    if hasattr(base, "lm_head") and isinstance(getattr(base, "lm_head"), nn.Module):
        # Match lm_head dtype, which can differ under quantized loading.
        lm_head = base.lm_head
        if hasattr(lm_head, "weight") and lm_head.weight.dtype != hidden.dtype:
            hidden = hidden.to(lm_head.weight.dtype)
        logits = lm_head(hidden)
    elif hasattr(base, "embed_out") and isinstance(getattr(base, "embed_out"), nn.Module):
        embed_out = base.embed_out
        if hasattr(embed_out, "weight") and embed_out.weight.dtype != hidden.dtype:
            hidden = hidden.to(embed_out.weight.dtype)
        logits = embed_out(hidden)
    else:
        out2 = base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        logits = out2.logits

    return hidden, logits


def gather_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Memory-friendly token logprob computation.

    We avoid creating a full fp32 copy of logits (which is extremely expensive for large-vocab models).
    The formula is:
        log p(y) = logits[y] - logsumexp(logits)

    Args:
        logits: [B, L, V]
        labels: [B, L] (token ids)

    Returns:
        logprobs: [B, L] float32
    """
    if labels.device != logits.device:
        labels = labels.to(logits.device)
    if labels.dtype != torch.long:
        labels = labels.long()

    # Keep logits in its original dtype (bf16/fp16) to avoid a huge fp32 clone.
    selected = torch.gather(logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    norm = torch.logsumexp(logits, dim=-1)

    # Cast the small [B,L] result to fp32 for stability in downstream PPO math.
    return (selected - norm).to(torch.float32)


def token_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Compute per-position categorical entropy from model logits."""
    probs = torch.softmax(logits, dim=-1)
    expected_logits = (probs * logits).sum(dim=-1)
    log_z = torch.logsumexp(logits, dim=-1)
    return (log_z - expected_logits).to(torch.float32)

@dataclass
class PackedBatch:
    input_ids: torch.Tensor          # [B, L]
    attention_mask: torch.Tensor     # [B, L]
    prompt_lens: torch.Tensor        # [B]
    response_lens: torch.Tensor      # [B]
    prompts: List[str]
    responses: List[str]


def build_packed_batch(
    tokenizer,
    prompts: List[str],
    response_ids_list: List[torch.Tensor],
    prompt_ids_list: List[torch.Tensor],
) -> PackedBatch:
    assert len(prompts) == len(response_ids_list) == len(prompt_ids_list)

    pad_id = int(tokenizer.pad_token_id)

    input_ids_list: List[torch.Tensor] = []
    attention_list: List[torch.Tensor] = []
    prompt_lens: List[int] = []
    response_lens: List[int] = []
    responses_text: List[str] = []

    for p_ids, r_ids in zip(prompt_ids_list, response_ids_list):
        if r_ids.numel() == 0:
            r_ids = torch.tensor([tokenizer.eos_token_id], dtype=torch.long)
        seq = torch.cat([p_ids, r_ids], dim=0)
        attn = torch.ones_like(seq)
        input_ids_list.append(seq)
        attention_list.append(attn)
        prompt_lens.append(int(p_ids.numel()))
        response_lens.append(int(r_ids.numel()))
        responses_text.append(tokenizer.decode(r_ids.tolist(), skip_special_tokens=True))

    max_len = max(int(s.numel()) for s in input_ids_list)
    B = len(input_ids_list)
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((B, max_len), dtype=torch.long)

    for i, (seq, attn) in enumerate(zip(input_ids_list, attention_list)):
        L = int(seq.numel())
        input_ids[i, :L] = seq
        attention_mask[i, :L] = attn

    return PackedBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_lens=torch.tensor(prompt_lens, dtype=torch.long),
        response_lens=torch.tensor(response_lens, dtype=torch.long),
        prompts=prompts,
        responses=responses_text,
    )


def extract_action_tensors(
    logprobs_all: torch.Tensor,  # [B, L-1]
    values_all: torch.Tensor,    # [B, L-1]
    prompt_lens: torch.Tensor,   # [B]
    response_lens: torch.Tensor, # [B]
    *,
    pad_to: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, Lm1 = logprobs_all.shape
    T_max = int(pad_to) if pad_to is not None else int(response_lens.max().item())

    logp_out = torch.zeros((B, T_max), dtype=logprobs_all.dtype, device=logprobs_all.device)
    v_out = torch.zeros((B, T_max), dtype=values_all.dtype, device=values_all.device)
    m_out = torch.zeros((B, T_max), dtype=torch.long, device=logprobs_all.device)

    for i in range(B):
        P = int(prompt_lens[i].item())
        T = int(response_lens[i].item())
        start = max(P - 1, 0)
        end = min(start + T, Lm1)
        seg_len = max(0, end - start)
        if seg_len <= 0:
            continue
        logp_out[i, :seg_len] = logprobs_all[i, start:end]
        v_out[i, :seg_len] = values_all[i, start:end]
        m_out[i, :seg_len] = 1

    return logp_out, v_out, m_out


def compute_gae(
    rewards: torch.Tensor,   # [B,T]
    values: torch.Tensor,    # [B,T]
    mask: torch.Tensor,      # [B,T]
    gamma: float,
    lam: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, T = rewards.shape
    advantages = torch.zeros_like(rewards)
    lastgaelam = torch.zeros((B,), device=rewards.device, dtype=rewards.dtype)

    for t in reversed(range(T)):
        m_t = mask[:, t].to(dtype=rewards.dtype)
        if t < T - 1:
            next_v = values[:, t + 1] * mask[:, t + 1].to(dtype=rewards.dtype)
        else:
            next_v = torch.zeros_like(values[:, t])
        delta = (rewards[:, t] + gamma * next_v - values[:, t]) * m_t
        lastgaelam = delta + gamma * lam * lastgaelam
        lastgaelam = lastgaelam * m_t
        advantages[:, t] = lastgaelam

    returns = advantages + values
    return advantages, returns


class ValueHead(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.device != self.proj.weight.device:
            hidden_states = hidden_states.to(self.proj.weight.device)
        if hidden_states.dtype != self.proj.weight.dtype:
            hidden_states = hidden_states.to(self.proj.weight.dtype)
        out = self.proj(hidden_states).squeeze(-1)
        return out


@dataclass
class PolicyBundle:
    model: PeftModel
    policy_adapter_name: str
    ref_adapter_name: str
    value_head: Optional[ValueHead]
    reward_head: Optional[RewardHead]
    input_device: torch.device
    head_device: torch.device


def _detect_hidden_size(model) -> int:
    base = _get_base(model)
    hs = getattr(base.config, "hidden_size", None)
    if hs is None:
        hs = getattr(base.config, "n_embd", None)
    if hs is None:
        raise ValueError("Cannot read hidden_size from model config.")
    return int(hs)


def _device_of_lm_head(model) -> torch.device:
    base = _get_base(model)
    if hasattr(base, "lm_head") and isinstance(getattr(base, "lm_head"), nn.Module):
        return next(base.lm_head.parameters()).device
    return next(base.parameters()).device


def _adapter_parameters(model: nn.Module, adapter_name: str) -> List[Tuple[str, nn.Parameter]]:
    """Return PEFT LoRA parameters that belong to a specific adapter."""
    out: List[Tuple[str, nn.Parameter]] = []
    seen = set()
    adapter_slots = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B")

    for module_name, module in model.named_modules():
        for slot in adapter_slots:
            container = getattr(module, slot, None)
            if container is None or adapter_name not in container:
                continue
            adapter_module = container[adapter_name]
            for param_name, param in adapter_module.named_parameters(recurse=True):
                key = id(param)
                if key in seen:
                    continue
                seen.add(key)
                full_name = f"{module_name}.{slot}.{adapter_name}.{param_name}".strip(".")
                out.append((full_name, param))

        modules_to_save = getattr(module, "modules_to_save", None)
        if modules_to_save is not None and adapter_name in modules_to_save:
            adapter_module = modules_to_save[adapter_name]
            for param_name, param in adapter_module.named_parameters(recurse=True):
                key = id(param)
                if key in seen:
                    continue
                seen.add(key)
                full_name = f"{module_name}.modules_to_save.{adapter_name}.{param_name}".strip(".")
                out.append((full_name, param))

    if not out:
        # Fallback for PEFT versions whose named_parameters include the adapter name directly.
        for name, param in model.named_parameters():
            if adapter_name in name:
                key = id(param)
                if key not in seen:
                    seen.add(key)
                    out.append((name, param))
    return out


def copy_adapter_parameters(
    model: nn.Module,
    src_adapter: str,
    dst_adapter: str,
    logger: Optional[logging.Logger] = None,
) -> int:
    """Copy LoRA weights from one adapter to another when both adapters exist."""
    copied = 0
    adapter_slots = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B")
    with torch.no_grad():
        for module in model.modules():
            for slot in adapter_slots:
                container = getattr(module, slot, None)
                if container is None or src_adapter not in container or dst_adapter not in container:
                    continue
                src_module = container[src_adapter]
                dst_module = container[dst_adapter]
                src_params = dict(src_module.named_parameters(recurse=True))
                for name, dst_param in dst_module.named_parameters(recurse=True):
                    src_param = src_params.get(name)
                    if src_param is None or src_param.shape != dst_param.shape:
                        continue
                    dst_param.copy_(src_param)
                    copied += 1
    if logger is not None:
        logger.info("[policy] Copied adapter parameters src='%s' -> dst='%s': tensors=%d", src_adapter, dst_adapter, copied)
    if copied == 0:
        raise RuntimeError(f"Failed to copy adapter parameters from '{src_adapter}' to '{dst_adapter}'.")
    return copied


def set_only_adapter_trainable(
    model: nn.Module,
    adapter_name: str,
    fallback_named_params: Optional[List[Tuple[str, nn.Parameter]]] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[int, int]:
    """Freeze everything, then explicitly enable one adapter's parameters."""
    if hasattr(model, "set_adapter"):
        model.set_adapter(adapter_name)

    for param in model.parameters():
        param.requires_grad_(False)

    adapter_params = _adapter_parameters(model, adapter_name)
    if not adapter_params and fallback_named_params:
        adapter_params = [
            (name, param)
            for name, param in fallback_named_params
            if param is not None
        ]
    for _, param in adapter_params:
        param.requires_grad_(True)

    tensor_count = len(adapter_params)
    scalar_count = int(sum(param.numel() for _, param in adapter_params))
    if logger is not None:
        sample_names = [name for name, _ in adapter_params[:6]]
        logger.info(
            "[policy] trainable adapter '%s': tensors=%d scalars=%d sample=%s",
            adapter_name,
            tensor_count,
            scalar_count,
            sample_names,
        )
    if tensor_count == 0 or scalar_count == 0:
        raise RuntimeError(
            f"No trainable parameters found for adapter '{adapter_name}'. "
            "Refusing to start RLHF because only the value head would update."
        )
    return tensor_count, scalar_count


def adapter_parameter_abs_sums(model: nn.Module, adapter_name: str) -> Tuple[int, float, float]:
    total_abs_sum = 0.0
    lora_b_abs_sum = 0.0
    tensor_count = 0
    for name, param in _adapter_parameters(model, adapter_name):
        abs_sum = float(param.detach().float().abs().sum().cpu())
        total_abs_sum += abs_sum
        if ".lora_B." in name:
            lora_b_abs_sum += abs_sum
        tensor_count += 1
    return tensor_count, total_abs_sum, lora_b_abs_sum


def collect_trainable_named(module: nn.Module) -> List[Tuple[str, nn.Parameter]]:
    return [(name, param) for name, param in module.named_parameters() if param.requires_grad]


def grad_norm_to_float(params: List[nn.Parameter], max_norm: float) -> float:
    norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
    if isinstance(norm, torch.Tensor):
        return float(norm.detach().float().cpu().item())
    return float(norm)


def load_policy_bundle_single_gpu(
    *,
    base_model_id: str,
    sft_lora_path: str,
    reward_head_path: str,
    reward_head_level: str,
    max_seq_length: int,
    load_in_4bit: bool,
    load_in_8bit: bool,
    logger: logging.Logger,
    need_value_head: bool,
    need_reward_head: bool = True,
    debug_tokenizer: bool = False,
    # LoRA config used only when --init_from_base_model is selected.
    lora_r: int = DEFAULT_LORA_R,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    lora_dropout: float = DEFAULT_LORA_DROPOUT,
    lora_target_modules: Optional[List[str]] = None,
) -> PolicyBundle:
    """Load a policy bundle initialized from SFT LoRA by default.

    If ``sft_lora_path`` is set, the adapter is loaded as the frozen reference
    adapter and then copied into a trainable policy adapter. If it is empty, the
    caller must have explicitly requested base/merged-model initialization.
    """
    if lora_target_modules is None:
        lora_target_modules = DEFAULT_LORA_TARGET_MODULES

    logger.info("[policy] loading base model (4bit=%s, 8bit=%s) ...", str(load_in_4bit), str(load_in_8bit))
    base, tokenizer2 = load_base_model_qlora(
        model_name_or_path=base_model_id,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        logger=logger,
    )
    if debug_tokenizer:
        log_tokenizer_model_diagnostics(
            logger,
            tokenizer=tokenizer2,
            model=base,
            prefix="[policy/unsloth]",
            sample_texts=None,
        )
    base.config.use_cache = False

    # prepare k-bit training (enables grad checkpointing hooks etc.)
    try:
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    except TypeError:
        try:
            base = prepare_model_for_kbit_training(base)
        except Exception:
            pass

    ref_name = "ref"
    policy_name = "policy"
    sft_lora_path = str(sft_lora_path or "").strip()

    if sft_lora_path:
        if not os.path.exists(sft_lora_path):
            raise FileNotFoundError(f"SFT LoRA adapter not found at: {sft_lora_path}")
        logger.info("[policy] loading SFT LoRA adapter as frozen ref from: %s", sft_lora_path)
        try:
            model = PeftModel.from_pretrained(base, sft_lora_path, adapter_name=ref_name, is_trainable=False)
        except TypeError:
            model = PeftModel.from_pretrained(base, sft_lora_path, is_trainable=False)
        logger.info("[policy] initialized ref adapter from SFT LoRA")
    else:
        logger.info("[policy] initializing LoRA from base/merged model because --init_from_base_model was set")
        logger.info(
            "[policy] LoRA config: r=%d, alpha=%d, dropout=%.2f, target_modules=%s",
            lora_r,
            lora_alpha,
            lora_dropout,
            str(lora_target_modules),
        )
        try:
            model = FastLanguageModel.get_peft_model(
                base,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=3407,
            )
            logger.info("[policy] initialized LoRA via Unsloth FastLanguageModel.get_peft_model")
        except Exception as e:
            raise RuntimeError(f"[policy] Unsloth get_peft_model failed: {e}")

    # Use the current adapter as the frozen reference.
    if hasattr(model, "peft_config"):
        current_adapter = list(model.peft_config.keys())[0] if model.peft_config else "default"
    else:
        current_adapter = "default"

    # Freeze the reference adapter before adding the trainable policy adapter.
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    # Create a trainable policy adapter with the same config as ref.
    logger.info("[policy] creating policy adapter (trainable) with same config")
    if not hasattr(model, "add_adapter"):
        raise RuntimeError("peft does not support add_adapter. Please upgrade peft.")

    from peft import LoraConfig

    template_adapter_name = ref_name if hasattr(model, "peft_config") and ref_name in model.peft_config else current_adapter
    if sft_lora_path and hasattr(model, "peft_config") and template_adapter_name in model.peft_config:
        policy_lora_config = copy.deepcopy(model.peft_config[template_adapter_name])
        if hasattr(policy_lora_config, "inference_mode"):
            policy_lora_config.inference_mode = False
    else:
        policy_lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
    model.add_adapter(policy_name, policy_lora_config)

    # Rename the original adapter to the canonical ref name when possible.
    if current_adapter != ref_name:
        try:
            if hasattr(model, "rename_adapter"):
                model.rename_adapter(current_adapter, ref_name)
            else:
                ref_name = current_adapter
                logger.info("[policy] peft does not support rename_adapter, using '%s' as ref", ref_name)
        except Exception:
            ref_name = current_adapter
            logger.info("[policy] rename_adapter failed, using '%s' as ref", ref_name)

    copy_adapter_parameters(model, src_adapter=ref_name, dst_adapter=policy_name, logger=logger)
    policy_trainable_snapshot = _adapter_parameters(model, policy_name)

    # Keep all adapters frozen until Unsloth applies its training hooks.
    for p in model.parameters():
        p.requires_grad_(False)
    model.set_adapter(policy_name)
    model.train()

    # Unsloth training patch (if available)
    unsloth_for_training(model, use_gradient_checkpointing=True)
    # Unsloth/PEFT can reset flags while patching; make the policy adapter trainable last.
    set_only_adapter_trainable(
        model,
        policy_name,
        fallback_named_params=policy_trainable_snapshot,
        logger=logger,
    )
    model.set_adapter(policy_name)
    model.train()

    # devices
    try:
        input_device = model.get_input_embeddings().weight.device
    except Exception:
        input_device = next(model.parameters()).device
    head_device = _device_of_lm_head(model)

    hidden_size = _detect_hidden_size(model)
    logger.info("[policy] hidden_size=%d | input_device=%s | head_device=%s", hidden_size, str(input_device), str(head_device))
    logger.info("[policy] ref_adapter='%s' | policy_adapter='%s'", ref_name, policy_name)

    # heads
    value_head = None
    if need_value_head:
        value_head = ValueHead(hidden_size).to(device=head_device, dtype=torch.float32)

    reward_head = None
    if need_reward_head:
        reward_head = RewardHead(hidden_size, reward_level=reward_head_level).to(device=head_device, dtype=torch.float32)
        logger.info("[policy] reward_disc_level=%s", reward_head_level)

        # If a directory is provided, load reward_head.pt inside it.
        if os.path.isdir(reward_head_path):
            reward_head_path = resolve_reward_disc_head_path(reward_head_path)
            logger.info(f"[policy] --reward_disc_head is a directory, trying to load: {reward_head_path}")

        if not os.path.exists(reward_head_path):
            # STRICT CHECK: If user requested disc reward but file is missing, FAIL.
            raise FileNotFoundError(f"reward_disc head not found at: {reward_head_path}. "
                                    f"Please check --reward_disc_head argument.")

        # Load state dict
        try:
            sd = torch.load(reward_head_path, map_location="cpu", weights_only=False)
            reward_head.load_state_dict(sd)
            logger.info(f"[policy] Successfully loaded reward_disc head from: {reward_head_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load reward_disc head from {reward_head_path}: {e}")

        for p in reward_head.parameters():
            p.requires_grad_(False)
        reward_head.eval()

    return PolicyBundle(
        model=model,
        policy_adapter_name=policy_name,
        ref_adapter_name=ref_name,
        value_head=value_head,
        reward_head=reward_head,
        input_device=input_device,
        head_device=head_device,
    )


# --------------------------- DPO helpers --------------------------- #

def sum_response_logprobs(
    logp_all: torch.Tensor,         # [B, L-1]
    prompt_lens: torch.Tensor,      # [B]
    response_lens: torch.Tensor,    # [B]
) -> torch.Tensor:
    """Sum response-token log probabilities for each sample."""
    B, Lm1 = logp_all.shape
    out = torch.zeros((B,), dtype=logp_all.dtype, device=logp_all.device)
    for i in range(B):
        P = int(prompt_lens[i].item())
        T = int(response_lens[i].item())
        start = max(P - 1, 0)
        end = min(start + T, Lm1)
        if end <= start:
            continue
        out[i] = logp_all[i, start:end].sum()
    return out


def dpo_loss(
    pi_logp_chosen: torch.Tensor,
    pi_logp_rejected: torch.Tensor,
    ref_logp_chosen: torch.Tensor,
    ref_logp_rejected: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Standard DPO loss."""
    # logits = beta * ( (pi_c - pi_r) - (ref_c - ref_r) )
    logits = beta * ((pi_logp_chosen - pi_logp_rejected) - (ref_logp_chosen - ref_logp_rejected))
    return -F.logsigmoid(logits).mean()


# --------------------------- main PPO training --------------------------- #

def train_ppo(
    *,
    args,
    logger: logging.Logger,
    tokenizer,
    prompts_all: List[str],
    bundle: PolicyBundle,
    reward_gen: Optional[RewardGenScorer],
    sample_log_path: Optional[str] = None,
) -> None:
    assert bundle.value_head is not None, "PPO requires value_head"

    policy_model = bundle.model
    value_head = bundle.value_head
    reward_head = bundle.reward_head

    device = bundle.input_device
    head_device = bundle.head_device

    # AMP / scaler
    compute_dtype = choose_compute_dtype()
    use_amp = torch.cuda.is_available() and compute_dtype in (torch.float16, torch.bfloat16)
    amp_dtype = compute_dtype if use_amp else torch.float32
    use_scaler = bool(use_amp and amp_dtype == torch.float16)
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    stop_cfg = build_stop_token_config(tokenizer, args)
    stop_token_ids = [int(x) for x in stop_cfg["stop_token_ids"]]
    generation_eos = generation_eos_token_arg(stop_token_ids)

    # optimizer (policy LoRA + value head)
    policy_named_params = collect_trainable_named(policy_model)
    value_named_params = collect_trainable_named(value_head)
    if not policy_named_params:
        raise RuntimeError("[ppo] No trainable policy parameters found; refusing to train value head only.")
    trainable_params = [p for _, p in policy_named_params] + [p for _, p in value_named_params]
    policy_scalars = int(sum(p.numel() for _, p in policy_named_params))
    value_scalars = int(sum(p.numel() for _, p in value_named_params))
    logger.info(
        "[ppo] Trainable tensors: total=%d policy=%d value=%d | scalars policy=%d value=%d",
        len(trainable_params),
        len(policy_named_params),
        len(value_named_params),
        policy_scalars,
        value_scalars,
    )

    # optional: bitsandbytes 8bit optimizer
    optimizer = None
    if args.optim.lower() in ("paged_adamw_8bit", "adamw_8bit"):
        try:
            import bitsandbytes as bnb  # type: ignore
            optim_cls = getattr(bnb.optim, "PagedAdamW8bit", None) if "paged" in args.optim.lower() else getattr(bnb.optim, "AdamW8bit", None)
            if optim_cls is None:
                raise AttributeError("bitsandbytes optimizer class not found")
            optimizer = optim_cls(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
            logger.info("[ppo] Using bitsandbytes optimizer: %s", args.optim)
        except Exception as e:
            # STRICT: if user asked for 8bit optim but it failed, crash.
            raise RuntimeError(f"[ppo] Failed to initialize bitsandbytes optimizer '{args.optim}': {e}")

    if optimizer is None:
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    fusion = MultiRewardFusion(ema_beta=args.fusion_ema_beta, eps=args.fusion_eps)

    # ------------------ Auto Resume Logic ------------------ #
    start_update = 1

    latest_ckpt_dir = None
    if getattr(args, "resume_checkpoint", None):
        # User specified explicit path
        if not os.path.exists(args.resume_checkpoint):
             raise FileNotFoundError(f"Resume checkpoint not found at: {args.resume_checkpoint}")
        latest_ckpt_dir = args.resume_checkpoint
        # parse step from name
        try:
            latest_step = int(re.search(r"checkpoint_(\d+)", os.path.basename(latest_ckpt_dir)).group(1))
        except Exception:
            logger.warning(f"Could not parse step from checkpoint name {latest_ckpt_dir}, defaulting to 0 (or trust internally saved step if any)")
            latest_step = 0
    else:
        # Auto mode: resume from the latest complete checkpoint when available.
        latest_ckpt_dir, latest_step = find_latest_checkpoint(args.output_dir)

    resume_policy_adapter_meta: Dict[str, object] = {
        "checkpoint_root": "start_from_base",
        "validated_checkpoint": True,
        "unvalidated_parent_allowed": False,
        "root_adapter_resume_allowed": False,
    }
    if latest_ckpt_dir:
        logger.info(f"[ppo] Found checkpoint: {latest_ckpt_dir} (step {latest_step}). Resuming...")
        allow_unvalidated_parent = bool(getattr(args, "allow_unvalidated_parent", False))
        if not allow_unvalidated_parent:
            require_validated_checkpoint_for_resume(
                latest_ckpt_dir,
                logger=logger,
                prefix="[ppo]",
                require_value_head=True,
            )

        # 1. Load Policy Adapter Weights
        # Note: Policy adapter is already created in bundle (initialized from base). We overwrite its weights.
        from peft import set_peft_model_state_dict
        adapter_path, adapter_file = resolve_policy_adapter_for_resume(
            latest_ckpt_dir,
            allow_root_adapter_resume=bool(getattr(args, "allow_root_adapter_resume", False)),
            logger=logger,
            prefix="[ppo]",
        )
        resume_policy_adapter_meta = {
            "resume_checkpoint": latest_ckpt_dir,
            "resume_step": int(latest_step),
            "resume_policy_adapter_dir": adapter_path,
            "resume_policy_adapter_file": adapter_file,
            "parent_checkpoint_complete_file": os.path.join(latest_ckpt_dir, "checkpoint_complete.json"),
            "validated_checkpoint": not allow_unvalidated_parent,
            "unvalidated_parent_allowed": allow_unvalidated_parent,
            "root_adapter_resume_allowed": bool(getattr(args, "allow_root_adapter_resume", False)),
        }
        logger.info(f"[ppo] Loading policy adapter weights from {adapter_file}")
        if adapter_file.endswith(".safetensors"):
            from safetensors.torch import load_file
            sd = load_file(adapter_file, device="cpu")
        else:
            sd = torch.load(adapter_file, map_location="cpu")

        # set_peft_model_state_dict handles putting weights into the correct adapter module
        load_result = set_peft_model_state_dict(policy_model, sd, adapter_name=bundle.policy_adapter_name)
        missing = getattr(load_result, "missing_keys", []) or []
        unexpected = getattr(load_result, "unexpected_keys", []) or []
        tensors, total_abs_sum, lora_b_abs_sum = adapter_parameter_abs_sums(policy_model, bundle.policy_adapter_name)
        logger.info(
            "[ppo] PEFT load_result missing=%d unexpected=%d | policy_adapter tensors=%d abs_sum=%.6e lora_B_abs_sum=%.6e",
            len(missing),
            len(unexpected),
            tensors,
            total_abs_sum,
            lora_b_abs_sum,
        )
        critical_missing, critical_unexpected = adapter_load_blockers(
            [str(x) for x in missing],
            [str(x) for x in unexpected],
            bundle.policy_adapter_name,
        )
        if critical_missing or critical_unexpected:
            raise RuntimeError(
                "[ppo] adapter load mismatch: "
                f"missing={critical_missing[:20]} unexpected={critical_unexpected[:20]}"
            )
        if tensors == 0 or total_abs_sum == 0.0 or lora_b_abs_sum == 0.0:
            raise RuntimeError("[ppo] loaded policy adapter appears empty/no-op")

        # 2. Load Value Head
        vh_path = os.path.join(latest_ckpt_dir, "value_head.pt")
        if os.path.exists(vh_path):
            logger.info(f"[ppo] Loading value head from {vh_path}")
            value_head.load_state_dict(torch.load(vh_path, map_location=head_device))

        # 3. Load Fusion State
        fs_path = os.path.join(latest_ckpt_dir, "fusion_state.json")
        if os.path.exists(fs_path):
            logger.info(f"[ppo] Loading fusion state from {fs_path}")
            with open(fs_path, "r") as f:
                fusion.load_state_dict(json.load(f))

        # 4. Load Optimizer State
        opt_path = os.path.join(latest_ckpt_dir, "optimizer.pt")
        resume_optimizer = str(getattr(args, "resume_optimizer", "auto")).lower()
        should_load_optimizer = resume_optimizer == "keep" or (
            resume_optimizer == "auto" and os.path.exists(opt_path)
        )
        if should_load_optimizer and os.path.exists(opt_path):
            logger.info(f"[ppo] Loading optimizer state from {opt_path}")
            try:
                optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))
            except Exception as e:
                logger.warning(f"[ppo] Failed to load optimizer state: {e}")
        elif resume_optimizer == "keep":
            raise FileNotFoundError(f"--resume_optimizer keep but optimizer.pt not found: {opt_path}")
        else:
            logger.info("[ppo] Not loading optimizer state; using fresh optimizer.")
        if bool(getattr(args, "reset_optimizer_hparams_on_resume", True)):
            for group in optimizer.param_groups:
                group["lr"] = float(args.lr)
                group["weight_decay"] = float(args.weight_decay)
            logger.info(
                "[ppo] Optimizer hparams reset after resume: lr=%g weight_decay=%g",
                float(args.lr),
                float(args.weight_decay),
            )

        start_update = latest_step + 1
        logger.info(f"[ppo] Resumed. Starting loop from update {start_update}.")

    stop_cfg = build_stop_token_config(tokenizer, args)
    stop_token_ids = [int(x) for x in stop_cfg["stop_token_ids"]]
    eos_token_ids = [int(x) for x in stop_cfg["eos_token_ids"]]
    pad_stop_id = stop_cfg["pad_stop_id"]
    generation_eos = generation_eos_token_arg(stop_token_ids)
    logger.info(
        "[gen] eos_token_ids=%s use_pad_token_as_eos=%s pad_token_id=%s extra_eos_token_ids=%s stop_token_ids=%s",
        str(eos_token_ids),
        str(bool(getattr(args, "use_pad_token_as_eos", False))),
        str(getattr(tokenizer, "pad_token_id", None)),
        str(getattr(args, "extra_eos_token_ids", [])),
        str(stop_token_ids),
    )

    t0 = time.time()
    prompt_idx = 0
    if latest_ckpt_dir and len(prompts_all) > 0:
        prompt_idx = ((int(start_update) - 1) * int(args.batch_size) * int(args.rollout_accum_steps)) % len(prompts_all)
        logger.info("[ppo] Resume prompt_idx=%d derived from start_update=%d.", int(prompt_idx), int(start_update))
    raw_kl_seq_history: List[float] = []
    ppo_diag_path = os.path.join(args.output_dir, "ppo_diagnostics.jsonl")
    ratio_outlier_log_path = os.path.join(args.output_dir, "ratio_outliers.jsonl")
    logger.info("[diag] ppo_diagnostics=%s", ppo_diag_path)
    logger.info("[diag] ratio_outliers=%s", ratio_outlier_log_path)

    def save_ppo_checkpoint(update_idx: int, suffix: str = "") -> str:
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint_{update_idx}{suffix}")
        os.makedirs(ckpt_dir, exist_ok=True)

        adapter_dir = os.path.join(ckpt_dir, "policy_adapter")
        trained_adapter_dir, trained_adapter_file = save_policy_adapter(
            policy_model,
            tokenizer,
            adapter_dir,
            bundle.policy_adapter_name,
            logger=logger,
            prefix="[ppo]",
        )
        write_checkpoint_manifest(
            ckpt_dir,
            adapter_dir,
            bundle.policy_adapter_name,
            trained_adapter_dir,
            trained_adapter_file,
            extra={
                **resume_policy_adapter_meta,
                "policy": str(args.policy),
                "reward_set": list(args.reward),
                "reward_disc_level": str(getattr(args, "reward_disc_level", "")),
                "reward_disc_head": str(getattr(args, "reward_disc_head", "")),
                "reward_gen_lora": str(getattr(args, "reward_gen_lora", "")),
                "base_model": str(getattr(args, "base_model", "")),
                "max_new_tokens": int(getattr(args, "max_new_tokens", 0)),
                "max_seq_length": int(getattr(args, "max_seq_length", 0)),
                "max_len_reward_models": int(getattr(args, "max_len_reward_models", 0)),
                "trainer": "rala_code",
            },
            logger=logger,
            prefix="[ppo]",
        )

        value_head_path = os.path.join(ckpt_dir, "value_head.pt")
        fusion_state_path = os.path.join(ckpt_dir, "fusion_state.json")
        optimizer_path = os.path.join(ckpt_dir, "optimizer.pt")
        torch.save(value_head.state_dict(), value_head_path)
        with open(fusion_state_path, "w", encoding="utf-8") as f:
            json.dump(fusion.state_dict(), f, ensure_ascii=False, indent=2)
        torch.save(optimizer.state_dict(), optimizer_path)
        with open(os.path.join(ckpt_dir, "optimizer_meta.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "lr": float(args.lr),
                    "weight_decay": float(args.weight_decay),
                    "optim": str(args.optim),
                    "max_grad_norm": float(args.max_grad_norm),
                    "resume_optimizer": str(getattr(args, "resume_optimizer", "auto")),
                    "reset_optimizer_hparams_on_resume": bool(
                        getattr(args, "reset_optimizer_hparams_on_resume", True)
                    ),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        write_checkpoint_complete(
            ckpt_dir,
            update_idx=update_idx,
            trained_adapter_dir=trained_adapter_dir,
            trained_adapter_file=trained_adapter_file,
            value_head_path=value_head_path,
            fusion_state_path=fusion_state_path,
            optimizer_path=optimizer_path,
            logger=logger,
            prefix="[ppo]",
        )
        return ckpt_dir

    # training loop
    for update in range(start_update, args.total_updates + 1):
        do_diag = int(args.ppo_diag_every) > 0 and (int(update) % int(args.ppo_diag_every) == 0)
        update_diag_records: List[Dict[str, object]] = []
        prompt_records: List[Dict[str, object]] = []
        rollout_response_ids: List[torch.Tensor] = []

        # 1) Rollout collection with micro-batches
        micro_input_ids: List[torch.Tensor] = []
        micro_attention: List[torch.Tensor] = []
        micro_prompt_lens: List[torch.Tensor] = []
        micro_response_lens: List[torch.Tensor] = []

        micro_old_logp: List[torch.Tensor] = []
        micro_old_v: List[torch.Tensor] = []
        micro_action_mask: List[torch.Tensor] = []
        micro_logp_ref: List[torch.Tensor] = []

        # reward streams (CPU)
        micro_rewards: Dict[str, List[torch.Tensor]] = {"disc": [], "gen": [], "endo": []}

        for roll_idx in range(int(args.rollout_accum_steps)):
            # sample prompts
            batch_prompt_indices = [(prompt_idx + i) % len(prompts_all) for i in range(args.batch_size)]
            batch_prompts = [prompts_all[i] for i in batch_prompt_indices]
            prompt_idx = (prompt_idx + args.batch_size) % len(prompts_all)

            # tokenize prompts for generation (left pad)
            gen_prompts = format_prompts_for_generation(
                tokenizer,
                batch_prompts,
                use_chat_template=bool(args.use_chat_template),
                chat_system_prompt=str(args.chat_system_prompt) if str(args.chat_system_prompt) else None,
                logger=logger,
            )
            tokenizer.padding_side = "left"
            tokenizer.truncation_side = "left"
            add_special = False if bool(args.use_chat_template) else True
            prompt_token_lens_raw = [0 for _ in gen_prompts]
            if do_diag:
                try:
                    raw_tok = tokenizer(
                        gen_prompts,
                        add_special_tokens=add_special,
                        truncation=False,
                        padding=False,
                    )
                    prompt_token_lens_raw = [int(len(ids)) for ids in raw_tok["input_ids"]]
                except Exception as e:
                    logger.warning("[diag] failed to count raw prompt token lengths: %s", str(e))
            tok = tokenizer(
                gen_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_prompt_tokens,
                add_special_tokens=add_special,
            )
            input_ids_gen = tok["input_ids"].to(device)
            attn_gen = tok["attention_mask"].to(device)
            ctx_len = int(input_ids_gen.size(1))
            prompt_tail_records: List[Dict[str, object]] = []
            for i in range(input_ids_gen.size(0)):
                prompt_len_i = int(attn_gen[i].sum().item())
                prompt_ids_i = input_ids_gen[i, ctx_len - prompt_len_i : ctx_len].detach().cpu()
                prompt_len_raw_i = (
                    int(prompt_token_lens_raw[i])
                    if i < len(prompt_token_lens_raw) and int(prompt_token_lens_raw[i]) > 0
                    else prompt_len_i
                )
                tail_record = prompt_tail_diag_record(
                    tokenizer=tokenizer,
                    update=int(update),
                    rollout=int(roll_idx),
                    sample_idx=int(i),
                    global_prompt_idx=int(batch_prompt_indices[i]),
                    prompt_ids=prompt_ids_i,
                    prompt_token_len_raw=int(prompt_len_raw_i),
                    prompt_token_len_used=int(prompt_len_i),
                    max_prompt_tokens=int(args.max_prompt_tokens),
                    use_chat_template=bool(args.use_chat_template),
                )
                prompt_tail_records.append(tail_record)
                if do_diag:
                    update_diag_records.append(tail_record)
                if bool(tail_record.get("tail_marker_missing")):
                    msg = (
                        f"[prompt-tail] missing generation marker after truncation: "
                        f"update={update} rollout={roll_idx} sample={i} raw_len={prompt_len_raw_i} "
                        f"used_len={prompt_len_i} tail={tail_record.get('tail_decoded')!r}"
                    )
                    if bool(getattr(args, "strict_prompt_tail_check", False)):
                        raise RuntimeError(msg)
                    logger.warning(msg)

            # generate (policy adapter)
            policy_model.set_adapter(bundle.policy_adapter_name)
            policy_model.eval()
            unsloth_for_inference(policy_model)

            with torch.inference_mode():
                gen = policy_model.generate(
                    input_ids=input_ids_gen,
                    attention_mask=attn_gen,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    top_p=args.top_p,
                    temperature=args.temperature,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=generation_eos,
                    use_cache=not bool(getattr(args, "load_in_8bit", False)),
                )

            if getattr(args, "debug_rollouts", 0) and int(args.debug_rollouts) > 0:
                try:
                    if int(update) <= int(getattr(args, "debug_updates", 1)) and int(roll_idx) < int(args.debug_rollouts):
                        mv = _model_vocab_size(policy_model)
                        # prompt OOV check
                        if mv > 0:
                            try:
                                oov_prompt = int((input_ids_gen >= mv).sum().item())
                                mx_prompt = int(input_ids_gen.max().item())
                                logger.info(
                                    "[debug] upd=%d roll=%d prompt_ids: ctx_len=%d max_id=%d oov_count=%d (model_vocab=%d) padding_side=%s",
                                    int(update),
                                    int(roll_idx),
                                    int(ctx_len),
                                    int(mx_prompt),
                                    int(oov_prompt),
                                    int(mv),
                                    str(getattr(tokenizer, "padding_side", None)),
                                )
                            except Exception:
                                pass
                        # prompt tail decode
                        try:
                            prompt_len = int(attn_gen[0].sum().item())
                            p_ids = input_ids_gen[0, ctx_len - prompt_len : ctx_len].detach().cpu().tolist()
                            tail_ids = p_ids[-48:] if len(p_ids) > 48 else p_ids
                            tail_dec = tokenizer.decode(tail_ids, skip_special_tokens=False)
                            logger.info(
                                "[debug] upd=%d roll=%d truncation_side=%s prompt_tail_marker=%s prompt_tail_ids=%s | tail_dec=%r",
                                int(update),
                                int(roll_idx),
                                str(getattr(tokenizer, "truncation_side", None)),
                                str(prompt_tail_has_generation_marker(tail_dec)),
                                str(tail_ids),
                                tail_dec[-200:],
                            )
                        except Exception:
                            pass
                        logger.info(
                            "[debug] upd=%d roll=%d generate: do_sample=%s top_p=%.4f temp=%.4f eos_ids=%s pad_id=%s stop_ids=%s max_new=%d",
                            int(update),
                            int(roll_idx),
                            str(bool(args.do_sample)),
                            float(args.top_p),
                            float(args.temperature),
                            str(eos_token_ids),
                            str(getattr(tokenizer, "pad_token_id", None)),
                            str(stop_token_ids),
                            int(args.max_new_tokens),
                        )
                        log_generation_diagnostics(
                            logger,
                            tokenizer=tokenizer,
                            model_vocab=mv,
                            gen_ids=gen,
                            ctx_len=int(ctx_len),
                            prefix=f"[debug] upd={int(update)} roll={int(roll_idx)}",
                            stop_token_ids=stop_token_ids,
                            pad_stop_id=pad_stop_id,
                            max_samples=int(getattr(args, "debug_samples", 2)),
                        )
                except Exception as e:
                    logger.warning("[debug] generation diagnostics failed: %s", str(e))

            # back to training patch (for the later PPO backward)
            unsloth_for_training(policy_model, use_gradient_checkpointing=True)

            # rebuild prompt_ids & response_ids (remove left pads)
            response_ids_list: List[torch.Tensor] = []
            prompt_ids_list: List[torch.Tensor] = []

            for i in range(gen.size(0)):
                prompt_len = int(attn_gen[i].sum().item())
                prompt_ids = input_ids_gen[i, ctx_len - prompt_len : ctx_len].detach().cpu()
                resp_ids = truncate_response_ids(
                    gen[i, ctx_len:],
                    pad_token_id=getattr(tokenizer, "pad_token_id", None),
                    stop_token_ids=stop_token_ids,
                )
                response_ids_list.append(resp_ids)
                prompt_ids_list.append(prompt_ids)

            packed = build_packed_batch(tokenizer, batch_prompts, response_ids_list, prompt_ids_list)
            if sample_log_path:
                records: List[Dict[str, object]] = []
                for i, (p, r) in enumerate(zip(packed.prompts, packed.responses)):
                    records.append(
                        {
                            "step": int(update),
                            "rollout": int(roll_idx),
                            "sample_idx": int(i),
                            "prompt": p,
                            "response": r,
                        }
                    )
                _append_jsonl(sample_log_path, records)

            if do_diag:
                for i, (p, r, r_ids) in enumerate(zip(packed.prompts, packed.responses, response_ids_list)):
                    prompt_len_used = int(packed.prompt_lens[i].item())
                    prompt_len_raw = int(prompt_token_lens_raw[i]) if i < len(prompt_token_lens_raw) else prompt_len_used
                    diag_rec = response_diag_record(
                        tokenizer=tokenizer,
                        eos_token_ids=eos_token_ids,
                        stop_token_ids=stop_token_ids,
                        pad_stop_id=pad_stop_id,
                        update=int(update),
                        rollout=int(roll_idx),
                        sample_idx=int(i),
                        global_prompt_idx=int(batch_prompt_indices[i]),
                        prompt=p,
                        response=r,
                        prompt_token_len_raw=prompt_len_raw,
                        prompt_token_len_used=prompt_len_used,
                        response_ids=r_ids,
                        max_prompt_tokens=int(args.max_prompt_tokens),
                        max_new_tokens=int(args.max_new_tokens),
                    )
                    if i < len(prompt_tail_records):
                        diag_rec.update(
                            {
                                "prompt_padding_side": prompt_tail_records[i].get("padding_side"),
                                "prompt_truncation_side": prompt_tail_records[i].get("truncation_side"),
                                "prompt_tail_has_generation_marker": prompt_tail_records[i].get(
                                    "tail_has_generation_marker"
                                ),
                                "prompt_tail_marker_missing": prompt_tail_records[i].get("tail_marker_missing"),
                                "prompt_tail_decoded": prompt_tail_records[i].get("tail_decoded"),
                            }
                        )
                    update_diag_records.append(diag_rec)
                    prompt_records.append(diag_rec)
                    rollout_response_ids.append(r_ids.detach().cpu())

            # teacher-forcing inputs
            input_ids = packed.input_ids.to(device)
            attention_mask = packed.attention_mask.to(device)

            # old logprobs/values under current policy (no_grad)
            policy_model.set_adapter(bundle.policy_adapter_name)
            # Match the later PPO forward path. Some Unsloth 4bit kernels can
            # produce different logprobs between eval/inference and train mode.
            policy_model.train()
            value_head.train()
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    hidden_pi, logits_pi = forward_hidden_and_logits(policy_model, input_ids, attention_mask)
                    values_all = value_head(hidden_pi).float()  # [B,L]

                    logits_pi_s = logits_pi[:, :-1, :]
                    labels_s = input_ids[:, 1:]
                    logp_all_pi = gather_logprobs(logits_pi_s, labels_s)  # [B,L-1]

                    v_all_s = values_all[:, :-1]
                    logp_pi, v_pi, action_mask = extract_action_tensors(
                        logp_all_pi,
                        v_all_s,
                        packed.prompt_lens,
                        packed.response_lens,
                    )

            # ref logprobs (+ optional disc reward) under frozen ref adapter (base-like)
            policy_model.set_adapter(bundle.ref_adapter_name)
            policy_model.train()
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    hidden_ref, logits_ref = forward_hidden_and_logits(policy_model, input_ids, attention_mask)
                    logits_ref_s = logits_ref[:, :-1, :]
                    labels_s = input_ids[:, 1:]
                    logp_all_ref = gather_logprobs(logits_ref_s, labels_s)
                    logp_ref, _, _ = extract_action_tensors(
                        logp_all_ref,
                        logp_all_ref,  # dummy
                        packed.prompt_lens,
                        packed.response_lens,
                    )
                    entropy_all_ref = token_entropy_from_logits(logits_ref_s)
                    entropy_ref, _, _ = extract_action_tensors(
                        entropy_all_ref,
                        entropy_all_ref,  # dummy
                        packed.prompt_lens,
                        packed.response_lens,
                    )

                if "disc" in args.reward:
                    r_d = reward_head(hidden_ref, attention_mask).float().detach().cpu()
                    micro_rewards["disc"].append(r_d)

            # restore policy adapter
            policy_model.set_adapter(bundle.policy_adapter_name)

            # endogenous reward: penalize high SFT/ref uncertainty via negative entropy
            logp_ref_cpu = logp_ref.detach().float().cpu()
            entropy_ref_cpu = entropy_ref.detach().float().cpu()
            action_mask_cpu = action_mask.detach().cpu()

            if "endo" in args.reward:
                mask_f = action_mask_cpu.to(dtype=entropy_ref_cpu.dtype)
                r_e_cpu = -((entropy_ref_cpu * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0))
                micro_rewards["endo"].append(r_e_cpu.to(torch.float32))

            # generative reward
            if "gen" in args.reward:
                if reward_gen is None:
                    raise RuntimeError("reward_gen is required when --reward includes 'gen'")
                r_g = reward_gen.score_batch(packed.prompts, packed.responses, max_length=args.max_len_reward_models)  # cpu
                policy_model.set_adapter(bundle.policy_adapter_name)
                micro_rewards["gen"].append(r_g.to(torch.float32))

            # store sequences (CPU) for re-forward during PPO
            micro_input_ids.append(packed.input_ids)
            micro_attention.append(packed.attention_mask)
            micro_prompt_lens.append(packed.prompt_lens)
            micro_response_lens.append(packed.response_lens)

            # store action-space tensors (CPU)
            micro_old_logp.append(logp_pi.detach().float().cpu())
            micro_old_v.append(v_pi.detach().float().cpu())
            micro_action_mask.append(action_mask_cpu)
            micro_logp_ref.append(logp_ref_cpu)

            # cleanup
            del tok, input_ids_gen, attn_gen, gen, input_ids, attention_mask, hidden_pi, logits_pi, hidden_ref, logits_ref

        # 2) concat micro-batches
        pad_id = int(tokenizer.pad_token_id)
        input_ids_cpu = pad_and_cat_2d(micro_input_ids, pad_id, dtype=torch.long)
        attention_mask_cpu = pad_and_cat_2d(micro_attention, 0, dtype=torch.long)
        prompt_lens_all = torch.cat(micro_prompt_lens, dim=0)
        response_lens_all = torch.cat(micro_response_lens, dim=0)

        old_logp_cpu = pad_and_cat_2d(micro_old_logp, 0.0, dtype=torch.float32)
        old_v_cpu = pad_and_cat_2d(micro_old_v, 0.0, dtype=torch.float32)
        action_mask_cpu = pad_and_cat_2d(micro_action_mask, 0, dtype=torch.long)
        logp_ref_cpu = pad_and_cat_2d(micro_logp_ref, 0.0, dtype=torch.float32)
        old_logp_single_cpu = old_logp_cpu
        B_eff = int(input_ids_cpu.size(0))
        response_token_ids_cpu = (
            pad_and_cat_1d(rollout_response_ids, pad_id, dtype=torch.long)
            if do_diag and rollout_response_ids
            else torch.empty((0, 0), dtype=torch.long)
        )

        def recompute_action_state_on_padded_batch() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            """Recompute logprobs on the exact padded tensors used by PPO.

            Qwen/Unsloth 4bit can produce different token logprobs when the
            same sequence is forwarded alone vs inside a right-padded batch.
            PPO ratios must compare against old logprobs from the same tensor
            shape/path used during the update.
            """
            old_logps: List[torch.Tensor] = []
            old_values: List[torch.Tensor] = []
            masks: List[torch.Tensor] = []
            ref_logps: List[torch.Tensor] = []
            ref_entropies: List[torch.Tensor] = []
            chunk = max(1, int(args.mini_batch_size))
            pad_to = int(response_lens_all.max().item())

            for start in range(0, B_eff, chunk):
                end = min(start + chunk, B_eff)
                mb_input_ids = input_ids_cpu[start:end].to(device)
                mb_attn = attention_mask_cpu[start:end].to(device)
                mb_prompt_lens = prompt_lens_all[start:end]
                mb_resp_lens = response_lens_all[start:end]

                policy_model.set_adapter(bundle.policy_adapter_name)
                policy_model.train()
                value_head.train()
                with torch.no_grad():
                    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                        hidden_pi, logits_pi = forward_hidden_and_logits(policy_model, mb_input_ids, mb_attn)
                        values_all = value_head(hidden_pi).float()
                        logits_pi_s = logits_pi[:, :-1, :]
                        labels_s = mb_input_ids[:, 1:]
                        logp_all_pi = gather_logprobs(logits_pi_s, labels_s)
                        v_all_s = values_all[:, :-1]
                        logp_pi, v_pi, action_mask = extract_action_tensors(
                            logp_all_pi,
                            v_all_s,
                            mb_prompt_lens,
                            mb_resp_lens,
                            pad_to=pad_to,
                        )

                policy_model.set_adapter(bundle.ref_adapter_name)
                policy_model.train()
                with torch.no_grad():
                    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                        _, logits_ref = forward_hidden_and_logits(policy_model, mb_input_ids, mb_attn)
                        logits_ref_s = logits_ref[:, :-1, :]
                        labels_s = mb_input_ids[:, 1:]
                        logp_all_ref = gather_logprobs(logits_ref_s, labels_s)
                        logp_ref, _, _ = extract_action_tensors(
                            logp_all_ref,
                            logp_all_ref,
                            mb_prompt_lens,
                            mb_resp_lens,
                            pad_to=pad_to,
                        )
                        entropy_all_ref = token_entropy_from_logits(logits_ref_s)
                        entropy_ref, _, _ = extract_action_tensors(
                            entropy_all_ref,
                            entropy_all_ref,
                            mb_prompt_lens,
                            mb_resp_lens,
                            pad_to=pad_to,
                        )

                old_logps.append(logp_pi.detach().float().cpu())
                old_values.append(v_pi.detach().float().cpu())
                masks.append(action_mask.detach().cpu())
                ref_logps.append(logp_ref.detach().float().cpu())
                ref_entropies.append(entropy_ref.detach().float().cpu())
                del mb_input_ids, mb_attn, hidden_pi, logits_pi, logits_ref

            policy_model.set_adapter(bundle.policy_adapter_name)
            return (
                pad_and_cat_2d(old_logps, 0.0, dtype=torch.float32),
                pad_and_cat_2d(old_values, 0.0, dtype=torch.float32),
                pad_and_cat_2d(masks, 0, dtype=torch.long),
                pad_and_cat_2d(ref_logps, 0.0, dtype=torch.float32),
                pad_and_cat_2d(ref_entropies, 0.0, dtype=torch.float32),
            )

        old_logp_cpu, old_v_cpu, action_mask_cpu, logp_ref_cpu, ref_entropy_cpu = recompute_action_state_on_padded_batch()
        if do_diag:
            update_diag_records.append(
                logprob_delta_diag(
                    update=int(update),
                    old_single=old_logp_single_cpu,
                    old_padded=old_logp_cpu,
                    mask=action_mask_cpu,
                )
            )

        # 3) fuse reward (CPU)
        rewards_for_fuse: Dict[str, torch.Tensor] = {}
        for k in args.reward:
            if len(micro_rewards[k]) == 0:
                # should not happen
                rewards_for_fuse[k] = torch.zeros((B_eff,), dtype=torch.float32)
            else:
                rewards_for_fuse[k] = torch.cat(micro_rewards[k], dim=0).to(torch.float32)
        if "endo" in args.reward:
            endo_mask = action_mask_cpu.to(dtype=ref_entropy_cpu.dtype)
            rewards_for_fuse["endo"] = (
                -((ref_entropy_cpu * endo_mask).sum(dim=1) / endo_mask.sum(dim=1).clamp(min=1.0))
            ).to(torch.float32)

        fused_cpu, fusion_info = fusion.fuse(rewards_for_fuse, active=args.reward, logger=logger, verbose=True)
        enable_stop_reward_shaping = bool(getattr(args, "enable_stop_reward_shaping", False))
        fused_cpu_pre_stop = fused_cpu.clamp(min=-args.reward_clip, max=args.reward_clip)
        stop_adjust = torch.zeros_like(fused_cpu_pre_stop)
        hit_stop_count = 0
        hit_eos_count = 0
        truncated_count = 0
        short_eos_count = 0
        for i, rec in enumerate(prompt_records[: int(fused_cpu_pre_stop.numel())]):
            resp_len = int(rec.get("response_token_len", 0) or 0)
            hit_stop = bool(rec.get("hit_any_stop", False))
            hit_eos = bool(rec.get("hit_eos", False))
            truncated = bool(rec.get("truncated_by_max_new_tokens", False))
            if hit_stop:
                hit_stop_count += 1
            if hit_eos:
                hit_eos_count += 1
            if truncated:
                truncated_count += 1
                if enable_stop_reward_shaping and float(args.trunc_penalty) > 0:
                    stop_adjust[i] -= float(args.trunc_penalty)
            if enable_stop_reward_shaping and hit_stop and float(args.eos_bonus) > 0:
                if resp_len >= int(args.eos_bonus_min_tokens):
                    stop_adjust[i] += float(args.eos_bonus)
                elif float(args.short_eos_penalty) > 0:
                    short_eos_count += 1
                    stop_adjust[i] -= float(args.short_eos_penalty)
            elif hit_stop and resp_len < int(args.eos_bonus_min_tokens):
                short_eos_count += 1
        fused_cpu = (fused_cpu_pre_stop + stop_adjust).clamp(min=-args.reward_clip, max=args.reward_clip)
        denom_samples = max(1, int(len(prompt_records)))
        if do_diag:
            reward_record: Dict[str, object] = {
                "type": "reward_fusion",
                "update": int(update),
                "fused": tensor_stats(fused_cpu),
                "fused_before_stop_adjust": tensor_stats(fused_cpu_pre_stop),
                "stop_adjust": tensor_stats(stop_adjust),
                "stop_reward_shaping_enabled": enable_stop_reward_shaping,
                "reward_clip": float(args.reward_clip),
                "trunc_penalty": float(args.trunc_penalty),
                "eos_bonus": float(args.eos_bonus),
                "eos_bonus_min_tokens": int(args.eos_bonus_min_tokens),
                "short_eos_penalty": float(args.short_eos_penalty),
                "hit_any_stop_rate": float(hit_stop_count) / float(denom_samples),
                "hit_eos_rate": float(hit_eos_count) / float(denom_samples),
                "truncated_by_max_new_tokens_rate": float(truncated_count) / float(denom_samples),
                "short_eos_rate": float(short_eos_count) / float(denom_samples),
                "alpha_disc": float(fusion_info.get("alpha_disc", 0.0)),
                "alpha_gen": float(fusion_info.get("alpha_gen", 0.0)),
                "alpha_endo": float(fusion_info.get("alpha_endo", 0.0)),
                "sigma_disc": float(fusion_info.get("sigma_disc", 0.0)),
                "sigma_gen": float(fusion_info.get("sigma_gen", 0.0)),
                "sigma_endo": float(fusion_info.get("sigma_endo", 0.0)),
            }
            for k in args.reward:
                reward_record[f"{k}_raw"] = tensor_stats(rewards_for_fuse[k])
                clipped = fused_cpu.abs() >= float(args.reward_clip)
                reward_record["fused_clip_frac"] = _float_or_none(clipped.float().mean().item())
            update_diag_records.append(reward_record)
            reward_sample_records: List[Dict[str, object]] = []
            for i, rec in enumerate(prompt_records[: int(fused_cpu.numel())]):
                sample_record: Dict[str, object] = {
                    "type": "reward_sample",
                    "update": int(update),
                    "rollout": int(rec.get("rollout", -1)),
                    "sample_idx": int(rec.get("sample_idx", i)),
                    "global_prompt_idx": int(rec.get("global_prompt_idx", -1)),
                    "prompt_hash": str(rec.get("prompt_hash", "")),
                    "fused_before_stop_adjust": _float_or_none(fused_cpu_pre_stop[i].item()),
                    "stop_adjust": _float_or_none(stop_adjust[i].item()),
                    "fused": _float_or_none(fused_cpu[i].item()),
                    "hit_eos": bool(rec.get("hit_eos", False)),
                    "hit_any_stop": bool(rec.get("hit_any_stop", False)),
                    "truncated_by_max_new_tokens": bool(rec.get("truncated_by_max_new_tokens", False)),
                    "response_token_len": int(rec.get("response_token_len", 0) or 0),
                    "repeat_4gram_max": int(rec.get("repeat_4gram_max", 0) or 0),
                    "code_fence_closed": bool(rec.get("code_fence_closed", True)),
                }
                for k in args.reward:
                    vals = rewards_for_fuse.get(k)
                    if vals is not None and int(vals.numel()) > i:
                        sample_record[f"{k}_raw"] = _float_or_none(vals[i].item())
                reward_sample_records.append(sample_record)
            update_diag_records.extend(reward_sample_records)

        # 4) token-level rewards + GAE (CPU)
        mask_f = action_mask_cpu.to(dtype=torch.float32)
        resp_len_f = mask_f.sum(dim=1).clamp(min=1.0)
        kl_raw = (old_logp_cpu - logp_ref_cpu) * mask_f
        kl_for_reward = kl_raw.clamp(
            min=-float(args.kl_token_clip),
            max=float(args.kl_token_clip),
        )
        rewards_cpu = (-args.kl_coef * kl_for_reward / resp_len_f[:, None]).to(torch.float32)
        rewards_cpu = rewards_cpu * mask_f

        raw_kl_mean = float(masked_mean(kl_raw, action_mask_cpu).item())
        raw_kl_seq = kl_raw.sum(dim=1) / resp_len_f
        raw_kl_seq_mean = float(raw_kl_seq.mean().item())
        raw_kl_seq_min = float(raw_kl_seq.min().item())
        raw_kl_seq_max = float(raw_kl_seq.max().item())
        neg_kl_frac = float((((kl_raw < 0) & action_mask_cpu.bool()).float().sum() / mask_f.sum().clamp(min=1.0)).item())
        kl_reward_sum = (rewards_cpu * mask_f).sum(dim=1)
        kl_reward_sum_mean = float(kl_reward_sum.mean().item())
        raw_kl_seq_history.append(raw_kl_seq_mean)
        rolling_window = max(1, int(args.neg_kl_rolling_window))
        rolling_kl = sum(raw_kl_seq_history[-rolling_window:]) / min(len(raw_kl_seq_history), rolling_window)

        logger.info(
            "[KL] raw_mean=%.4f raw_seq_mean=%.4f raw_seq_min=%.4f raw_seq_max=%.4f neg_frac=%.3f "
            "kl_reward_sum=%.4f rolling%d=%.4f clip=%.2f",
            raw_kl_mean,
            raw_kl_seq_mean,
            raw_kl_seq_min,
            raw_kl_seq_max,
            neg_kl_frac,
            kl_reward_sum_mean,
            rolling_window,
            rolling_kl,
            float(args.kl_token_clip),
        )

        pre_update_stop_reason = ""
        if bool(args.enable_kl_watchdog) and update > int(args.kl_watchdog_warmup):
            if raw_kl_seq_mean < -float(args.neg_kl_stop):
                pre_update_stop_reason = f"raw_kl_seq_mean={raw_kl_seq_mean:.4f} < -{float(args.neg_kl_stop):.4f}"
            elif rolling_kl < -float(args.neg_kl_rolling_stop):
                pre_update_stop_reason = (
                    f"rolling{rolling_window}_raw_kl={rolling_kl:.4f} < "
                    f"-{float(args.neg_kl_rolling_stop):.4f}"
                )
        if pre_update_stop_reason:
            logger.warning("[STOP] KL watchdog triggered before PPO update %d: %s", update, pre_update_stop_reason)
            ckpt_dir = save_ppo_checkpoint(update, suffix="_pre_kl_stop")
            logger.info("[PPO] Saved pre-stop checkpoint to %s", ckpt_dir)
            break

        # add fused external reward to last token of each response
        for i in range(rewards_cpu.size(0)):
            T_i = int(response_lens_all[i].item())
            if T_i <= 0:
                continue
            if T_i - 1 < rewards_cpu.size(1):
                rewards_cpu[i, T_i - 1] += fused_cpu[i]

        advantages_cpu, returns_cpu = compute_gae(
            rewards=rewards_cpu,
            values=old_v_cpu,
            mask=action_mask_cpu,
            gamma=args.gamma,
            lam=args.lam,
        )

        # advantage normalization (over valid tokens)
        adv_mean = masked_mean(advantages_cpu, action_mask_cpu)
        adv_std = masked_std(advantages_cpu, action_mask_cpu)
        advantages_cpu = (advantages_cpu - adv_mean) / (adv_std + 1e-8)

        # move tensors to devices
        input_ids = input_ids_cpu.to(device)
        attention_mask = attention_mask_cpu.to(device)

        old_logp = old_logp_cpu.to(head_device)
        old_v = old_v_cpu.to(head_device)
        advantages = advantages_cpu.to(head_device)
        returns = returns_cpu.to(head_device)
        action_mask = action_mask_cpu.to(head_device)

        # 5) PPO update
        policy_model.set_adapter(bundle.policy_adapter_name)
        policy_model.train()
        value_head.train()

        B = int(input_ids.size(0))
        mb_size = int(args.mini_batch_size)
        idxs = list(range(B))
        if int(update) == int(start_update):
            num_minibatches = math.ceil(float(B) / float(max(1, mb_size))) * max(1, int(args.ppo_epochs))
            if int(args.ppo_epochs) == 1 and int(args.ppo_grad_accum_steps) >= int(num_minibatches):
                msg = (
                    "[ppo] Clip surrogate will see ratio~=1 for all minibatches because "
                    "ppo_epochs=1 and ppo_grad_accum_steps covers the whole rollout batch. "
                    "Trust-region behavior is currently enforced by lr/grad_clip/post-update watchdog."
                )
                if not bool(getattr(args, "allow_single_step_surrogate", False)):
                    raise RuntimeError(msg + " Pass --allow_single_step_surrogate only for diagnostic one-step updates.")
                logger.warning(msg)

        ppo_stats = {
            "pg_loss": 0.0,
            "vf_loss": 0.0,
            "kl": raw_kl_mean,
            "raw_kl_seq_mean": raw_kl_seq_mean,
            "kl_reward_sum": kl_reward_sum_mean,
            "grad_norm": 0.0,
            "approx_kl": 0.0,
            "clipfrac": 0.0,
            "ratio_min": float("inf"),
            "ratio_max": 0.0,
            "post_approx_kl": 0.0,
            "post_clipfrac": 0.0,
            "post_ratio_min": 1.0,
            "post_ratio_max": 1.0,
            "post_logratio_abs_max": 0.0,
        }
        stat_count = 0

        optimizer.zero_grad(set_to_none=True)
        accum_steps = max(1, int(args.ppo_grad_accum_steps))
        accum_count = 0

        def optimizer_step_with_clip() -> None:
            if use_scaler:
                scaler.unscale_(optimizer)
                ppo_stats["grad_norm"] = grad_norm_to_float(trainable_params, args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                ppo_stats["grad_norm"] = grad_norm_to_float(trainable_params, args.max_grad_norm)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        for epoch in range(int(args.ppo_epochs)):
            random.shuffle(idxs)
            for start in range(0, B, mb_size):
                mb_idx = idxs[start : start + mb_size]

                mb_input_ids = input_ids[mb_idx]
                mb_attn = attention_mask[mb_idx]
                mb_prompt_lens = prompt_lens_all[mb_idx]
                mb_resp_lens = response_lens_all[mb_idx]

                mb_old_logp = old_logp[mb_idx]
                mb_old_v = old_v[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_ret = returns[mb_idx]
                mb_mask = action_mask[mb_idx]

                policy_model.set_adapter(bundle.policy_adapter_name)
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    hidden_new, logits_new = forward_hidden_and_logits(policy_model, mb_input_ids, mb_attn)
                    values_new_all = value_head(hidden_new)  # [b,L] fp32

                    logits_new_s = logits_new[:, :-1, :]
                    labels_s = mb_input_ids[:, 1:]
                    logp_all_new = gather_logprobs(logits_new_s, labels_s)

                    v_all_new_s = values_new_all[:, :-1]
                    logp_new, v_new, _ = extract_action_tensors(
                        logp_all_new,
                        v_all_new_s,
                        mb_prompt_lens,
                        mb_resp_lens,
                        pad_to=old_logp.size(1),
                    )

                # PPO policy loss
                log_ratio_raw = (logp_new - mb_old_logp).float()
                log_ratio = log_ratio_raw.clamp(
                    min=-float(args.log_ratio_clip),
                    max=float(args.log_ratio_clip),
                )
                ratio = torch.exp(log_ratio)
                ratio_clipped = torch.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.clip_range)
                pg1 = ratio * mb_adv
                pg2 = ratio_clipped * mb_adv
                pg_loss = -torch.minimum(pg1, pg2)
                pg_loss = masked_mean(pg_loss, mb_mask)
                approx_kl = masked_mean(0.5 * log_ratio_raw.pow(2), mb_mask)
                clipfrac = masked_mean(((ratio - 1.0).abs() > args.clip_range).to(torch.float32), mb_mask)
                valid_ratio = ratio[mb_mask.bool()]
                if valid_ratio.numel() > 0:
                    ratio_min = float(valid_ratio.min().detach().cpu())
                    ratio_max = float(valid_ratio.max().detach().cpu())
                else:
                    ratio_min = 1.0
                    ratio_max = 1.0

                # value loss
                v_pred = v_new.float()
                v_old = mb_old_v.float()
                v_clipped = v_old + torch.clamp(v_pred - v_old, -args.clip_range_vf, args.clip_range_vf)
                vf1 = (v_pred - mb_ret.float()) ** 2
                vf2 = (v_clipped - mb_ret.float()) ** 2
                vf_loss = 0.5 * masked_mean(torch.maximum(vf1, vf2), mb_mask)

                loss = pg_loss + args.vf_coef * vf_loss
                if not torch.isfinite(loss).all():
                    raise RuntimeError(
                        f"[ppo] Non-finite loss at update={update} epoch={epoch} mini_batch_start={start}: "
                        f"pg={float(pg_loss.detach().cpu())} vf={float(vf_loss.detach().cpu())}"
                    )

                loss_to_backprop = loss / float(accum_steps)
                if use_scaler:
                    scaler.scale(loss_to_backprop).backward()
                else:
                    loss_to_backprop.backward()

                accum_count += 1

                ppo_stats["pg_loss"] += float(pg_loss.detach().cpu())
                ppo_stats["vf_loss"] += float(vf_loss.detach().cpu())
                ppo_stats["approx_kl"] += float(approx_kl.detach().cpu())
                ppo_stats["clipfrac"] += float(clipfrac.detach().cpu())
                ppo_stats["ratio_min"] = min(float(ppo_stats["ratio_min"]), ratio_min)
                ppo_stats["ratio_max"] = max(float(ppo_stats["ratio_max"]), ratio_max)
                stat_count += 1

                if accum_count % accum_steps == 0:
                    optimizer_step_with_clip()

        if accum_count % accum_steps != 0:
            optimizer_step_with_clip()

        if stat_count > 0:
            ppo_stats["pg_loss"] /= stat_count
            ppo_stats["vf_loss"] /= stat_count
            ppo_stats["approx_kl"] /= stat_count
            ppo_stats["clipfrac"] /= stat_count
        if not math.isfinite(float(ppo_stats["ratio_min"])):
            ppo_stats["ratio_min"] = 1.0

        post_diag_done = False
        if do_diag and bool(args.enable_post_update_diag):
            post_logp_cpu, _, post_action_mask_cpu, post_ref_logp_cpu, _ = recompute_action_state_on_padded_batch()
            post_record = post_update_diag_record(
                update=int(update),
                post_logp=post_logp_cpu,
                old_logp=old_logp_cpu,
                post_ref_logp=post_ref_logp_cpu,
                mask=post_action_mask_cpu,
                clip_range=float(args.clip_range),
            )
            update_diag_records.append(post_record)
            ratio_stats = post_record.get("post_ratio_old_new", {})
            logratio_stats = post_record.get("post_logratio_old_new", {})
            if isinstance(ratio_stats, dict):
                if ratio_stats.get("min") is not None:
                    ppo_stats["post_ratio_min"] = float(ratio_stats["min"])
                if ratio_stats.get("max") is not None:
                    ppo_stats["post_ratio_max"] = float(ratio_stats["max"])
            if post_record.get("post_approx_kl_old_new") is not None:
                ppo_stats["post_approx_kl"] = float(post_record["post_approx_kl_old_new"])
            if post_record.get("post_clipfrac_old_new_clip_range") is not None:
                ppo_stats["post_clipfrac"] = float(post_record["post_clipfrac_old_new_clip_range"])
            if isinstance(logratio_stats, dict):
                lr_min = logratio_stats.get("min")
                lr_max = logratio_stats.get("max")
                vals = [abs(float(x)) for x in (lr_min, lr_max) if x is not None]
                if vals:
                    ppo_stats["post_logratio_abs_max"] = max(vals)

            outlier_records = []
            if response_token_ids_cpu.numel() > 0:
                outlier_records = ratio_outlier_records(
                    tokenizer=tokenizer,
                    update=int(update),
                    old_logp=old_logp_cpu,
                    new_logp=post_logp_cpu,
                    ref_logp=post_ref_logp_cpu,
                    advantages=advantages_cpu,
                    returns=returns_cpu,
                    mask=post_action_mask_cpu,
                    response_token_ids=response_token_ids_cpu,
                    prompt_records=prompt_records,
                    threshold=float(args.ratio_outlier_threshold),
                    topk=int(args.ratio_outlier_topk),
                    low_threshold=float(args.ratio_low_outlier_threshold),
                )
                _append_jsonl(ratio_outlier_log_path, outlier_records)
            if outlier_records:
                high_outlier_count = sum(1 for r in outlier_records if str(r.get("direction", "high")) == "high")
                low_outlier_count = sum(1 for r in outlier_records if str(r.get("direction", "")) == "low")
                logger.warning(
                    "[PPO-post] upd=%d ratio_outliers=%d high=%d low=%d max_ratio=%.4g min_ratio=%.4g written=%s",
                    int(update),
                    int(len(outlier_records)),
                    int(high_outlier_count),
                    int(low_outlier_count),
                    float(ppo_stats["post_ratio_max"]),
                    float(ppo_stats["post_ratio_min"]),
                    ratio_outlier_log_path,
                )
            if isinstance(ratio_stats, dict) and isinstance(logratio_stats, dict):
                logger.info(
                    "[PPO-post] approx_kl=%.6f clipfrac@%.3f=%.4f ratio=[%.4g, %.4g] ratio_p01=%s ratio_p05=%s ratio_p95=%s ratio_p99=%s logratio_p95=%s logratio_p99=%s",
                    float(ppo_stats["post_approx_kl"]),
                    float(args.clip_range),
                    float(ppo_stats["post_clipfrac"]),
                    float(ppo_stats["post_ratio_min"]),
                    float(ppo_stats["post_ratio_max"]),
                    str(ratio_stats.get("p01")),
                    str(ratio_stats.get("p05")),
                    str(ratio_stats.get("p95")),
                    str(ratio_stats.get("p99")),
                    str(logratio_stats.get("p95")),
                    str(logratio_stats.get("p99")),
                )
            post_diag_done = True

        if do_diag:
            update_diag_records.append(
                {
                    "type": "ppo_train_stats",
                    "update": int(update),
                    "pg_loss": _float_or_none(ppo_stats["pg_loss"]),
                    "vf_loss": _float_or_none(ppo_stats["vf_loss"]),
                    "grad_norm": _float_or_none(ppo_stats["grad_norm"]),
                    "approx_kl": _float_or_none(ppo_stats["approx_kl"]),
                    "clipfrac": _float_or_none(ppo_stats["clipfrac"]),
                    "ratio_min": _float_or_none(ppo_stats["ratio_min"]),
                    "ratio_max": _float_or_none(ppo_stats["ratio_max"]),
                    "raw_kl_seq_mean": _float_or_none(ppo_stats["raw_kl_seq_mean"]),
                    "kl_reward_sum": _float_or_none(ppo_stats["kl_reward_sum"]),
                    "post_approx_kl": _float_or_none(ppo_stats["post_approx_kl"]),
                    "post_clipfrac": _float_or_none(ppo_stats["post_clipfrac"]),
                    "post_ratio_min": _float_or_none(ppo_stats["post_ratio_min"]),
                    "post_ratio_max": _float_or_none(ppo_stats["post_ratio_max"]),
                    "post_logratio_abs_max": _float_or_none(ppo_stats["post_logratio_abs_max"]),
                }
            )

        if do_diag and update_diag_records:
            _append_jsonl(ppo_diag_path, update_diag_records)

        # logging
        mean_fused = float(fused_cpu.mean().item())
        std_fused = float(fused_cpu.std(unbiased=False).item())
        logger.info(
            "[PPO] upd=%d | B=%d (micro=%d x accum=%d) | fused=%.4f+/-%.4f | alpha(d,g,e)=(%.3f,%.3f,%.3f) | kl=%.4f | pg=%.4f vf=%.4f",
            update,
            B,
            int(args.batch_size),
            int(args.rollout_accum_steps),
            mean_fused,
            std_fused,
            fusion_info["alpha_disc"],
            fusion_info["alpha_gen"],
            fusion_info["alpha_endo"],
            ppo_stats["kl"],
            ppo_stats["pg_loss"],
            ppo_stats["vf_loss"],
        )
        logger.info(
            "[PPO] grad_norm=%.4f | approx_kl=%.6f clipfrac=%.4f ratio=[%.4g, %.4g] kl_reward_sum=%.4f raw_kl_seq=%.4f",
            ppo_stats["grad_norm"],
            ppo_stats["approx_kl"],
            ppo_stats["clipfrac"],
            ppo_stats["ratio_min"],
            ppo_stats["ratio_max"],
            ppo_stats["kl_reward_sum"],
            ppo_stats["raw_kl_seq_mean"],
        )

        post_update_stop_reason = ""
        if bool(args.enable_ppo_watchdog) and update > int(args.ppo_watchdog_warmup):
            watchdog_approx_kl = float(ppo_stats["post_approx_kl"]) if post_diag_done else float(ppo_stats["approx_kl"])
            watchdog_ratio_max = float(ppo_stats["post_ratio_max"]) if post_diag_done else float(ppo_stats["ratio_max"])
            watchdog_ratio_min = float(ppo_stats["post_ratio_min"]) if post_diag_done else float(ppo_stats["ratio_min"])
            watchdog_logratio_abs = float(ppo_stats["post_logratio_abs_max"]) if post_diag_done else 0.0
            if float(ppo_stats["pg_loss"]) > float(args.pg_loss_stop):
                post_update_stop_reason = f"pg_loss={ppo_stats['pg_loss']:.4f} > {float(args.pg_loss_stop):.4f}"
            elif float(ppo_stats["pg_loss"]) < -float(args.pg_loss_neg_warn):
                logger.warning(
                    "[watchdog] upd=%d pg_loss=%.4f < -%.4f (warning only)",
                    int(update),
                    float(ppo_stats["pg_loss"]),
                    float(args.pg_loss_neg_warn),
                )
            elif float(ppo_stats["grad_norm"]) > float(args.grad_norm_stop):
                post_update_stop_reason = f"grad_norm={ppo_stats['grad_norm']:.4f} > {float(args.grad_norm_stop):.4f}"
            elif watchdog_approx_kl > float(args.approx_kl_stop):
                src = "post_approx_kl" if post_diag_done else "approx_kl"
                post_update_stop_reason = f"{src}={watchdog_approx_kl:.6f} > {float(args.approx_kl_stop):.6f}"
            elif watchdog_ratio_max > float(args.ratio_max_stop):
                src = "post_ratio_max" if post_diag_done else "ratio_max"
                post_update_stop_reason = f"{src}={watchdog_ratio_max:.4g} > {float(args.ratio_max_stop):.4g}"
            elif float(args.ratio_min_stop) > 0 and watchdog_ratio_min < float(args.ratio_min_stop):
                src = "post_ratio_min" if post_diag_done else "ratio_min"
                post_update_stop_reason = f"{src}={watchdog_ratio_min:.4g} < {float(args.ratio_min_stop):.4g}"
            elif (
                bool(post_diag_done)
                and float(args.post_logratio_abs_max_stop) > 0
                and watchdog_logratio_abs > float(args.post_logratio_abs_max_stop)
            ):
                post_update_stop_reason = (
                    f"post_logratio_abs_max={watchdog_logratio_abs:.4g} > "
                    f"{float(args.post_logratio_abs_max_stop):.4g}"
                )
        if post_update_stop_reason:
            logger.warning("[STOP] PPO watchdog triggered after update %d: %s", update, post_update_stop_reason)
            ckpt_dir = save_ppo_checkpoint(update, suffix="_ppo_stop")
            logger.info("[PPO] Saved watchdog checkpoint to %s", ckpt_dir)
            break

        # save checkpoint
        if update % args.save_every == 0 or update == args.total_updates:
            ckpt_dir = save_ppo_checkpoint(update)
            logger.info("[PPO] Saved checkpoint to %s", ckpt_dir)

    logger.info("[PPO] Done. total_time=%.1fs", time.time() - t0)


# --------------------------- main DPO training --------------------------- #

def train_dpo(
    *,
    args,
    logger: logging.Logger,
    tokenizer,
    prompts_all: List[str],
    bundle: PolicyBundle,
    reward_gen: Optional[RewardGenScorer],
    sample_log_path: Optional[str] = None,
) -> None:
    policy_model = bundle.model
    reward_head = bundle.reward_head
    device = bundle.input_device

    compute_dtype = choose_compute_dtype()
    use_amp = torch.cuda.is_available() and compute_dtype in (torch.float16, torch.bfloat16)
    amp_dtype = compute_dtype if use_amp else torch.float32
    use_scaler = bool(use_amp and amp_dtype == torch.float16)
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    stop_cfg = build_stop_token_config(tokenizer, args)
    stop_token_ids = [int(x) for x in stop_cfg["stop_token_ids"]]
    generation_eos = generation_eos_token_arg(stop_token_ids)

    # optimizer (LoRA only)
    policy_named_params = collect_trainable_named(policy_model)
    if not policy_named_params:
        raise RuntimeError("[dpo] No trainable policy parameters found.")
    trainable_params = [p for _, p in policy_named_params]
    policy_scalars = int(sum(p.numel() for _, p in policy_named_params))
    logger.info(
        "[dpo] Trainable tensors: policy=%d | scalars policy=%d",
        len(policy_named_params),
        policy_scalars,
    )

    optimizer = None
    if args.optim.lower() in ("paged_adamw_8bit", "adamw_8bit"):
        try:
            import bitsandbytes as bnb  # type: ignore
            optim_cls = getattr(bnb.optim, "PagedAdamW8bit", None) if "paged" in args.optim.lower() else getattr(bnb.optim, "AdamW8bit", None)
            if optim_cls is None:
                raise AttributeError("bitsandbytes optimizer class not found")
            optimizer = optim_cls(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
            logger.info("[dpo] Using bitsandbytes optimizer: %s", args.optim)
        except Exception as e:
            # STRICT
            raise RuntimeError(f"[dpo] Failed to initialize bitsandbytes optimizer '{args.optim}': {e}")

    if optimizer is None:
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    fusion = MultiRewardFusion(ema_beta=args.fusion_ema_beta, eps=args.fusion_eps)

    # ------------------ Auto Resume Logic ------------------ #
    start_step = 1
    latest_ckpt_dir = None

    if getattr(args, "resume_checkpoint", None):
        if not os.path.exists(args.resume_checkpoint):
             raise FileNotFoundError(f"Resume checkpoint not found at: {args.resume_checkpoint}")
        latest_ckpt_dir = args.resume_checkpoint
        try:
            latest_step = int(re.search(r"checkpoint_(\d+)", os.path.basename(latest_ckpt_dir)).group(1))
        except Exception:
            latest_step = 0
    else:
        latest_ckpt_dir, latest_step = find_latest_checkpoint(args.output_dir)

    resume_policy_adapter_meta: Dict[str, object] = {
        "checkpoint_root": "start_from_base",
        "validated_checkpoint": True,
        "unvalidated_parent_allowed": False,
        "root_adapter_resume_allowed": False,
    }
    if latest_ckpt_dir:
        logger.info(f"[dpo] Found checkpoint: {latest_ckpt_dir} (step {latest_step}). Resuming...")
        allow_unvalidated_parent = bool(getattr(args, "allow_unvalidated_parent", False))
        if not allow_unvalidated_parent:
            require_validated_checkpoint_for_resume(
                latest_ckpt_dir,
                logger=logger,
                prefix="[dpo]",
                require_value_head=False,
            )

        # 1. Load Policy Adapter Weights
        from peft import set_peft_model_state_dict
        adapter_path, adapter_file = resolve_policy_adapter_for_resume(
            latest_ckpt_dir,
            allow_root_adapter_resume=bool(getattr(args, "allow_root_adapter_resume", False)),
            logger=logger,
            prefix="[dpo]",
        )
        resume_policy_adapter_meta = {
            "resume_checkpoint": latest_ckpt_dir,
            "resume_step": int(latest_step),
            "resume_policy_adapter_dir": adapter_path,
            "resume_policy_adapter_file": adapter_file,
            "parent_checkpoint_complete_file": os.path.join(latest_ckpt_dir, "checkpoint_complete.json"),
            "validated_checkpoint": not allow_unvalidated_parent,
            "unvalidated_parent_allowed": allow_unvalidated_parent,
            "root_adapter_resume_allowed": bool(getattr(args, "allow_root_adapter_resume", False)),
        }
        logger.info(f"[dpo] Loading policy adapter weights from {adapter_file}")
        if adapter_file.endswith(".safetensors"):
            from safetensors.torch import load_file
            sd = load_file(adapter_file, device="cpu")
        else:
            sd = torch.load(adapter_file, map_location="cpu")
        load_result = set_peft_model_state_dict(policy_model, sd, adapter_name=bundle.policy_adapter_name)
        missing = getattr(load_result, "missing_keys", []) or []
        unexpected = getattr(load_result, "unexpected_keys", []) or []
        tensors, total_abs_sum, lora_b_abs_sum = adapter_parameter_abs_sums(policy_model, bundle.policy_adapter_name)
        logger.info(
            "[dpo] PEFT load_result missing=%d unexpected=%d | policy_adapter tensors=%d abs_sum=%.6e lora_B_abs_sum=%.6e",
            len(missing),
            len(unexpected),
            tensors,
            total_abs_sum,
            lora_b_abs_sum,
        )
        critical_missing, critical_unexpected = adapter_load_blockers(
            [str(x) for x in missing],
            [str(x) for x in unexpected],
            bundle.policy_adapter_name,
        )
        if critical_missing or critical_unexpected:
            raise RuntimeError(
                "[dpo] adapter load mismatch: "
                f"missing={critical_missing[:20]} unexpected={critical_unexpected[:20]}"
            )
        if tensors == 0 or total_abs_sum == 0.0 or lora_b_abs_sum == 0.0:
            raise RuntimeError("[dpo] loaded policy adapter appears empty/no-op")

        # 2. Load Fusion State
        fs_path = os.path.join(latest_ckpt_dir, "fusion_state.json")
        if os.path.exists(fs_path):
            logger.info(f"[dpo] Loading fusion state from {fs_path}")
            with open(fs_path, "r") as f:
                fusion.load_state_dict(json.load(f))

        # 3. Load Optimizer State
        opt_path = os.path.join(latest_ckpt_dir, "optimizer.pt")
        if os.path.exists(opt_path):
            logger.info(f"[dpo] Loading optimizer state from {opt_path}")
            try:
                optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))
            except Exception as e:
                logger.warning(f"[dpo] Failed to load optimizer state: {e}")

        start_step = latest_step + 1
        logger.info(f"[dpo] Resumed. Starting loop from step {start_step}.")

    t0 = time.time()
    prompt_idx = 0

    for step in range(start_step, args.dpo_updates + 1):
        # 1) sample prompts
        batch_prompts = [prompts_all[(prompt_idx + i) % len(prompts_all)] for i in range(args.batch_size)]
        prompt_idx = (prompt_idx + args.batch_size) % len(prompts_all)

        # 2) generate K candidates per prompt (online preference)
        K = int(args.dpo_candidates)
        assert K >= 2

        gen_prompts = format_prompts_for_generation(
            tokenizer,
            batch_prompts,
            use_chat_template=bool(args.use_chat_template),
            chat_system_prompt=str(args.chat_system_prompt) if str(args.chat_system_prompt) else None,
            logger=logger,
        )
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"
        add_special = False if bool(args.use_chat_template) else True
        tok = tokenizer(
            gen_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_prompt_tokens,
            add_special_tokens=add_special,
        )
        input_ids_gen = tok["input_ids"].to(device)
        attn_gen = tok["attention_mask"].to(device)

        policy_model.set_adapter(bundle.policy_adapter_name)
        policy_model.eval()
        unsloth_for_inference(policy_model)

        with torch.inference_mode():
            gen = policy_model.generate(
                input_ids=input_ids_gen,
                attention_mask=attn_gen,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                top_p=args.top_p,
                temperature=args.temperature,
                num_return_sequences=K,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=generation_eos,
                use_cache=not bool(getattr(args, "load_in_8bit", False)),
            )

        unsloth_for_training(policy_model, use_gradient_checkpointing=True)

        # gen shape: [B*K, ctx+new]
        ctx_len = int(input_ids_gen.size(1))
        all_prompt_ids: List[torch.Tensor] = []
        all_resp_ids: List[torch.Tensor] = []
        all_prompts: List[str] = []

        # duplicate prompts K times (same order as num_return_sequences)
        for i in range(len(batch_prompts)):
            prompt_len = int(attn_gen[i].sum().item())
            prompt_ids = input_ids_gen[i, ctx_len - prompt_len : ctx_len].detach().cpu()
            for k in range(K):
                idx = i * K + k
                resp_ids = truncate_response_ids(
                    gen[idx, ctx_len:],
                    pad_token_id=getattr(tokenizer, "pad_token_id", None),
                    stop_token_ids=stop_token_ids,
                )
                all_prompt_ids.append(prompt_ids)
                all_resp_ids.append(resp_ids)
                all_prompts.append(batch_prompts[i])

        packed_all = build_packed_batch(tokenizer, all_prompts, all_resp_ids, all_prompt_ids)
        if sample_log_path:
            records: List[Dict[str, object]] = []
            for i, (p, r) in enumerate(zip(packed_all.prompts, packed_all.responses)):
                records.append(
                    {
                        "step": int(step),
                        "sample_idx": int(i),
                        "prompt_idx": int(i // K),
                        "candidate_idx": int(i % K),
                        "prompt": p,
                        "response": r,
                    }
                )
            _append_jsonl(sample_log_path, records)

        # 3) compute reward streams for each candidate (CPU)
        input_ids_all = packed_all.input_ids.to(device)
        attention_all = packed_all.attention_mask.to(device)

        # ref adapter forward
        policy_model.set_adapter(bundle.ref_adapter_name)
        policy_model.eval()
        with torch.no_grad():
            hidden_ref, logits_ref = forward_hidden_and_logits(policy_model, input_ids_all, attention_all)
            logits_ref_s = logits_ref[:, :-1, :]
            labels_s = input_ids_all[:, 1:]
            logp_all_ref = gather_logprobs(logits_ref_s, labels_s)  # [B*K, L-1]
            logp_ref_sum = sum_response_logprobs(logp_all_ref, packed_all.prompt_lens, packed_all.response_lens)  # [B*K]

            rewards: Dict[str, torch.Tensor] = {}
            if "disc" in args.reward:
                r_d = reward_head(hidden_ref, attention_all).float().detach().cpu()
                rewards["disc"] = r_d
            if "endo" in args.reward:
                entropy_all_ref = token_entropy_from_logits(logits_ref_s)
                entropy_ref_tok, _, mask_tok = extract_action_tensors(
                    entropy_all_ref,
                    entropy_all_ref,
                    packed_all.prompt_lens,
                    packed_all.response_lens,
                )
                entropy_ref_tok_cpu = entropy_ref_tok.detach().float().cpu()
                mask_cpu = mask_tok.detach().cpu().to(dtype=entropy_ref_tok_cpu.dtype)
                r_e = -((entropy_ref_tok_cpu * mask_cpu).sum(dim=1) / mask_cpu.sum(dim=1).clamp(min=1.0))
                rewards["endo"] = r_e.to(torch.float32)

        # restore policy adapter
        policy_model.set_adapter(bundle.policy_adapter_name)

        # gen reward
        if "gen" in args.reward:
            if reward_gen is None:
                raise RuntimeError("reward_gen is required when --reward includes 'gen'")
            r_g = reward_gen.score_batch(packed_all.prompts, packed_all.responses, max_length=args.max_len_reward_models)
            rewards["gen"] = r_g.to(torch.float32)

        # fuse reward across all candidates
        fused_all, fusion_info = fusion.fuse(rewards, active=args.reward, logger=logger, verbose=True)
        fused_all = fused_all.clamp(min=-args.reward_clip, max=args.reward_clip)  # [B*K] CPU

        # 4) pick chosen/rejected per prompt
        fused_np = fused_all.numpy().reshape(args.batch_size, K)
        chosen_indices: List[int] = []
        rejected_indices: List[int] = []
        for i in range(args.batch_size):
            best = int(fused_np[i].argmax())
            worst = int(fused_np[i].argmin())
            chosen_indices.append(i * K + best)
            rejected_indices.append(i * K + worst)

        # 5) build chosen & rejected batches
        def _slice_packed(packed: PackedBatch, indices: List[int]) -> PackedBatch:
            idx = torch.tensor(indices, dtype=torch.long)
            return PackedBatch(
                input_ids=packed.input_ids[idx].clone(),
                attention_mask=packed.attention_mask[idx].clone(),
                prompt_lens=packed.prompt_lens[idx].clone(),
                response_lens=packed.response_lens[idx].clone(),
                prompts=[packed.prompts[i] for i in indices],
                responses=[packed.responses[i] for i in indices],
            )

        packed_c = _slice_packed(packed_all, chosen_indices)
        packed_r = _slice_packed(packed_all, rejected_indices)

        # 6) compute logp under policy and ref for chosen/rejected
        # forward policy (trainable)
        policy_model.train()
        policy_model.set_adapter(bundle.policy_adapter_name)

        def _logp_sum_under_model(packed: PackedBatch, adapter_name: str, train: bool) -> torch.Tensor:
            policy_model.set_adapter(adapter_name)
            policy_model.train(mode=train)
            ids = packed.input_ids.to(device)
            attn = packed.attention_mask.to(device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                hidden, logits = forward_hidden_and_logits(policy_model, ids, attn)
                logits_s = logits[:, :-1, :]
                labels_s = ids[:, 1:]
                logp_all = gather_logprobs(logits_s, labels_s)
                logp_sum = sum_response_logprobs(logp_all, packed.prompt_lens.to(device), packed.response_lens.to(device))
            return logp_sum.float()

        # policy logps (need grad)
        pi_logp_c = _logp_sum_under_model(packed_c, bundle.policy_adapter_name, train=True)
        pi_logp_r = _logp_sum_under_model(packed_r, bundle.policy_adapter_name, train=True)

        # ref logps (no grad)
        with torch.no_grad():
            ref_logp_c = _logp_sum_under_model(packed_c, bundle.ref_adapter_name, train=False)
            ref_logp_r = _logp_sum_under_model(packed_r, bundle.ref_adapter_name, train=False)

        loss = dpo_loss(pi_logp_c, pi_logp_r, ref_logp_c, ref_logp_r, beta=args.dpo_beta)
        if not torch.isfinite(loss).all():
            raise RuntimeError(f"[dpo] Non-finite loss at step={step}: {float(loss.detach().cpu())}")

        optimizer.zero_grad(set_to_none=True)
        grad_norm = 0.0
        if use_scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = grad_norm_to_float(trainable_params, args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = grad_norm_to_float(trainable_params, args.max_grad_norm)
            optimizer.step()

        if step % args.log_every == 0:
            logger.info(
                "[DPO] step=%d/%d | loss=%.4f | alpha(d,g,e)=(%.3f,%.3f,%.3f) | chosen_mean=%.3f rejected_mean=%.3f grad_norm=%.4f",
                step,
                args.dpo_updates,
                float(loss.detach().cpu()),
                fusion_info["alpha_disc"],
                fusion_info["alpha_gen"],
                fusion_info["alpha_endo"],
                float(fused_all[chosen_indices].mean().item()),
                float(fused_all[rejected_indices].mean().item()),
                grad_norm,
            )

        if step % args.save_every == 0 or step == args.dpo_updates:
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint_{step}")
            os.makedirs(ckpt_dir, exist_ok=True)

            adapter_dir = os.path.join(ckpt_dir, "policy_adapter")
            trained_adapter_dir, trained_adapter_file = save_policy_adapter(
                policy_model,
                tokenizer,
                adapter_dir,
                bundle.policy_adapter_name,
                logger=logger,
                prefix="[dpo]",
            )
            write_checkpoint_manifest(
                ckpt_dir,
                adapter_dir,
                bundle.policy_adapter_name,
                trained_adapter_dir,
                trained_adapter_file,
                extra={
                    **resume_policy_adapter_meta,
                    "policy": str(args.policy),
                    "reward_set": list(args.reward),
                    "reward_disc_level": str(getattr(args, "reward_disc_level", "")),
                    "reward_disc_head": str(getattr(args, "reward_disc_head", "")),
                    "reward_gen_lora": str(getattr(args, "reward_gen_lora", "")),
                    "base_model": str(getattr(args, "base_model", "")),
                    "max_new_tokens": int(getattr(args, "max_new_tokens", 0)),
                    "max_seq_length": int(getattr(args, "max_seq_length", 0)),
                    "max_len_reward_models": int(getattr(args, "max_len_reward_models", 0)),
                    "trainer": "rala_code",
                },
                logger=logger,
                prefix="[dpo]",
            )

            fusion_state_path = os.path.join(ckpt_dir, "fusion_state.json")
            optimizer_path = os.path.join(ckpt_dir, "optimizer.pt")
            with open(fusion_state_path, "w", encoding="utf-8") as f:
                json.dump(fusion.state_dict(), f, ensure_ascii=False, indent=2)

            # optimizer state
            torch.save(optimizer.state_dict(), optimizer_path)
            write_checkpoint_complete(
                ckpt_dir,
                update_idx=step,
                trained_adapter_dir=trained_adapter_dir,
                trained_adapter_file=trained_adapter_file,
                fusion_state_path=fusion_state_path,
                optimizer_path=optimizer_path,
                logger=logger,
                prefix="[dpo]",
            )

            logger.info("[DPO] Saved checkpoint to %s", ckpt_dir)

        # cleanup
        del tok, input_ids_gen, attn_gen, gen, input_ids_all, attention_all

    logger.info("[DPO] Done. total_time=%.1fs", time.time() - t0)


# --------------------------- CLI + main --------------------------- #

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("RALA RLHF with Unsloth + QLoRA (single GPU)")

    # required by request
    p.add_argument("--reward", nargs="*", default=None,
                   help='Reward streams to fuse. Example: --reward disc endo ; or --reward \'["disc","endo"]\' . '
                        'Allowed: disc gen endo. Default: all.')
    p.add_argument("--policy", type=str, default="PPO", choices=["PPO", "DPO", "ppo", "dpo"],
                   help="RLHF policy algorithm: PPO or DPO.")

    # paths
    p.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL_ID)
    p.add_argument("--reward_disc_head", type=str, default=DEFAULT_REWARD_DISC_HEAD_PATH)
    p.add_argument("--reward_disc_level", type=str, default="auto", choices=["auto", "embedding_level", "token_level"],
                   help="Discriminative reward scoring granularity. auto reads training_meta.json and falls back to embedding_level.")
    p.add_argument("--reward_gen_lora", type=str, default=DEFAULT_REWARD_GEN_DIR)
    p.add_argument("--data", type=str, default=DEFAULT_RLHF_DATA_PATH)
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--sft_lora", type=str, default=os.environ.get("SFT_LORA", ""),
                   help="Dual-head SFT LoRA adapter directory used to initialize frozen ref and trainable policy adapters.")
    p.add_argument("--init_from_base_model", action="store_true", default=False,
                   help="Use --base_model directly as the SFT/merged policy initialization instead of loading --sft_lora.")
    p.add_argument("--prompt_style", type=str, default="instruction", choices=["instruction", "completion", "raw"],
                   help="How to interpret dataset prompts. completion tries prefix/original_prompt/prompt.")
    p.add_argument("--prompt_field", type=str, default="",
                   help="If set, use this JSON field as prompt (overrides prompt_style).")
    p.add_argument("--use_chat_template", action="store_true", default=False,
                   help="Wrap prompts with tokenizer.chat_template before generation.")
    p.add_argument("--chat_system_prompt", type=str, default="",
                   help="Optional system prompt when --use_chat_template is set.")

    # qlora
    p.add_argument("--load_in_4bit", action="store_true", default=DEFAULT_LOAD_IN_4BIT)
    p.add_argument("--load_in_8bit", action="store_true", default=DEFAULT_LOAD_IN_8BIT,
                   help="Use Unsloth 8bit loading instead of 4bit/full precision.")
    p.add_argument("--no_8bit", action="store_true", help="Disable 8bit loading.")
    p.add_argument("--no_4bit", action="store_true", help="Disable 4bit loading.")
    p.add_argument("--max_seq_length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)

    # LoRA config for explicit base/merged-model initialization
    p.add_argument("--lora_r", type=int, default=DEFAULT_LORA_R,
                   help="LoRA rank")
    p.add_argument("--lora_alpha", type=int, default=DEFAULT_LORA_ALPHA,
                   help="LoRA alpha")
    p.add_argument("--lora_dropout", type=float, default=DEFAULT_LORA_DROPOUT,
                   help="LoRA dropout")

    # generation
    p.add_argument("--max_prompt_tokens", type=int, default=DEFAULT_MAX_PROMPT_TOKENS)
    p.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument("--do_sample", action="store_true", default=DEFAULT_DO_SAMPLE)
    p.add_argument("--no_sample", dest="do_sample", action="store_false",
                   help="Disable sampling for generation (greedy decoding).")
    p.add_argument("--top_p", type=float, default=DEFAULT_TOP_P)
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--use_pad_token_as_eos", action="store_true", default=False,
                   help="Also treat tokenizer.pad_token_id as a generation stop token.")
    p.add_argument("--extra_eos_token_ids", type=str, nargs="*", default=None,
                   help="Extra generation stop token ids. Supports repeated args, comma-separated, or JSON list.")
    p.add_argument("--strict_prompt_tail_check", action="store_true", default=False,
                   help="Fail if a chat-templated generation prompt tail lacks an assistant/generation marker.")

    # PPO
    p.add_argument("--total_updates", type=int, default=DEFAULT_TOTAL_UPDATES)
    p.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--rollout_accum_steps", type=int, default=DEFAULT_ROLLOUT_ACCUM_STEPS)
    p.add_argument("--ppo_epochs", type=int, default=DEFAULT_PPO_EPOCHS)
    p.add_argument("--mini_batch_size", type=int, default=DEFAULT_MINI_BATCH_SIZE)
    p.add_argument("--ppo_grad_accum_steps", type=int, default=DEFAULT_PPO_GRAD_ACCUM_STEPS)
    p.add_argument("--allow_single_step_surrogate", action="store_true", default=False,
                   help="Allow ppo_epochs=1 with ppo_grad_accum_steps covering the whole rollout batch.")

    # DPO
    p.add_argument("--dpo_updates", type=int, default=DEFAULT_DPO_UPDATES)
    p.add_argument("--dpo_beta", type=float, default=DEFAULT_DPO_BETA)
    p.add_argument("--dpo_candidates", type=int, default=DEFAULT_DPO_CANDIDATES)

    # optimization
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    p.add_argument("--max_grad_norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
    p.add_argument("--optim", type=str, default="adamw",
                   help="Optimizer: paged_adamw_8bit / adamw_8bit / adamw")

    # PPO losses
    p.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    p.add_argument("--lam", type=float, default=DEFAULT_LAMBDA_GAE)
    p.add_argument("--clip_range", type=float, default=DEFAULT_CLIP_RANGE)
    p.add_argument("--clip_range_vf", type=float, default=DEFAULT_CLIP_RANGE_VF)
    p.add_argument("--vf_coef", type=float, default=DEFAULT_VF_COEF)
    p.add_argument("--ent_coef", type=float, default=DEFAULT_ENT_COEF)
    p.add_argument("--kl_coef", type=float, default=DEFAULT_KL_COEF)
    p.add_argument("--kl_token_clip", type=float, default=DEFAULT_KL_TOKEN_CLIP,
                   help="Clip signed per-token policy-vs-ref KL before applying length-normalized KL reward.")
    p.add_argument("--log_ratio_clip", type=float, default=DEFAULT_LOG_RATIO_CLIP,
                   help="Clamp PPO log_ratio before exponentiating to ratio.")
    p.add_argument("--enable_kl_watchdog", action="store_true", default=False,
                   help="Stop PPO early when raw signed KL drifts too negative.")
    p.add_argument("--kl_watchdog_warmup", type=int, default=50)
    p.add_argument("--neg_kl_stop", type=float, default=5.0,
                   help="Stop if raw sequence-mean signed KL is below -this value after warmup.")
    p.add_argument("--neg_kl_rolling_stop", type=float, default=2.0,
                   help="Stop if rolling raw sequence-mean signed KL is below -this value after warmup.")
    p.add_argument("--neg_kl_rolling_window", type=int, default=5)
    p.add_argument("--enable_ppo_watchdog", action="store_true", default=False,
                   help="Stop PPO early on pg/grad/ratio/approx-KL instability.")
    p.add_argument("--ppo_watchdog_warmup", type=int, default=20)
    p.add_argument("--pg_loss_stop", type=float, default=10.0)
    p.add_argument("--pg_loss_neg_warn", type=float, default=10.0,
                   help="Warn when pg_loss goes below -this value after PPO watchdog warmup, but do not stop.")
    p.add_argument("--grad_norm_stop", type=float, default=1000.0)
    p.add_argument("--approx_kl_stop", type=float, default=0.05)
    p.add_argument("--ratio_max_stop", type=float, default=50.0)
    p.add_argument("--ratio_min_stop", type=float, default=0.0,
                   help="Stop if post-update ratio min falls below this positive threshold.")
    p.add_argument("--post_logratio_abs_max_stop", type=float, default=0.0,
                   help="Stop if post-update max(abs(logratio)) exceeds this positive threshold.")
    p.add_argument("--ppo_diag_every", type=int, default=1,
                   help="Write PPO diagnostic JSONL every N updates. Set 0 to disable.")
    p.add_argument("--enable_post_update_diag", action="store_true", default=True,
                   help="After optimizer.step(), recompute policy/ref logprobs and log true post-update ratio/KL.")
    p.add_argument("--disable_post_update_diag", dest="enable_post_update_diag", action="store_false",
                   help="Disable post-update logprob diagnostics.")
    p.add_argument("--ratio_outlier_threshold", type=float, default=10.0,
                   help="Write top token-level ratio outliers above this post-update ratio.")
    p.add_argument("--ratio_low_outlier_threshold", type=float, default=10.0,
                   help="Write bottom token-level ratio outliers below this cutoff. Values >1 are interpreted as reciprocal thresholds, e.g. 10 -> ratio < 0.1.")
    p.add_argument("--ratio_outlier_topk", type=int, default=8,
                   help="Max ratio outlier records per update.")

    # fusion
    p.add_argument("--fusion_ema_beta", type=float, default=DEFAULT_FUSION_EMA_BETA)
    p.add_argument("--fusion_eps", type=float, default=DEFAULT_FUSION_EPS)
    p.add_argument("--reward_clip", type=float, default=DEFAULT_REWARD_CLIP)
    p.add_argument("--enable_stop_reward_shaping", action="store_true", default=False,
                   help="Optionally add stop/truncation reward shaping before clipping. Default off.")
    p.add_argument("--trunc_penalty", type=float, default=0.0,
                   help="Subtract this fused reward amount when generation reaches max_new_tokens without a stop token.")
    p.add_argument("--eos_bonus", type=float, default=0.0,
                   help="Add this fused reward amount when a stop token is emitted after eos_bonus_min_tokens.")
    p.add_argument("--eos_bonus_min_tokens", type=int, default=0,
                   help="Minimum response length required for eos_bonus.")
    p.add_argument("--short_eos_penalty", type=float, default=0.0,
                   help="Subtract this amount when a stop token appears before eos_bonus_min_tokens.")

    # reward_gen
    p.add_argument("--max_len_reward_models", type=int, default=DEFAULT_MAX_LEN_REWARD_MODELS)

    # resume
    p.add_argument("--resume_checkpoint", type=str, default=None,
                   help="Path to specific checkpoint to resume from. If not provided, will try to auto-resume from output_dir.")
    p.add_argument("--allow_root_adapter_resume", action="store_true", default=False,
                   help="Allow resume from checkpoint_xxx/policy_adapter root instead of policy_adapter/policy.")
    p.add_argument("--allow_unvalidated_parent", action="store_true", default=False,
                   help="Allow resume from a checkpoint that does not contain validation metadata.")
    p.add_argument("--resume_optimizer", type=str, default="auto", choices=["auto", "keep", "drop"],
                   help="Optimizer resume mode for PPO checkpoints: auto loads optimizer.pt when present, keep requires it, drop skips it.")
    p.add_argument("--reset_optimizer_hparams_on_resume", dest="reset_optimizer_hparams_on_resume",
                   action="store_true", default=True,
                   help="After loading optimizer state, reset lr/weight_decay from CLI args.")
    p.add_argument("--no_reset_optimizer_hparams_on_resume", dest="reset_optimizer_hparams_on_resume",
                   action="store_false",
                   help="Keep lr/weight_decay from the loaded optimizer state.")

    # misc
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--log_every", type=int, default=1)

    # debug
    p.add_argument("--debug_tokenizer", action="store_true", default=False,
                   help="Log tokenizer/model alignment diagnostics (vocab sizes, special token ids, OOV ids).")
    p.add_argument("--debug_updates", type=int, default=1,
                   help="When debugging generation, log first N updates (only effective if --debug_rollouts > 0).")
    p.add_argument("--debug_rollouts", type=int, default=0,
                   help="Log token-level generation diagnostics for the first N rollouts per update.")
    p.add_argument("--debug_samples", type=int, default=2,
                   help="Log token-level diagnostics for the first N samples in a generated batch.")

    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    # normalize args
    args.policy = args.policy.upper()
    args.reward = parse_reward_list(args.reward)
    args.extra_eos_token_ids = parse_int_list(args.extra_eos_token_ids)
    args.sft_lora = str(args.sft_lora or "").strip()
    if args.sft_lora and args.init_from_base_model:
        raise SystemExit("Use either --sft_lora or --init_from_base_model, not both.")
    if not args.sft_lora and not args.init_from_base_model:
        raise SystemExit(
            "Default RLHF initialization requires --sft_lora "
            "or SFT_LORA. Use --init_from_base_model only when --base_model is already a merged SFT policy."
        )
    _require_arg(args.base_model, "base_model", "BASE_MODEL")
    _require_arg(args.data, "data", "RLHF_DATA")
    _require_arg(args.output_dir, "output_dir", "RLHF_OUTPUT_DIR")
    if "disc" in args.reward:
        _require_arg(args.reward_disc_head, "reward_disc_head", "REWARD_DISC_HEAD")
    if "gen" in args.reward:
        _require_arg(args.reward_gen_lora, "reward_gen_lora", "REWARD_GEN_LORA")

    if args.no_8bit:
        args.load_in_8bit = False
    if args.no_4bit:
        args.load_in_4bit = False
    if args.load_in_8bit:
        args.load_in_4bit = False

    # output dir
    args.output_dir = _abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    reward_order = ["disc", "gen", "endo"]
    active_rewards = [k for k in reward_order if k in args.reward]
    log_name = "_".join(active_rewards) if active_rewards else "rlhf_train"
    logger = setup_logging(os.path.join(args.output_dir, f"{log_name}.log"))
    sample_log_path = os.path.join(args.output_dir, "generate_samples.jsonl")

    # seed & perf flags
    set_seed(int(args.seed))
    torch.backends.cuda.matmul.allow_tf32 = True

    logger.info("==== RALA RLHF Unsloth + QLoRA (single GPU) ====")
    logger.info(
        "policy=%s | reward=%s | 4bit=%s | 8bit=%s",
        args.policy,
        str(args.reward),
        str(args.load_in_4bit),
        str(args.load_in_8bit),
    )
    logger.info("base_model=%s", args.base_model)
    logger.info("sft_lora=%s", args.sft_lora)
    logger.info("init_from_base_model=%s", str(bool(args.init_from_base_model)))
    logger.info("reward_disc_head=%s", args.reward_disc_head)
    logger.info("reward_disc_level=%s", args.reward_disc_level)
    logger.info("reward_gen_lora=%s", args.reward_gen_lora)
    logger.info("data=%s", args.data)
    logger.info("prompt_style=%s | prompt_field=%s | use_chat_template=%s",
                args.prompt_style, str(args.prompt_field), str(bool(args.use_chat_template)))
    if args.use_chat_template:
        logger.info("chat_system_prompt_len=%d", len(str(args.chat_system_prompt or "")))
    logger.info(
        "prompt_tail_check: strict=%s generation_markers=%s",
        str(bool(args.strict_prompt_tail_check)),
        str(PROMPT_GENERATION_MARKERS),
    )
    logger.info(
        "ppo_safety: kl_coef=%.6g kl_token_clip=%.3f log_ratio_clip=%.3f "
        "kl_watchdog=%s ppo_watchdog=%s post_diag=%s diag_every=%d allow_single_step_surrogate=%s "
        "ratio_stop=[min %.4g max %.4g] post_logratio_abs_max_stop=%.4g",
        float(args.kl_coef),
        float(args.kl_token_clip),
        float(args.log_ratio_clip),
        str(bool(args.enable_kl_watchdog)),
        str(bool(args.enable_ppo_watchdog)),
        str(bool(args.enable_post_update_diag)),
        int(args.ppo_diag_every),
        str(bool(args.allow_single_step_surrogate)),
        float(args.ratio_min_stop),
        float(args.ratio_max_stop),
        float(args.post_logratio_abs_max_stop),
    )
    logger.info(
        "resume_optimizer=%s reset_optimizer_hparams_on_resume=%s",
        str(args.resume_optimizer),
        str(bool(args.reset_optimizer_hparams_on_resume)),
    )
    logger.info(
        "stop_reward: enabled=%s trunc_penalty=%.4g eos_bonus=%.4g eos_bonus_min_tokens=%d short_eos_penalty=%.4g",
        str(bool(getattr(args, "enable_stop_reward_shaping", False))),
        float(args.trunc_penalty),
        float(args.eos_bonus),
        int(args.eos_bonus_min_tokens),
        float(args.short_eos_penalty),
    )
    if int(args.max_len_reward_models) > int(args.max_seq_length):
        logger.warning(
            "max_len_reward_models=%d exceeds max_seq_length=%d; reward model inputs may be truncated by the model.",
            int(args.max_len_reward_models),
            int(args.max_seq_length),
        )
    if int(args.max_prompt_tokens) + int(args.max_new_tokens) > int(args.max_seq_length):
        logger.warning(
            "max_prompt_tokens + max_new_tokens = %d exceeds max_seq_length=%d; PPO/reward inputs may be truncated.",
            int(args.max_prompt_tokens) + int(args.max_new_tokens),
            int(args.max_seq_length),
        )
    if float(args.ent_coef) != 0.0:
        logger.warning("ent_coef=%.6g is currently ignored by the PPO loss in this script.", float(args.ent_coef))
    logger.info("output_dir=%s", args.output_dir)
    logger.info("samples_log=%s", sample_log_path)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this RLHF script.")

    # tokenize
    # Load tokenizer once from the base model for stable preprocessing.
    tokenizer = AutoTokenizer.from_pretrained(_abspath(args.base_model), use_fast=True, trust_remote_code=True)
    ensure_pad_token(tokenizer)

    # dataset
    prompts_all = load_prompts(
        _abspath(args.data),
        prompt_style=str(args.prompt_style),
        prompt_field=str(args.prompt_field) if str(args.prompt_field).strip() else None,
    )
    logger.info("Loaded %d prompts", len(prompts_all))

    # load policy bundle
    need_value_head = (args.policy == "PPO")
    need_reward_head = ("disc" in args.reward)
    reward_disc_level = "embedding_level"
    if need_reward_head:
        reward_disc_level = resolve_reward_disc_level(
            _abspath(args.reward_disc_head),
            str(args.reward_disc_level),
            logger=logger,
        )
    bundle = load_policy_bundle_single_gpu(
        base_model_id=_abspath(args.base_model),
        sft_lora_path=_abspath(args.sft_lora) if args.sft_lora else "",
        reward_head_path=_abspath(args.reward_disc_head),
        reward_head_level=reward_disc_level,
        max_seq_length=int(args.max_seq_length),
        load_in_4bit=bool(args.load_in_4bit),
        load_in_8bit=bool(args.load_in_8bit),
        logger=logger,
        need_value_head=need_value_head,
        need_reward_head=need_reward_head,
        debug_tokenizer=bool(getattr(args, "debug_tokenizer", False)),
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    # Print vocab size after model load (requested).
    try:
        model_vocab = int(getattr(bundle.model.config, "vocab_size", -1))
    except Exception:
        model_vocab = -1
    logger.info("[policy] vocab_size=%d | tokenizer_len=%d | tokenizer_vocab_size=%d",
                model_vocab, len(tokenizer), int(getattr(tokenizer, "vocab_size", len(tokenizer))))
    if getattr(args, "debug_tokenizer", False):
        sample_texts: List[str] = []
        try:
            if len(prompts_all) > 0:
                sample_texts.append(prompts_all[0])
            if len(prompts_all) > 1:
                sample_texts.append(prompts_all[1])
        except Exception:
            sample_texts = []
        log_tokenizer_model_diagnostics(
            logger,
            tokenizer=tokenizer,
            model=bundle.model,
            prefix="[policy/main]",
            sample_texts=sample_texts if sample_texts else None,
            max_samples=2,
        )

    # load reward_gen scorer if needed
    reward_gen = None
    if "gen" in args.reward:
        # Default to shared mode to save VRAM
        shared_model = bundle.model
        restore = bundle.policy_adapter_name

        reward_gen = RewardGenScorer(
            base_model_id=_abspath(args.base_model),
            reward_gen_dir=_abspath(args.reward_gen_lora),
            tokenizer=tokenizer,
            max_seq_length=int(args.max_len_reward_models),
            load_in_4bit=bool(args.load_in_4bit),
            logger=logger,
            shared_model=shared_model,
            restore_adapter_name=restore,
        )
        # PEFT load_adapter(..., is_trainable=False) can reset requires_grad flags on the
        # shared model. Reactivate policy LoRA after reward_gen is attached.
        set_only_adapter_trainable(bundle.model, bundle.policy_adapter_name, logger=logger)
        bundle.model.set_adapter(bundle.policy_adapter_name)

    if args.policy == "PPO":
        train_ppo(
            args=args,
            logger=logger,
            tokenizer=tokenizer,
            prompts_all=prompts_all,
            bundle=bundle,
            reward_gen=reward_gen,
            sample_log_path=sample_log_path,
        )
    else:
        train_dpo(
            args=args,
            logger=logger,
            tokenizer=tokenizer,
            prompts_all=prompts_all,
            bundle=bundle,
            reward_gen=reward_gen,
            sample_log_path=sample_log_path,
        )


if __name__ == "__main__":
    main()
