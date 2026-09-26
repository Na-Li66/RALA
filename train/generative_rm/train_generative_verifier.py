"""Train the RALA generative verifier reward model.

The verifier estimates r_g(x, y) as the probability of answering "Yes" to a
fixed correctness prompt. The same verification prompt and Yes/No token
selection routine are used by the RALA scorer.

Expected input rows are JSON/JSONL objects with `prompt`, `chosen`, and
`rejected` fields. The output directory contains the verifier LoRA adapter,
the tokenizer files, and `verifier_config.json`.
"""

from __future__ import annotations

import os

# --------------------------- Runtime environment setup --------------------------- #

def _set_offline_env() -> None:
    """Set non-interactive Hugging Face runtime defaults."""
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_set_offline_env()

# Import Unsloth before Transformers/PEFT so its runtime patches are installed.
try:
    from unsloth import FastLanguageModel  # type: ignore
except Exception:
    FastLanguageModel = None  # noqa: N816

import argparse
import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

try:
    from transformers import BitsAndBytesConfig
except Exception:  # pragma: no cover
    BitsAndBytesConfig = None  # type: ignore

try:
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "peft is required for this script. Please ensure your environment matches the RLHF/SFT scripts."
    ) from exc


# --------------------------- configurable defaults --------------------------- #
# CLI arguments override these optional environment defaults.
MODEL_ID = os.environ.get("MODEL_ID", "")
SFT_LORA_DIR = os.environ.get("SFT_LORA_DIR", "")
PREF_DATA_PATH = os.environ.get("PREF_DATA_PATH", "")
OUTPUT_DIR = os.environ.get("REWARD_GEN_OUTPUT_DIR", os.environ.get("OUTPUT_DIR", ""))

# reward_gen LoRA config
REWARD_LORA_RANK = 32
REWARD_LORA_ALPHA = 64
REWARD_LORA_DROPOUT = 0.1
REWARD_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Sequence/training
MAX_LENGTH = 1024
EPOCHS = 3
LR = 2e-4
WEIGHT_DECAY = 0.0
PER_DEVICE_BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8
MAX_GRAD_NORM = 1.0
WARMUP_RATIO = 0.1

# Loss mixing: total = ce + LAMBDA_RANK * rank
LAMBDA_RANK = 1.5

# Verification prompt shared with the RALA scorer.
VERIFICATION_PROMPT = """Based on the question and the response provided, is the response correct and complete?
Answer with only "Yes" or "No".

Answer:"""

# Eval/logging
VAL_RATIO = 0.02
LOG_EVERY_OPT_STEPS = 10
EVAL_EVERY_OPT_STEPS = 100

# Optional row limit (0 = no limit)
MAX_ROWS = 0

SEED = 42

# Unsloth / QLoRA
DEFAULT_LOAD_IN_4BIT = True
DEFAULT_USE_GRAD_CHECKPOINTING = True

# Logging file (written into OUTPUT_DIR)
LOG_FILE = "reward_gen_train.log"
# --------------------------------------------------------------------------- #


# --------------------------- path & logging helpers --------------------------- #
def _script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _abspath(rel_or_abs: str) -> str:
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


def setup_logging(log_path: str) -> logging.Logger:
    logger = logging.getLogger("reward_gen")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    fh = logging.FileHandler(log_path, encoding="utf-8", mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    class _TqdmHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            try:
                tqdm.write(self.format(record))
            except Exception:
                self.handleError(record)

    sh: logging.Handler = _TqdmHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ensure_pad_token(tokenizer) -> None:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def _short(text: str, max_chars: int = 200) -> str:
    text = (text or "").replace("\n", "\\n")
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


# --------------------------- Unsloth helpers --------------------------- #
def choose_compute_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


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


def patch_unsloth_statistics_hooks(logger: Optional[logging.Logger] = None) -> None:
    """
    Disable optional Unsloth statistics hooks that may trigger Hub metadata
    access before model loading.
    """
    try:
        import importlib

        def _noop(*args, **kwargs):  # noqa: ANN001
            return None

        patched_count = 0
        for mod_name in [
            "unsloth.models._utils",
            "unsloth.models.llama",
            "unsloth.models.qwen2",
            "unsloth.models.qwen2_5",
            "unsloth.models.mistral",
        ]:
            try:
                mod = importlib.import_module(mod_name)
            except Exception:
                continue

            for fn_name in ["get_statistics", "_get_statistics", "stats_check"]:
                if hasattr(mod, fn_name):
                    try:
                        setattr(mod, fn_name, _noop)
                        patched_count += 1
                    except Exception:
                        pass

        if logger and patched_count > 0:
            logger.info("[runtime] Disabled %d optional Unsloth statistics hooks.", patched_count)
    except Exception:
        if logger:
            logger.info("[runtime] Unsloth statistics hook setup skipped.")


def load_base_model_qlora(
    *,
    model_name_or_path: str,
    max_seq_length: int,
    load_in_4bit: bool,
    logger: logging.Logger,
):
    """Load the base model with Unsloth when available, otherwise Transformers."""
    patch_unsloth_statistics_hooks(logger)

    dtype = choose_compute_dtype()

    if FastLanguageModel is not None:
        logger.info("[load] Using Unsloth FastLanguageModel.from_pretrained(load_in_4bit=%s)", str(load_in_4bit))

        kwargs = dict(
            model_name=model_name_or_path,
            max_seq_length=int(max_seq_length),
            dtype=None,  # let Unsloth pick bf16/fp16
            load_in_4bit=bool(load_in_4bit),
            local_files_only=True,
        )

        try:
            model, _tok = FastLanguageModel.from_pretrained(**kwargs)
            return model
        except Exception as e:
            msg = str(e)
            if "unslothai/other" in msg or "get_statistics" in msg or "snapshot_download" in msg:
                logger.info("[runtime] Retrying model load after disabling optional Unsloth statistics hooks.")
                patch_unsloth_statistics_hooks(logger)
                model, _tok = FastLanguageModel.from_pretrained(**kwargs)
                return model
            raise

    if load_in_4bit and BitsAndBytesConfig is None:
        raise RuntimeError("BitsAndBytesConfig not available and Unsloth not installed.")

    quant_cfg = None
    if load_in_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )

    logger.info("[load] Using Transformers loader (4bit=%s).", str(load_in_4bit))
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        device_map={"": 0} if torch.cuda.is_available() else None,
        trust_remote_code=True,
        quantization_config=quant_cfg,
        local_files_only=True,
    )
    return model


# --------------------------- YES/NO token ids --------------------------- #
def _get_yes_no_token_ids(tokenizer) -> Tuple[int, int]:
    """
    Keep synchronized with the RALA reward scorer.

    yes_candidates = ["Yes","yes"," Yes"," yes"]
    no_candidates  = ["No","no"," No"," no"]
    If no single-token match is found, use tokenizer.encode("Yes")[0] /
    tokenizer.encode("No")[0].
    """
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


# --------------------------- P1: robust truncation --------------------------- #
def _encode_verification_format(
    tokenizer,
    prompt: str,
    response: str,
    verification_prompt: str,
    max_length: int,
) -> Tuple[List[int], List[int]]:
    """
    Encode:
      [BOS] + "Question: " + prompt_head + "\n\nResponse: " + response_tail + "\n\n" + verification_prompt

    If too long, keep:
      - question head
      - response tail
      - verification prompt (always kept)
    """
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
    return input_ids, [1] * len(input_ids)


def _pad_2d(seqs: List[List[int]], pad_value: int) -> torch.Tensor:
    max_len = max(len(s) for s in seqs) if seqs else 0
    out = torch.full((len(seqs), max_len), pad_value, dtype=torch.long)
    for i, s in enumerate(seqs):
        if not s:
            continue
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out


@dataclass
class Batch:
    chosen_input_ids: torch.Tensor
    chosen_attention_mask: torch.Tensor
    rejected_input_ids: torch.Tensor
    rejected_attention_mask: torch.Tensor


def collate_fn(tokenizer, verification_prompt: str, max_length: int):
    pad_id = int(tokenizer.pad_token_id)

    def _collate(examples: List[Dict[str, str]]) -> Batch:
        chosen_ids: List[List[int]] = []
        chosen_attn: List[List[int]] = []
        rejected_ids: List[List[int]] = []
        rejected_attn: List[List[int]] = []

        for ex in examples:
            p = ex["prompt"]
            c = ex["chosen"]
            r = ex["rejected"]

            ids, attn = _encode_verification_format(tokenizer, p, c, verification_prompt, max_length)
            chosen_ids.append(ids)
            chosen_attn.append(attn)

            ids, attn = _encode_verification_format(tokenizer, p, r, verification_prompt, max_length)
            rejected_ids.append(ids)
            rejected_attn.append(attn)

        all_ids = chosen_ids + rejected_ids
        all_attn = chosen_attn + rejected_attn

        input_ids = _pad_2d(all_ids, pad_id)
        attention_mask = _pad_2d(all_attn, 0)

        bs = len(examples)
        return Batch(
            chosen_input_ids=input_ids[:bs],
            chosen_attention_mask=attention_mask[:bs],
            rejected_input_ids=input_ids[bs:],
            rejected_attention_mask=attention_mask[bs:],
        )

    return _collate


# --------------------------- P0: correct position gathering --------------------------- #
def _last_token_logits(logits: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    attn = attention_mask.long()
    idx = attn.sum(dim=1) - 1
    idx = idx.clamp(min=0, max=logits.size(1) - 1)
    batch = torch.arange(logits.size(0), device=logits.device)
    return logits[batch, idx, :]


def _yes_no_logits(
    logits: torch.Tensor,
    attention_mask: torch.Tensor,
    yes_token_id: int,
    no_token_id: int,
) -> torch.Tensor:
    last = _last_token_logits(logits, attention_mask)
    yn = torch.stack([last[:, yes_token_id], last[:, no_token_id]], dim=-1).float()
    return yn


def _p_yes(yes_no_logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(yes_no_logits, dim=-1)[:, 0]


# --------------------------- eval --------------------------- #
@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    yes_token_id: int,
    no_token_id: int,
    lambda_rank: float,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_ce = 0.0
    total_rank = 0.0
    total_acc = 0.0
    total_yes_prob = 0.0
    total_yes_prob_rej = 0.0
    total_batches = 0

    for batch in data_loader:
        bs = batch.chosen_input_ids.size(0)

        input_ids = torch.cat([batch.chosen_input_ids, batch.rejected_input_ids], dim=0).to(device)
        attn = torch.cat([batch.chosen_attention_mask, batch.rejected_attention_mask], dim=0).to(device)

        targets = torch.cat(
            [
                torch.zeros(bs, dtype=torch.long, device=device),  # chosen -> Yes (class 0)
                torch.ones(bs, dtype=torch.long, device=device),   # rejected -> No  (class 1)
            ],
            dim=0,
        )

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            out = model(input_ids=input_ids, attention_mask=attn, use_cache=False, return_dict=True)
            logits = out.logits

            yn_logits = _yes_no_logits(logits, attn, yes_token_id, no_token_id)
            ce_loss = F.cross_entropy(yn_logits, targets)

            score = yn_logits[:, 0] - yn_logits[:, 1]
            score_c, score_r = score[:bs], score[bs:]
            diff = score_c - score_r
            rank_loss = F.softplus(-diff).mean()

            loss = ce_loss + float(lambda_rank) * rank_loss

            probs_yes = _p_yes(yn_logits)
            p_yes_chosen = probs_yes[:bs].mean()
            p_yes_rejected = probs_yes[bs:].mean()

        total_loss += float(loss.detach().cpu())
        total_ce += float(ce_loss.detach().cpu())
        total_rank += float(rank_loss.detach().cpu())
        total_acc += float((diff > 0).float().mean().detach().cpu())
        total_yes_prob += float(p_yes_chosen.detach().cpu())
        total_yes_prob_rej += float(p_yes_rejected.detach().cpu())
        total_batches += 1

    if total_batches == 0:
        return {
            "loss": float("nan"),
            "mle": float("nan"),
            "rank": float("nan"),
            "rank_acc": float("nan"),
            "yes_prob": float("nan"),
            "yes_prob_rejected": float("nan"),
        }

    return {
        "loss": total_loss / total_batches,
        "mle": total_ce / total_batches,
        "rank": total_rank / total_batches,
        "rank_acc": total_acc / total_batches,
        "yes_prob": total_yes_prob / total_batches,
        "yes_prob_rejected": total_yes_prob_rej / total_batches,
    }


# --------------------------- cli --------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("reward_gen_train_instruct (Unsloth + QLoRA ready)")
    p.add_argument("--model_id", type=str, default=MODEL_ID)
    p.add_argument("--sft_lora_dir", type=str, default=SFT_LORA_DIR)
    p.add_argument("--pref_data_path", type=str, default=PREF_DATA_PATH)
    p.add_argument("--output_dir", type=str, default=OUTPUT_DIR)

    p.add_argument("--max_length", type=int, default=MAX_LENGTH)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--batch_size", type=int, default=PER_DEVICE_BATCH_SIZE)
    p.add_argument("--grad_accum", type=int, default=GRAD_ACCUM_STEPS)
    p.add_argument("--max_grad_norm", type=float, default=MAX_GRAD_NORM)
    p.add_argument("--warmup_ratio", type=float, default=WARMUP_RATIO)
    p.add_argument("--lambda_rank", type=float, default=LAMBDA_RANK)
    p.add_argument("--val_ratio", type=float, default=VAL_RATIO)

    p.add_argument("--lora_r", type=int, default=REWARD_LORA_RANK)
    p.add_argument("--lora_alpha", type=int, default=REWARD_LORA_ALPHA)
    p.add_argument("--lora_dropout", type=float, default=REWARD_LORA_DROPOUT)
    p.add_argument("--lora_target_modules", type=str, default=",".join(REWARD_TARGET_MODULES))

    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--max_rows", type=int, default=MAX_ROWS)

    p.add_argument("--load_in_4bit", action="store_true", default=DEFAULT_LOAD_IN_4BIT)
    p.add_argument("--no_4bit", action="store_true", help="Force disable 4bit even if --load_in_4bit set.")
    p.add_argument("--grad_ckpt", action="store_true", default=DEFAULT_USE_GRAD_CHECKPOINTING)

    p.add_argument("--log_every", type=int, default=LOG_EVERY_OPT_STEPS)
    p.add_argument("--eval_every", type=int, default=EVAL_EVERY_OPT_STEPS)

    return p


def _save_verifier_config(
    save_dir: str,
    *,
    verification_prompt: str,
    yes_token_id: int,
    no_token_id: int,
    max_length: int,
    load_in_4bit: bool,
    lora_cfg: Dict[str, object],
) -> None:
    cfg = {
        "verification_prompt": verification_prompt,
        "yes_token_id": int(yes_token_id),
        "no_token_id": int(no_token_id),
        "max_length": int(max_length),
        "load_in_4bit": bool(load_in_4bit),
        "lora": lora_cfg,
    }
    with open(os.path.join(save_dir, "verifier_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def _safe_save_adapter(model, save_dir: str, tokenizer, *, adapter_name: Optional[str] = None) -> None:
    os.makedirs(save_dir, exist_ok=True)
    # Try saving only selected adapter if supported
    if adapter_name is not None:
        try:
            model.save_pretrained(save_dir, selected_adapters=[adapter_name])
        except TypeError:
            model.save_pretrained(save_dir)
        except Exception:
            model.save_pretrained(save_dir)
    else:
        model.save_pretrained(save_dir)

    try:
        tokenizer.save_pretrained(save_dir)
    except Exception:
        pass


# --------------------------- main --------------------------- #
def main() -> None:
    args = build_arg_parser().parse_args()

    # normalize
    load_in_4bit = bool(args.load_in_4bit) and (not bool(args.no_4bit))
    lora_target_modules = [x.strip() for x in str(args.lora_target_modules).split(",") if x.strip()]
    _require_arg(args.model_id, "model_id", "MODEL_ID")
    _require_arg(args.pref_data_path, "pref_data_path", "PREF_DATA_PATH")
    _require_arg(args.output_dir, "output_dir", "REWARD_GEN_OUTPUT_DIR")

    # resolve paths
    model_id = _abspath(args.model_id)
    pref_data_path = _abspath(args.pref_data_path)
    output_dir = _abspath(args.output_dir)

    use_sft_adapter = bool(args.sft_lora_dir and str(args.sft_lora_dir).strip())
    sft_lora_dir = _abspath(args.sft_lora_dir) if use_sft_adapter else ""

    # logging
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging(os.path.join(output_dir, LOG_FILE))

    seed_everything(int(args.seed))
    torch.backends.cuda.matmul.allow_tf32 = True

    logger.info("==== reward_gen (generative verifier) training | Unsloth=%s | 4bit=%s ====",
                str(FastLanguageModel is not None), str(load_in_4bit))
    logger.info("MODEL_ID: %s", model_id)
    logger.info("SFT_LORA_DIR: %s", sft_lora_dir if use_sft_adapter else "(not used)")
    logger.info("PREF_DATA_PATH: %s", pref_data_path)
    logger.info("OUTPUT_DIR: %s", output_dir)
    logger.info("Hyperparams: max_len=%d epochs=%d lr=%g wd=%g bs=%d accum=%d warmup=%.3f lambda_rank=%.3f",
                int(args.max_length), int(args.epochs), float(args.lr), float(args.weight_decay),
                int(args.batch_size), int(args.grad_accum), float(args.warmup_ratio), float(args.lambda_rank))
    logger.info("Reward LoRA: r=%d alpha=%d dropout=%.3f target_modules=%s",
                int(args.lora_r), int(args.lora_alpha), float(args.lora_dropout), ",".join(lora_target_modules))

    if not os.path.exists(model_id):
        raise FileNotFoundError(f"Base model path not found: {model_id}")
    if use_sft_adapter and not os.path.exists(sft_lora_dir):
        raise FileNotFoundError(f"SFT LoRA dir not found: {sft_lora_dir}")
    if not os.path.exists(pref_data_path):
        raise FileNotFoundError(f"Preference dataset not found: {pref_data_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training.")

    logger.info("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    _ensure_pad_token(tokenizer)

    # YES/NO token ids shared with the RALA scorer.
    yes_token_id, no_token_id = _get_yes_no_token_ids(tokenizer)
    logger.info("Verification tokens: yes_id=%d ('%s') | no_id=%d ('%s')",
                yes_token_id, tokenizer.decode([yes_token_id]),
                no_token_id, tokenizer.decode([no_token_id]))

    logger.info("Loading preference dataset ...")
    dataset = load_dataset("json", data_files=pref_data_path, split="train")
    if int(args.max_rows) and int(args.max_rows) > 0:
        dataset = dataset.select(range(min(int(args.max_rows), len(dataset))))
    dataset = dataset.shuffle(seed=int(args.seed))

    if len(dataset) <= 1:
        raise RuntimeError("Preference dataset too small (need at least 2 rows).")

    test_size = max(1, int(round(len(dataset) * float(args.val_ratio))))
    if test_size >= len(dataset):
        test_size = 1
    split = dataset.train_test_split(test_size=test_size, seed=int(args.seed))
    train_ds = split["train"]
    eval_ds = split["test"]

    logger.info("Dataset size: train=%d eval=%d (val_ratio=%.3f)", len(train_ds), len(eval_ds), float(args.val_ratio))
    preview = train_ds[0]
    logger.info("Sample prompt: %s", _short(preview["prompt"], 200))
    logger.info("Sample chosen: %s", _short(preview["chosen"], 200))
    logger.info("Sample rejected: %s", _short(preview["rejected"], 200))

    logger.info("Loading base model (Unsloth preferred) ...")
    base = load_base_model_qlora(
        model_name_or_path=model_id,
        max_seq_length=int(args.max_length),
        load_in_4bit=load_in_4bit,
        logger=logger,
    )
    base.config.use_cache = False

    # If k-bit, prepare model for training kernels/hooks
    if load_in_4bit:
        try:
            base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=bool(args.grad_ckpt))
        except TypeError:
            try:
                base = prepare_model_for_kbit_training(base)
            except Exception:
                pass

    # Optional frozen SFT adapter.
    if use_sft_adapter:
        logger.info("Loading SFT LoRA adapter (frozen) from: %s", sft_lora_dir)
        try:
            base = PeftModel.from_pretrained(base, sft_lora_dir, is_trainable=False, adapter_name="sft")
        except TypeError:
            base = PeftModel.from_pretrained(base, sft_lora_dir)
        for p in base.parameters():
            p.requires_grad_(False)
        base.eval()
    else:
        for p in base.parameters():
            p.requires_grad_(False)

    logger.info("Attaching reward_gen LoRA adapter (trainable) ...")
    reward_lora_cfg = LoraConfig(
        r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_target_modules,
    )

    # If base already is a PeftModel (because SFT loaded), add a new adapter.
    # Else, create a new PeftModel via get_peft_model.
    adapter_name = "reward_gen"
    if isinstance(base, PeftModel) and hasattr(base, "add_adapter"):
        base.add_adapter(adapter_name, reward_lora_cfg)
        try:
            base.set_adapter(adapter_name)
        except Exception:
            pass
        model = base
    else:
        model = get_peft_model(base, reward_lora_cfg)
        try:
            if hasattr(model, "rename_adapter"):
                model.rename_adapter("default", adapter_name)
                model.set_adapter(adapter_name)
        except Exception:
            adapter_name = "default"  # Keep as default, not None

    # Enable training for LoRA parameters
    # LoRA parameters typically contain 'lora_A', 'lora_B', 'lora_', etc. in their names
    lora_keywords = ["lora_", "lora_A", "lora_B", ".lora"]

    def is_lora_param(name: str) -> bool:
        name_lower = name.lower()
        # Check for LoRA-specific keywords
        for kw in lora_keywords:
            if kw.lower() in name_lower:
                return True
        # Also check for adapter name if known
        if adapter_name and adapter_name.lower() in name_lower:
            return True
        return False

    # First freeze all, then selectively unfreeze LoRA
    for p in model.parameters():
        p.requires_grad_(False)

    trainable_count = 0
    for n, p in model.named_parameters():
        if is_lora_param(n):
            p.requires_grad_(True)
            trainable_count += 1

    if trainable_count == 0:
        logger.warning("[LoRA] No LoRA params detected by name. Trying to unfreeze all non-quantized params.")
        for n, p in model.named_parameters():
            # Skip quantized/frozen base weights (usually has 'weight' and no 'lora')
            # But unfreeze if it's a bias or if the layer was added by peft
            if not p.is_cuda or p.numel() == 0:
                continue
            # Try unfreezing everything except large embedding/output layers
            if "embed" not in n.lower() and "lm_head" not in n.lower():
                p.requires_grad_(True)
                trainable_count += 1

    model.train()
    unsloth_for_training(model, use_gradient_checkpointing=bool(args.grad_ckpt))

    try:
        device = model.get_input_embeddings().weight.device
    except Exception:
        device = next(model.parameters()).device
    logger.info("Model device: %s", str(device))

    trainable = [p for p in model.parameters() if p.requires_grad]
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    logger.info("Trainable params: %d", sum(p.numel() for p in trainable))
    logger.info("Trainable tensors count: %d", len(trainable_names))
    if trainable_names:
        logger.info("Trainable tensors (first 10): %s", trainable_names[:10])
    if not trainable:
        # Print parameter names when no trainable parameters are detected.
        all_names = [n for n, p in model.named_parameters()]
        logger.error("All parameter names (first 20): %s", all_names[:20])
        raise RuntimeError("No trainable parameters found (LoRA attach failed?).")

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn(tokenizer, VERIFICATION_PROMPT, int(args.max_length)),
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn(tokenizer, VERIFICATION_PROMPT, int(args.max_length)),
    )

    num_update_steps_per_epoch = math.ceil(len(train_loader) / max(1, int(args.grad_accum)))
    max_train_steps = int(args.epochs) * num_update_steps_per_epoch
    warmup_steps = int(round(max_train_steps * float(args.warmup_ratio)))

    optimizer = torch.optim.AdamW(trainable, lr=float(args.lr), weight_decay=float(args.weight_decay))
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps,
    )

    use_amp = True
    amp_dtype = choose_compute_dtype()
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))
    logger.info("AMP enabled=%s | amp_dtype=%s | scaler=%s", str(use_amp), str(amp_dtype), str(scaler.is_enabled()))
    logger.info("Steps: train_batches/epoch=%d update_steps/epoch=%d total_update_steps=%d warmup_steps=%d",
                len(train_loader), num_update_steps_per_epoch, max_train_steps, warmup_steps)

    # Save tokenizer once
    try:
        tokenizer.save_pretrained(output_dir)
    except Exception:
        pass

    # Save verifier config
    _save_verifier_config(
        output_dir,
        verification_prompt=VERIFICATION_PROMPT,
        yes_token_id=yes_token_id,
        no_token_id=no_token_id,
        max_length=int(args.max_length),
        load_in_4bit=load_in_4bit,
        lora_cfg={
            "r": int(args.lora_r),
            "alpha": int(args.lora_alpha),
            "dropout": float(args.lora_dropout),
            "target_modules": lora_target_modules,
        },
    )

    # best checkpoint tracking (rank_acc)
    best_metric = -float("inf")
    best_checkpoint_dir = os.path.join(output_dir, "best_checkpoint")
    best_step = 0

    # quick sanity forward
    logger.info("Running forward sanity check...")
    model.eval()
    first_batch = next(iter(train_loader))
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
        test_ids = first_batch.chosen_input_ids[:1].to(device)
        test_attn = first_batch.chosen_attention_mask[:1].to(device)
        test_out = model(input_ids=test_ids, attention_mask=test_attn, use_cache=False, return_dict=True)
        test_yn = _yes_no_logits(test_out.logits, test_attn, yes_token_id, no_token_id)
        if torch.isnan(test_yn).any() or torch.isinf(test_yn).any():
            logger.error("NaN/Inf detected in sanity check.")
        else:
            logger.info("Sanity check passed. yn_logits=%s", str(test_yn.detach().cpu()))
    model.train()

    logger.info("Training start ...")
    global_opt_step = 0
    running = {"loss": 0.0, "mle": 0.0, "rank": 0.0, "acc": 0.0, "yes_prob": 0.0, "yes_prob_rejected": 0.0, "n": 0}
    t0 = time.time()

    for epoch in range(int(args.epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        total_train_batches = len(train_loader)
        remainder = total_train_batches % int(args.grad_accum)

        pbar = tqdm(
            total=num_update_steps_per_epoch,
            desc=f"Epoch {epoch + 1}/{int(args.epochs)}",
            unit="opt_step",
            leave=True,
        )

        for step, batch in enumerate(train_loader):
            is_last_batch = step == total_train_batches - 1
            if remainder != 0 and step >= total_train_batches - remainder:
                accum_divisor = float(remainder)
            else:
                accum_divisor = float(args.grad_accum)

            bs = batch.chosen_input_ids.size(0)
            input_ids = torch.cat([batch.chosen_input_ids, batch.rejected_input_ids], dim=0).to(device)
            attn = torch.cat([batch.chosen_attention_mask, batch.rejected_attention_mask], dim=0).to(device)

            targets = torch.cat(
                [
                    torch.zeros(bs, dtype=torch.long, device=device),  # chosen -> Yes
                    torch.ones(bs, dtype=torch.long, device=device),   # rejected -> No
                ],
                dim=0,
            )

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                out = model(input_ids=input_ids, attention_mask=attn, use_cache=False, return_dict=True)
                yn_logits = _yes_no_logits(out.logits, attn, yes_token_id, no_token_id)

                ce_loss = F.cross_entropy(yn_logits, targets)

                score = yn_logits[:, 0] - yn_logits[:, 1]
                score_c, score_r = score[:bs], score[bs:]
                diff = score_c - score_r
                rank_loss = F.softplus(-diff).mean()

                loss = ce_loss + float(args.lambda_rank) * rank_loss

                probs_yes = _p_yes(yn_logits)
                p_yes_chosen = probs_yes[:bs].mean()
                p_yes_rejected = probs_yes[bs:].mean()

            loss_to_backprop = loss / accum_divisor
            scaler.scale(loss_to_backprop).backward()

            do_opt_step = ((step + 1) % int(args.grad_accum) == 0) or is_last_batch
            if do_opt_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, float(args.max_grad_norm))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()

                global_opt_step += 1
                pbar.update(1)

                rank_acc = float((diff > 0).float().mean().detach().cpu())
                running["loss"] += float(loss.detach().cpu())
                running["mle"] += float(ce_loss.detach().cpu())
                running["rank"] += float(rank_loss.detach().cpu())
                running["acc"] += rank_acc
                running["yes_prob"] += float(p_yes_chosen.detach().cpu())
                running["yes_prob_rejected"] += float(p_yes_rejected.detach().cpu())
                running["n"] += 1

                n_for_bar = max(1, running["n"])
                dt = time.time() - t0
                pbar.set_postfix(
                    {
                        "opt": f"{global_opt_step}",
                        "lr": f"{float(lr_scheduler.get_last_lr()[0]):.2e}",
                        "loss": f"{running['loss'] / n_for_bar:.4f}",
                        "ce": f"{running['mle'] / n_for_bar:.4f}",
                        "rank": f"{running['rank'] / n_for_bar:.4f}",
                        "acc": f"{running['acc'] / n_for_bar:.3f}",
                        "pY_c": f"{running['yes_prob'] / n_for_bar:.3f}",
                        "pY_r": f"{running['yes_prob_rejected'] / n_for_bar:.3f}",
                        "t": f"{dt:.0f}s",
                    }
                )

                if global_opt_step % int(args.log_every) == 0:
                    n = max(1, running["n"])
                    logger.info(
                        "epoch=%d opt_step=%d lr=%.3e loss=%.4f (ce=%.4f rank=%.4f) "
                        "rank_acc=%.3f p(Yes)_c=%.3f p(Yes)_r=%.3f time=%.1fs",
                        epoch + 1,
                        global_opt_step,
                        float(lr_scheduler.get_last_lr()[0]),
                        running["loss"] / n,
                        running["mle"] / n,
                        running["rank"] / n,
                        running["acc"] / n,
                        running["yes_prob"] / n,
                        running["yes_prob_rejected"] / n,
                        dt,
                    )
                    running = {"loss": 0.0, "mle": 0.0, "rank": 0.0, "acc": 0.0, "yes_prob": 0.0, "yes_prob_rejected": 0.0, "n": 0}

                if global_opt_step % int(args.eval_every) == 0:
                    metrics = evaluate(
                        model,
                        eval_loader,
                        device=device,
                        use_amp=use_amp,
                        amp_dtype=amp_dtype,
                        yes_token_id=yes_token_id,
                        no_token_id=no_token_id,
                        lambda_rank=float(args.lambda_rank),
                    )

                    current_metric = float(metrics["rank_acc"])
                    is_best = current_metric > best_metric

                    if is_best:
                        best_metric = current_metric
                        best_step = global_opt_step
                        _safe_save_adapter(model, best_checkpoint_dir, tokenizer, adapter_name=adapter_name)

                        best_info = {
                            "step": best_step,
                            "rank_acc": metrics["rank_acc"],
                            "yes_prob": metrics["yes_prob"],
                            "yes_prob_rejected": metrics["yes_prob_rejected"],
                            "loss": metrics["loss"],
                            "ce": metrics["mle"],
                            "rank": metrics["rank"],
                        }
                        with open(os.path.join(best_checkpoint_dir, "best_metrics.json"), "w", encoding="utf-8") as f:
                            json.dump(best_info, f, indent=2)

                        _save_verifier_config(
                            best_checkpoint_dir,
                            verification_prompt=VERIFICATION_PROMPT,
                            yes_token_id=yes_token_id,
                            no_token_id=no_token_id,
                            max_length=int(args.max_length),
                            load_in_4bit=load_in_4bit,
                            lora_cfg={
                                "r": int(args.lora_r),
                                "alpha": int(args.lora_alpha),
                                "dropout": float(args.lora_dropout),
                                "target_modules": lora_target_modules,
                            },
                        )

                        logger.info("[NEW BEST] opt_step=%d rank_acc=%.4f -> saved to %s",
                                    global_opt_step, metrics["rank_acc"], best_checkpoint_dir)

                    logger.info("[eval] opt_step=%d loss=%.4f ce=%.4f rank=%.4f rank_acc=%.3f pY_c=%.3f pY_r=%.3f%s",
                                global_opt_step, metrics["loss"], metrics["mle"], metrics["rank"],
                                metrics["rank_acc"], metrics["yes_prob"], metrics["yes_prob_rejected"],
                                " *" if is_best else "")
                    model.train()

        pbar.close()

        # epoch end save (latest)
        _safe_save_adapter(model, output_dir, tokenizer, adapter_name=adapter_name)
        logger.info("Saved latest verifier adapter to %s (epoch %d done)", output_dir, epoch + 1)

    logger.info("=" * 60)
    logger.info("Training complete.")
    logger.info("Final adapter: %s", output_dir)
    logger.info("Best checkpoint: %s (step=%d, rank_acc=%.4f)", best_checkpoint_dir, best_step, best_metric)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
