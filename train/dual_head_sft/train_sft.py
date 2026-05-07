#!/usr/bin/env python3
"""Train dual-head SFT with an auxiliary reward head.

LoRA SFT updates the policy while a lightweight discriminative reward head is
trained in-place from self-generated negatives. The discriminative loss consumes
detached hidden states, so it updates only the reward head and does not
back-propagate into the shared SFT backbone.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def set_offline_env() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


set_offline_env()

try:
    from unsloth import FastLanguageModel  # type: ignore
except Exception:
    FastLanguageModel = None  # type: ignore

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForTokenClassification,
)
from trl import SFTConfig, SFTTrainer


def read_rows(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        data = json.loads(text)
        if isinstance(data, dict):
            data = data.get("data", [])
        return [dict(x) for x in data]
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def ensure_pad_token(tokenizer: Any) -> None:
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id


def _first_nonempty(record: Dict[str, Any], names: Tuple[str, ...]) -> str:
    for name in names:
        value = record.get(name)
        if value is not None and str(value) != "":
            return str(value)
    return ""


def _plain_messages(messages: List[Dict[str, Any]]) -> str:
    return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)


def _split_messages(messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str]:
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") == "assistant":
            return messages[:idx], str(messages[idx].get("content", ""))
    raise ValueError("Dual-head SFT requires conversational rows to contain an assistant response.")


def _completion_from_full_text(full_text: str, prompt_text: str, fallback: str) -> str:
    if full_text.startswith(prompt_text):
        completion = full_text[len(prompt_text) :]
        if completion:
            return completion
    return fallback


def record_to_text(record: Dict[str, Any], tokenizer: Any, use_chat_template: bool) -> str:
    if isinstance(record.get("text"), str):
        return record["text"]
    if isinstance(record.get("messages"), list):
        messages = record["messages"]
        if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return _plain_messages(messages)
    prompt = _first_nonempty(record, ("prompt", "instruction", "query", "question"))
    completion = _first_nonempty(record, ("completion", "response", "output", "answer"))
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}],
            tokenize=False,
            add_generation_prompt=False,
        )
    return prompt.rstrip() + "\n" + completion.lstrip()


def record_to_prompt_completion(record: Dict[str, Any], tokenizer: Any, use_chat_template: bool) -> Tuple[str, str]:
    if isinstance(record.get("messages"), list):
        messages = record["messages"]
        prompt_messages, assistant_response = _split_messages(messages)
        if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            return prompt_text, _completion_from_full_text(full_text, prompt_text, assistant_response)
        return _plain_messages(prompt_messages).rstrip() + "\nassistant: ", assistant_response

    prompt = _first_nonempty(record, ("prompt", "instruction", "query", "question"))
    completion = _first_nonempty(record, ("completion", "response", "output", "answer"))
    if prompt or completion:
        if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}],
                tokenize=False,
                add_generation_prompt=False,
            )
            return prompt_text, _completion_from_full_text(full_text, prompt_text, completion)
        return prompt.rstrip() + "\n", completion.lstrip()

    if isinstance(record.get("text"), str):
        raise ValueError(
            "Dual-head SFT requires separable prompt/completion or messages rows; "
            "plain text-only rows cannot be used for online negative mining."
        )
    raise ValueError("Unsupported SFT row format.")


def load_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.train_file:
        rows = read_rows(args.train_file)
    else:
        rows = [dict(x) for x in load_dataset(args.dataset_name, args.dataset_config, split=args.split)]
    return rows[: args.max_rows] if args.max_rows > 0 else rows


def split_rows(
    rows: List[Dict[str, Any]], val_ratio: float, seed: int
) -> tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
    if val_ratio <= 0 or len(rows) < 2:
        return rows, None
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    n_val = max(1, min(len(rows) - 1, int(round(len(rows) * val_ratio))))
    return rows[n_val:], rows[:n_val]


def tokenize_prompt_completion(
    prompt_text: str,
    completion_text: str,
    tokenizer: Any,
    max_length: int,
    require_prompt: bool = True,
) -> Dict[str, Any]:
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    completion_ids = tokenizer(completion_text, add_special_tokens=False).input_ids
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if len(completion_ids) == 0 and eos_id is not None:
        completion_ids = [eos_id]
    if require_prompt and len(prompt_ids) == 0:
        raise ValueError("Dual-head SFT requires non-empty prompts for negative generation.")

    if max_length and len(prompt_ids) + len(completion_ids) > max_length:
        completion_limit = max_length - 1 if require_prompt and len(prompt_ids) > 0 else max_length
        completion_budget = max(1, min(len(completion_ids), completion_limit))
        prompt_budget = max_length - completion_budget
        prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget > 0 else []
        completion_ids = completion_ids[:completion_budget]

    input_ids = prompt_ids + completion_ids
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(prompt_ids) + completion_ids
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def build_dual_head_dataset(
    rows: List[Dict[str, Any]],
    tokenizer: Any,
    args: argparse.Namespace,
    *,
    train: bool,
) -> tuple[Dataset, int]:
    disc_count = 0
    disc_indices: set[int] = set()
    if train:
        if args.disc_max_prompts <= 0 or args.disc_max_prompts >= len(rows):
            disc_indices = set(range(len(rows)))
        else:
            disc_indices = set(random.Random(args.seed).sample(range(len(rows)), args.disc_max_prompts))
        disc_count = len(disc_indices)

    examples: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        prompt_text, completion_text = record_to_prompt_completion(row, tokenizer, args.use_chat_template)
        item = tokenize_prompt_completion(
            prompt_text,
            completion_text,
            tokenizer,
            max_length=args.max_length,
            require_prompt=True,
        )
        item["disc_mask"] = bool(train and idx in disc_indices)
        examples.append(item)
    return Dataset.from_list(examples), disc_count


def make_sft_config(args: argparse.Namespace, *, prepared_dataset: bool) -> SFTConfig:
    kwargs: Dict[str, Any] = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler_type,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "eval_strategy": "steps" if args.val_ratio > 0 else "no",
        "save_strategy": "steps",
        "bf16": args.bf16,
        "fp16": args.fp16,
        "max_grad_norm": args.max_grad_norm,
        "optim": args.optim,
        "report_to": [],
        "seed": args.seed,
        "max_length": args.max_length,
        "packing": args.packing,
        "dataset_text_field": "text",
        "completion_only_loss": None,
    }
    if prepared_dataset:
        kwargs.update(
            {
                "completion_only_loss": True,
                "dataset_kwargs": {"skip_prepare_dataset": True},
                "remove_unused_columns": False,
            }
        )
    if is_dataclass(SFTConfig):
        valid = {f.name for f in fields(SFTConfig)}
        kwargs = {k: v for k, v in kwargs.items() if k in valid}
    return SFTConfig(**kwargs)


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any]:
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)
    target_modules = args.lora_target_modules.split(",")
    if FastLanguageModel is not None and not args.no_unsloth:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_id,
            max_seq_length=args.max_length,
            dtype=dtype,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            trust_remote_code=args.trust_remote_code,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            target_modules=target_modules,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            use_gradient_checkpointing=args.use_gradient_checkpointing,
            random_state=args.seed,
        )
        return model, tokenizer

    quant_config = None
    if args.load_in_4bit or args.load_in_8bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            bnb_4bit_compute_dtype=dtype or torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    if args.load_in_4bit or args.load_in_8bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=bool(args.use_gradient_checkpointing))
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        ),
    )
    return model, tokenizer


class RewardHead(nn.Module):
    """Linear reward head over the final hidden state of the response."""

    def __init__(self, hidden_size: int, head_hidden_size: int, dropout: float) -> None:
        super().__init__()
        del head_hidden_size, dropout
        self.proj = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        lengths = attention_mask.long().sum(dim=1) - 1
        lengths = torch.clamp(lengths, min=0)
        batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        pooled = hidden_states[batch_idx, lengths]
        pooled = pooled.to(device=self.proj.weight.device, dtype=self.proj.weight.dtype)
        return self.proj(pooled).squeeze(-1)


def _model_hidden_size(model: nn.Module) -> int:
    config = getattr(model, "config", None)
    if config is not None and getattr(config, "hidden_size", None) is not None:
        return int(config.hidden_size)
    if hasattr(model, "get_base_model"):
        base = model.get_base_model()
        config = getattr(base, "config", None)
        if config is not None and getattr(config, "hidden_size", None) is not None:
            return int(config.hidden_size)
    raise ValueError("Cannot infer hidden_size for reward head.")


def attach_reward_head(model: nn.Module, args: argparse.Namespace) -> None:
    if hasattr(model, "reward_head"):
        return
    reward_head = RewardHead(
        hidden_size=_model_hidden_size(model),
        head_hidden_size=args.reward_head_hidden_size,
        dropout=args.reward_head_dropout,
    )
    try:
        first_param = next(model.parameters())
        reward_head.to(device=first_param.device, dtype=torch.float32)
    except StopIteration:
        reward_head.to(dtype=torch.float32)
    model.add_module("reward_head", reward_head)


@dataclass
class DualHeadDataCollator:
    tokenizer: Any
    label_pad_token_id: int = -100

    def __post_init__(self) -> None:
        self.base = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            label_pad_token_id=self.label_pad_token_id,
        )

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        copied = [dict(feature) for feature in features]
        disc_mask = torch.tensor([bool(feature.pop("disc_mask", False)) for feature in copied], dtype=torch.bool)
        batch = self.base(copied)
        batch["disc_mask"] = disc_mask
        return batch


class DualHeadSFTTrainer(SFTTrainer):
    """SFTTrainer with an in-place discriminative reward-head update."""

    def __init__(
        self,
        *args: Any,
        negatives_per_prompt: int,
        negative_top_p: float,
        negative_temperature: float,
        negative_max_new_tokens: int,
        disc_loss_every_n_steps: int,
        disc_loss_weight: float,
        margin_m0: float,
        margin_gamma: float,
        margin_t0: float,
        **kwargs: Any,
    ) -> None:
        if negatives_per_prompt < 1:
            raise ValueError("negatives_per_prompt must be >= 1.")
        if disc_loss_every_n_steps < 1:
            raise ValueError("disc_loss_every_n_steps must be >= 1.")
        self.negatives_per_prompt = negatives_per_prompt
        self.negative_top_p = negative_top_p
        self.negative_temperature = negative_temperature
        self.negative_max_new_tokens = negative_max_new_tokens
        self.disc_loss_every_n_steps = disc_loss_every_n_steps
        self.disc_loss_weight = disc_loss_weight
        self.margin_m0 = margin_m0
        self.margin_gamma = margin_gamma
        self.margin_t0 = margin_t0
        super().__init__(*args, **kwargs)

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ) -> Any:
        clean_inputs = {k: v for k, v in inputs.items() if k != "disc_mask"}
        return super().compute_loss(
            model,
            clean_inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False) -> None:
        output_dir = output_dir or self.args.output_dir
        try:
            super().save_model(output_dir, _internal_call=_internal_call)
        except TypeError:
            super().save_model(output_dir)
        if self.is_world_process_zero():
            unwrapped = self.accelerator.unwrap_model(self.model)
            if hasattr(unwrapped, "reward_head"):
                os.makedirs(output_dir, exist_ok=True)
                torch.save(unwrapped.reward_head.state_dict(), os.path.join(output_dir, "reward_head.pt"))

    def _tokenizer(self) -> Any:
        tokenizer = getattr(self, "processing_class", None)
        if hasattr(tokenizer, "tokenizer"):
            tokenizer = tokenizer.tokenizer
        if tokenizer is None:
            raise ValueError("Dual-head SFT requires processing_class=tokenizer.")
        return tokenizer

    def _pad_eos_ids(self) -> tuple[int, Optional[int]]:
        tokenizer = self._tokenizer()
        pad_id = getattr(tokenizer, "pad_token_id", None)
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if pad_id is None:
            pad_id = eos_id
        if pad_id is None:
            raise ValueError("Tokenizer must define pad_token_id or eos_token_id.")
        return int(pad_id), int(eos_id) if eos_id is not None else None

    def _build_prompt_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_mask = (labels == -100) & (attention_mask == 1)
        prompt_lengths = prompt_mask.long().sum(dim=1)
        if torch.any(prompt_lengths <= 0):
            raise ValueError("Dual-head SFT encountered an example with an empty prompt.")
        max_prompt = int(prompt_lengths.max().item())
        pad_id, _ = self._pad_eos_ids()
        batch_size = input_ids.size(0)
        prompt_ids = input_ids.new_full((batch_size, max_prompt), pad_id)
        prompt_attention = input_ids.new_zeros((batch_size, max_prompt))

        for i in range(batch_size):
            plen = int(prompt_lengths[i].item())
            start = max_prompt - plen
            prompt_ids[i, start:] = input_ids[i, :plen]
            prompt_attention[i, start:] = 1
        return prompt_ids, prompt_attention

    @staticmethod
    def _generated_attention_mask(
        prompt_attention: torch.Tensor,
        generated_ids: torch.Tensor,
        eos_id: Optional[int],
    ) -> torch.Tensor:
        prompt_len = prompt_attention.size(1)
        returns_per_prompt = generated_ids.size(0) // prompt_attention.size(0)
        prefix_attention = prompt_attention.repeat_interleave(returns_per_prompt, dim=0).to(generated_ids.device)
        gen_len = generated_ids.size(1) - prompt_len
        if gen_len <= 0:
            return prefix_attention
        gen_attention = torch.ones((generated_ids.size(0), gen_len), dtype=torch.long, device=generated_ids.device)
        if eos_id is not None:
            generated_tail = generated_ids[:, prompt_len:]
            eos_hits = generated_tail.eq(eos_id)
            for i in range(eos_hits.size(0)):
                pos = torch.nonzero(eos_hits[i], as_tuple=False)
                if pos.numel() > 0:
                    first_eos = int(pos[0].item())
                    if first_eos + 1 < gen_len:
                        gen_attention[i, first_eos + 1 :] = 0
        return torch.cat([prefix_attention, gen_attention], dim=1)

    def _score_with_detached_hidden(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        unwrapped = self.accelerator.unwrap_model(model)
        with torch.no_grad():
            outputs = unwrapped(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            hidden_states = outputs.hidden_states[-1].detach()
        return unwrapped.reward_head(hidden_states, attention_mask)

    def _margin(self) -> float:
        step = float(self.state.global_step)
        return self.margin_m0 + self.margin_gamma * math.log1p(step / self.margin_t0)

    def _compute_disc_loss(self, model: nn.Module, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]
        disc_mask = inputs.get("disc_mask")
        if disc_mask is not None:
            keep = disc_mask.to(device=input_ids.device, dtype=torch.bool)
            if int(keep.sum().item()) == 0:
                return input_ids.new_zeros((), dtype=torch.float32)
            input_ids = input_ids[keep]
            attention_mask = attention_mask[keep]
            labels = labels[keep]

        prompt_ids, prompt_attention = self._build_prompt_batch(input_ids, attention_mask, labels)
        pad_id, eos_id = self._pad_eos_ids()
        unwrapped = self.accelerator.unwrap_model(model)

        was_training = model.training
        model.eval()
        with torch.no_grad():
            negative_ids = unwrapped.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_attention,
                do_sample=True,
                top_p=self.negative_top_p,
                temperature=self.negative_temperature,
                max_new_tokens=self.negative_max_new_tokens,
                num_return_sequences=self.negatives_per_prompt,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
                use_cache=True,
            )
        if was_training:
            model.train()

        negative_attention = self._generated_attention_mask(prompt_attention, negative_ids, eos_id)
        batch_size = input_ids.size(0)
        with torch.no_grad():
            all_negative_scores = self._score_with_detached_hidden(model, negative_ids, negative_attention)
            all_negative_scores = all_negative_scores.view(batch_size, self.negatives_per_prompt)
            hardest_idx = torch.argmin(all_negative_scores, dim=1)

        negative_ids = negative_ids.view(batch_size, self.negatives_per_prompt, -1)
        negative_attention = negative_attention.view(batch_size, self.negatives_per_prompt, -1)
        batch_idx = torch.arange(batch_size, device=input_ids.device)
        hard_negative_ids = negative_ids[batch_idx, hardest_idx]
        hard_negative_attention = negative_attention[batch_idx, hardest_idx]

        reward_pos = self._score_with_detached_hidden(model, input_ids, attention_mask)
        reward_neg = self._score_with_detached_hidden(model, hard_negative_ids, hard_negative_attention)
        margin = reward_pos.new_tensor(self._margin())
        return -F.logsigmoid(reward_pos - reward_neg - margin).mean()

    def training_step(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        num_items_in_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            sft_loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        unwrapped = self.accelerator.unwrap_model(model)
        if hasattr(unwrapped, "reward_head"):
            dummy = None
            for param in unwrapped.reward_head.parameters():
                dummy = param.sum() if dummy is None else dummy + param.sum()
            if dummy is not None:
                sft_loss = sft_loss + dummy * 0.0

        if self.args.n_gpu > 1:
            sft_loss = sft_loss.mean()
        if self.args.gradient_accumulation_steps > 1:
            sft_loss = sft_loss / self.args.gradient_accumulation_steps
        self.accelerator.backward(sft_loss)

        do_disc = (self.state.global_step % self.disc_loss_every_n_steps) == 0
        if do_disc:
            with self.compute_loss_context_manager():
                disc_loss = self._compute_disc_loss(model, inputs) * self.disc_loss_weight
        else:
            disc_loss = sft_loss.new_zeros(())

        if self.args.n_gpu > 1:
            disc_loss = disc_loss.mean()
        if self.args.gradient_accumulation_steps > 1:
            disc_loss = disc_loss / self.args.gradient_accumulation_steps
        if do_disc and disc_loss.requires_grad:
            self.accelerator.backward(disc_loss)

        if self.state.global_step % self.args.logging_steps == 0:
            scale = float(self.args.gradient_accumulation_steps)
            self.log(
                {
                    "sft_loss": float(sft_loss.detach().item() * scale),
                    "disc_loss": float(disc_loss.detach().item() * scale),
                    "disc_margin": float(self._margin()),
                    "disc_do_step": float(do_disc),
                }
            )
        return (sft_loss + disc_loss).detach()


def write_dual_head_config(args: argparse.Namespace, disc_train_prompts: int) -> None:
    path = Path(args.output_dir) / "dual_head_config.json"
    payload = {
        "enabled": True,
        "reward_head_architecture": "linear",
        "reward_head_hidden_size": None,
        "reward_head_activation": None,
        "reward_head_dropout": 0.0,
        "reward_head_input": "final hidden state of the response sequence",
        "gradient_isolation": "discriminative loss uses detached hidden states and updates reward_head only",
        "disc_train_prompts": disc_train_prompts,
        "negatives_per_prompt": args.negatives_per_prompt,
        "negative_top_p": args.negative_top_p,
        "negative_temperature": args.negative_temperature,
        "negative_max_new_tokens": args.negative_max_new_tokens,
        "disc_loss_every_n_steps": args.disc_loss_every_n_steps,
        "disc_loss_weight": args.disc_loss_weight,
        "margin_m0": args.margin_m0,
        "margin_gamma": args.margin_gamma,
        "margin_t0": args.margin_t0,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Train a dual-head SFT LoRA adapter.")
    p.add_argument("--model_id", required=True)
    p.add_argument("--train_file", default="")
    p.add_argument("--dataset_name", default="")
    p.add_argument("--dataset_config", default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_rows", type=int, default=0)
    p.add_argument("--val_ratio", type=float, default=0.02)
    p.add_argument("--use_chat_template", action="store_true")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--no_unsloth", action="store_true")
    p.add_argument("--load_in_4bit", action="store_true", default=True)
    p.add_argument("--load_in_8bit", action="store_true")
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--num_train_epochs", type=float, default=3.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--per_device_eval_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--lr_scheduler_type", default="cosine")
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--optim", default="paged_adamw_8bit")
    p.add_argument("--bf16", action="store_true", default=torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--packing", action="store_true")
    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    p.add_argument("--use_gradient_checkpointing", default="unsloth")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--disable_dual_head", action="store_true")
    p.add_argument("--reward_head_hidden_size", type=int, default=1024)
    p.add_argument("--reward_head_dropout", type=float, default=0.0)
    p.add_argument("--disc_max_prompts", type=int, default=7000)
    p.add_argument("--negatives_per_prompt", type=int, default=5)
    p.add_argument("--negative_top_p", type=float, default=0.90)
    p.add_argument("--negative_temperature", type=float, default=1.0)
    p.add_argument("--negative_max_new_tokens", type=int, default=256)
    p.add_argument("--disc_loss_every_n_steps", type=int, default=1)
    p.add_argument("--disc_loss_weight", type=float, default=1.0)
    p.add_argument("--margin_m0", type=float, default=0.0)
    p.add_argument("--margin_gamma", type=float, default=0.1)
    p.add_argument("--margin_t0", type=float, default=1000.0)
    args = p.parse_args()

    if args.load_in_8bit:
        args.load_in_4bit = False
    if not args.train_file and not args.dataset_name:
        raise SystemExit("Provide --train_file or --dataset_name.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model, tokenizer = load_model_and_tokenizer(args)
    ensure_pad_token(tokenizer)
    rows = load_rows(args)
    train_rows, eval_rows = split_rows(rows, args.val_ratio, args.seed)

    if args.disable_dual_head:
        train_ds = Dataset.from_list([{"text": record_to_text(r, tokenizer, args.use_chat_template)} for r in train_rows])
        eval_ds = Dataset.from_list([{"text": record_to_text(r, tokenizer, args.use_chat_template)} for r in eval_rows]) if eval_rows else None
        trainer_kwargs = {
            "model": model,
            "args": make_sft_config(args, prepared_dataset=False),
            "train_dataset": train_ds,
            "eval_dataset": eval_ds,
            "processing_class": tokenizer,
        }
        try:
            trainer = SFTTrainer(**trainer_kwargs)
        except TypeError:
            trainer_kwargs.pop("processing_class")
            trainer_kwargs["tokenizer"] = tokenizer
            trainer = SFTTrainer(**trainer_kwargs)
        trainer.train()
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        return

    attach_reward_head(model, args)
    train_ds, disc_train_prompts = build_dual_head_dataset(train_rows, tokenizer, args, train=True)
    eval_ds = build_dual_head_dataset(eval_rows, tokenizer, args, train=False)[0] if eval_rows else None
    trainer = DualHeadSFTTrainer(
        model=model,
        args=make_sft_config(args, prepared_dataset=True),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=DualHeadDataCollator(tokenizer=tokenizer),
        negatives_per_prompt=args.negatives_per_prompt,
        negative_top_p=args.negative_top_p,
        negative_temperature=args.negative_temperature,
        negative_max_new_tokens=args.negative_max_new_tokens,
        disc_loss_every_n_steps=args.disc_loss_every_n_steps,
        disc_loss_weight=args.disc_loss_weight,
        margin_m0=args.margin_m0,
        margin_gamma=args.margin_gamma,
        margin_t0=args.margin_t0,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    write_dual_head_config(args, disc_train_prompts)


if __name__ == "__main__":
    main()
