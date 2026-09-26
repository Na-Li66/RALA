import os
from argparse import ArgumentParser

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def adapter_weight_file(adapter_dir):
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        path = os.path.join(adapter_dir, name)
        if os.path.isfile(path):
            return path
    return None

def assert_policy_adapter_dir(adapter_dir, *, allow_root_policy_adapter=False):
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

if __name__ == '__main__':
    import torch
    from accelerate import Accelerator
    from accelerate import DistributedDataParallelKwargs

    kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters=True)]
    accelerator = Accelerator(mixed_precision="bf16", kwargs_handlers=kwargs_handlers)

    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, default=os.environ.get("MODEL_PATH", ""), help="Path to the model")
    parser.add_argument("--logdir", type=str, default="mbpp_log", help="Directory to save logs")
    parser.add_argument("--language", type=str, default="python", help="Language to evaluate")
    parser.add_argument("--dataroot", type=str, default="data", help="Path to data directory")
    parser.add_argument("--lora_dir", type=str, default=None, help="Path to the LoRA adapter (optional)")
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--max_gen_len", type=int, default=512)
    parser.add_argument("--max_prompt_tokens", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_sample", type=int, default=1)
    parser.add_argument("--k_sample", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.55)
    parser.add_argument("--top_p", type=float, default=0.85)
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit")
    parser.add_argument("--skip_score", action="store_true", help="Generate samples but skip functional scoring")
    parser.add_argument("--use_chat_template", action="store_true", help="Use the training chat template for prompts")
    parser.add_argument(
        "--prompt_style",
        choices=("plain", "fewshot"),
        default="plain",
        help="MBPP prompt format. 'fewshot' uses the 3-shot [BEGIN]/[DONE] MBPP protocol.",
    )
    parser.add_argument(
        "--chat_system_prompt",
        type=str,
        default="You are a helpful coding assistant. Provide a concise, correct answer. Stop after the final answer.",
    )
    parser.add_argument("--use_pad_token_as_eos", action="store_true")
    parser.add_argument("--extra_eos_token_ids", type=str, default="")
    parser.add_argument("--no_unsloth", action="store_true", help="Force Transformers+PEFT loading")
    parser.add_argument("--load_in_4bit", dest="load_in_4bit", action="store_true", default=True)
    parser.add_argument("--no_load_in_4bit", dest="load_in_4bit", action="store_false")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument(
        "--allow_root_policy_adapter",
        action="store_true",
        default=False,
        help="Allow loading checkpoint_xxx/policy_adapter root.",
    )
    args = parser.parse_args()
    if not str(args.model_path or "").strip():
        raise SystemExit("Provide --model_path or set MODEL_PATH.")
    if args.load_in_8bit:
        args.load_in_4bit = False

    model_path = args.model_path
    logdir = args.logdir
    language = args.language
    dataroot = args.dataroot

    try:
        from unsloth import FastLanguageModel  # type: ignore
        has_unsloth = True
    except Exception:
        FastLanguageModel = None  # type: ignore
        has_unsloth = False

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from mbpp import MBPP as evaltor

    if accelerator.is_main_process:
        print(f"Loading model from: {model_path}")
        print(f"Logging to: {logdir}")
        print(f"Dataroot: {dataroot}")
        print(
            "generation: "
            f"use_chat_template={args.use_chat_template} "
            f"max_prompt_tokens={args.max_prompt_tokens} "
            f"max_gen_len={args.max_gen_len} "
            f"temperature={args.temperature} top_p={args.top_p} "
            f"prompt_style={args.prompt_style} "
            f"use_pad_token_as_eos={args.use_pad_token_as_eos} "
            f"load_in_4bit={args.load_in_4bit} load_in_8bit={args.load_in_8bit}"
        )

    os.makedirs(logdir, exist_ok=True)

    tokenizer_cfg = dict(
        cls=AutoTokenizer,
        model_path=model_path,
    )
    if args.lora_dir:
        adapter_root = os.path.dirname(os.path.abspath(args.lora_dir))
        if os.path.isfile(os.path.join(adapter_root, "tokenizer_config.json")):
            tokenizer_cfg["model_path"] = adapter_root
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
        k_sample=args.k_sample,
        batch_size=args.batch_size,
        language=language,
        max_gen_len=args.max_gen_len,
        temperature=args.temperature,
        top_p=args.top_p,
        use_chat_template=args.use_chat_template,
        chat_system_prompt=args.chat_system_prompt,
        max_prompt_tokens=args.max_prompt_tokens,
        use_pad_token_as_eos=args.use_pad_token_as_eos,
        extra_eos_token_ids=extra_eos_token_ids,
        prompt_style=args.prompt_style,
        limit=args.limit,
        skip_score=args.skip_score,
    )

    if args.lora_dir:
        adapter_file = assert_policy_adapter_dir(
            args.lora_dir,
            allow_root_policy_adapter=bool(args.allow_root_policy_adapter),
        )
        if accelerator.is_main_process:
            print(f"Loading LoRA adapter from: {os.path.abspath(args.lora_dir)}")
            print(f"adapter_file: {adapter_file}")
        if has_unsloth and not args.no_unsloth:
            try:
                model, _tokenizer = FastLanguageModel.from_pretrained(  # type: ignore[misc]
                    model_name=args.lora_dir,
                    max_seq_length=args.max_seq_len,
                    dtype=None,
                    load_in_4bit=args.load_in_4bit,
                    load_in_8bit=args.load_in_8bit,
                )
                FastLanguageModel.for_inference(model)  # type: ignore[union-attr]
            except Exception as e:
                if accelerator.is_main_process:
                    print(f"[warn] Unsloth load(adapter) failed: {e}")
                model, _tokenizer = FastLanguageModel.from_pretrained(  # type: ignore[misc]
                    model_name=model_path,
                    max_seq_length=args.max_seq_len,
                    dtype=None,
                    load_in_4bit=args.load_in_4bit,
                    load_in_8bit=args.load_in_8bit,
                )
                model.load_adapter(args.lora_dir)
                FastLanguageModel.for_inference(model)  # type: ignore[union-attr]
        else:
            from peft import PeftModel
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
            model = PeftModel.from_pretrained(model, args.lora_dir)
    else:
        if has_unsloth and not args.no_unsloth:
            model, _tokenizer = FastLanguageModel.from_pretrained(  # type: ignore[misc]
                model_name=model_path,
                max_seq_length=args.max_seq_len,
                dtype=None,
                load_in_4bit=args.load_in_4bit,
                load_in_8bit=args.load_in_8bit,
            )
            FastLanguageModel.for_inference(model)  # type: ignore[union-attr]
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
    model.eval()

    evaluator.eval_model(model, accelerator)
