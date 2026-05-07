# RALA Code Package

This package contains training and evaluation code for RALA:

1. dual-head SFT,
2. generative RM training and AI preference data construction,
3. RALA RLHF with discriminative, generative, and endogenous rewards,
4. benchmark evaluation for reward learning, code generation, and math reasoning.

## Training

Dual-head SFT:

```bash
python train/dual_head_sft/train_sft.py \
  --model_id /path/to/base-model \
  --train_file /path/to/sft-data.jsonl \
  --output_dir /path/to/sft-output
```

Generative RM data construction:

```bash
python train/generative_rm/ai_preference_construction/generate_candidate_responses.py \
  --data_path /path/to/sft-prompts.jsonl \
  --base_model_dir /path/to/base-model \
  --sft_lora_dir /path/to/dual-head-sft-lora \
  --out_path /path/to/candidates.jsonl

python train/generative_rm/ai_preference_construction/select_max_distance_pairs.py \
  --input_path /path/to/candidates.jsonl \
  --output_path /path/to/max-margin-pairs.jsonl

python train/generative_rm/ai_preference_construction/build_ai_preference_pairs.py \
  --input_path /path/to/max-margin-pairs.jsonl \
  --output_path /path/to/preference-pairs.jsonl \
  --review_model_path /path/to/local-judge-model
```

Generative RM training:

```bash
python train/generative_rm/train_generative_verifier.py \
  --model_id /path/to/base-model \
  --pref_data_path /path/to/preference-pairs.jsonl \
  --output_dir /path/to/generative-rm-output
```

RALA RLHF:

```bash
python train/rala_rlhf/train_rala_code.py \
  --base_model /path/to/base-model \
  --sft_lora /path/to/dual-head-sft-lora \
  --reward_disc_head /path/to/reward_head.pt \
  --reward_gen_lora /path/to/generative-rm-lora \
  --data /path/to/rlhf-prompts.jsonl \
  --output_dir /path/to/rlhf-output
```

Use `train/rala_rlhf/train_rala_math.py` with the same path arguments for the
math setting.

## Evaluation

The evaluation scripts implement benchmark prompt construction and scoring for
the following metrics:

- RM-Bench: official 3x3 style-matrix protocol and report pairwise preference accuracy summarized by domain (Chat/Math/Code/Safety) and difficulty (Easy/Normal/Hard).
- HumanEval: pass@1 by default (`--n_sample 1 --k_sample 1`).
- MBPP: pass@1 on the 500-task split by default (`--n_sample 1 --k_sample 1`).
- MATH-500: pass@1.
- AIME 2024: avg@32 by default (`--num_samples 32`).

Datasets are not hardcoded. Pass benchmark data directories or files with the
CLI arguments shown below.

HumanEval:

```bash
python evaluation/HumanEval/eval_humaneval.py \
  --base_model /path/to/base-model \
  --checkpoint_dir /path/to/checkpoint-or-policy-adapter \
  --dataroot /path/to/humaneval-data \
  --logdir /path/to/humaneval-logs \
  --n_sample 1 \
  --k_sample 1
```

MBPP:

```bash
python evaluation/MBPP/eval_mbpp.py \
  --model_path /path/to/base-model \
  --lora_dir /path/to/policy-adapter \
  --dataroot /path/to/mbpp-data \
  --logdir /path/to/mbpp-logs \
  --n_sample 1 \
  --k_sample 1
```

The MBPP data root should contain `mbpp.jsonl` and `mbpp_test.jsonl`. With the
full `mbpp.jsonl`, the loader evaluates rows 10-509 for the 500-task protocol.
If only `mbpp_test.jsonl` is present, it evaluates that file as provided.

MATH-500:

```bash
python evaluation/MathBenchmark/eval_math500.py \
  --model_id /path/to/base-model \
  --lora_dir /path/to/policy-adapter \
  --data_path /path/to/math500.jsonl \
  --output_json /path/to/math500_result.json
```

AIME 2024:

```bash
python evaluation/MathBenchmark/eval_aime2024.py \
  --model_id /path/to/base-model \
  --lora_dir /path/to/policy-adapter \
  --data_path /path/to/aime2024.jsonl \
  --output_json /path/to/aime2024_result.json \
  --num_samples 32
```

RM-Bench:

```bash
python evaluation/RM-Bench/evaluate_rmbench.py \
  --method rala \
  --pairing official \
  --model_id /path/to/base-or-sft-model \
  --adapter_dir /path/to/sft-or-policy-adapter \
  --disc_adapter_dir /path/to/dual-head-sft-adapter \
  --disc_head /path/to/reward_head.pt \
  --reward_gen_lora /path/to/generative-rm-lora \
  --data_path /path/to/rmbench.jsonl \
  --output_json /path/to/rmbench_result.json
```

The official RM-Bench schema contains three `chosen` responses and three
`rejected` responses per prompt, ordered as concise, detailed plain text, and
detailed markdown. The evaluator compares all 3x3 pairs and writes aggregate
metrics under `summary.rmbench_official.table`.
