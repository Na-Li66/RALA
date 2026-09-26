import json
import os
import time

import numpy as np
import torch
from human_eval.evaluation import evaluate_functional_correctness
from transformers import AutoTokenizer
from utils.dataset import MBPPDataset
from utils.utils import postprocess_code_generation


class MBPP:
    def __init__(
        self,
        data_root,
        max_seq_len=2048,
        language="python",
        max_gen_len=80,
        batch_size=1,
        log_dir=None,
        temperature=0.55,
        issft=False,
        top_p=0.85,
        model_name="",
        inference_increment=True,
        tokenizer_cfg=None,
        n_sample=1,
        k_sample=1,
        use_chat_template=False,
        chat_system_prompt="",
        max_prompt_tokens=None,
        use_pad_token_as_eos=False,
        extra_eos_token_ids=None,
        prompt_style="plain",
        limit=None,
        skip_score=False,
    ):
        self.data_root = data_root
        self.max_seq_len = max_seq_len
        self.max_gen_len = max_gen_len
        self.batch_size = batch_size
        self.k = k_sample
        self.n_sample = n_sample
        self.language = language
        self.log_dir = log_dir
        self.sft = issft
        self.temperature = temperature
        self.top_p = top_p
        self.model_name = tokenizer_cfg["model_path"].replace("/", "_")
        self.inference_increment = inference_increment
        self.use_chat_template = use_chat_template
        self.chat_system_prompt = chat_system_prompt
        self.max_prompt_tokens = max_prompt_tokens
        self.use_pad_token_as_eos = use_pad_token_as_eos
        self.extra_eos_token_ids = extra_eos_token_ids or []
        self.prompt_style = prompt_style
        self.limit = limit
        self.skip_score = skip_score
        os.makedirs(self.log_dir, exist_ok=True)
        self.problem_file = os.path.join(self.data_root, "mbpp_test.jsonl")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_cfg["model_path"], trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.prompt_style not in {"plain", "fewshot"}:
            raise ValueError(f"Unsupported MBPP prompt_style: {self.prompt_style}")
        if self.prompt_style == "fewshot" and self.use_chat_template:
            raise ValueError("MBPP few-shot prompt style must not be wrapped in a chat template.")

    def _fewshot_example_prompt(self, data):
        tests = "\n".join(data["test"])
        code = data.get("code", "").replace("\r", "").replace("\t", "    ")
        return (
            f"You are an expert Python programmer, and here is your task: {data['prompt']} "
            f"Your code should pass these tests:\n\n{tests}\n[BEGIN]\n{code}\n[DONE]\n"
        )

    def _build_fewshot_prompt(self, data, dataset, prompt_limit):
        tests = "\n".join(data["test"])
        task_prompt = (
            f"You are an expert Python programmer, and here is your task: {data['prompt']} "
            f"Your code should pass these tests:\n\n{tests}\n[BEGIN]"
        )
        prefixes = []
        current = ""
        for example in getattr(dataset, "few_shot_examples", []):
            current += self._fewshot_example_prompt(example)
            prefixes.append(current)
        for prefix in reversed(prefixes):
            candidate = prefix + task_prompt
            if len(self.tokenizer(candidate, add_special_tokens=True, truncation=False)["input_ids"]) < prompt_limit:
                return candidate, True
        return task_prompt, True

    def _build_eval_prompt(self, data, dataset, prompt_limit):
        if self.prompt_style == "fewshot":
            return self._build_fewshot_prompt(data, dataset, prompt_limit)
        tests = "\n".join(data["test"])
        raw = (
            "Write a Python solution for the following task. Return only the code, "
            "without markdown fences.\n\n"
            f"Task: {data['prompt']}\n\n"
            f"Your code should pass these tests:\n{tests}"
        )
        if not self.use_chat_template:
            return raw, True
        messages = []
        if self.chat_system_prompt:
            messages.append({"role": "system", "content": self.chat_system_prompt})
        messages.append({"role": "user", "content": raw})
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True), False

    def _generation_eos_ids(self):
        eos_ids = []
        if self.tokenizer.eos_token_id is not None:
            eos_ids.append(int(self.tokenizer.eos_token_id))
        if self.use_pad_token_as_eos and self.tokenizer.pad_token_id is not None:
            eos_ids.append(int(self.tokenizer.pad_token_id))
        eos_ids.extend(int(value) for value in self.extra_eos_token_ids if value is not None)
        deduped = []
        for value in eos_ids:
            if value not in deduped:
                deduped.append(value)
        return deduped[0] if len(deduped) == 1 else deduped

    @torch.no_grad()
    def eval_model(self, gpt, accelerator):
        assert self.log_dir is not None, "log_dir should not be None when evaluating MBPP"
        dataset = MBPPDataset(self.data_root, samplenum=self.n_sample)
        self.problem_file = dataset.problem_file
        nprompt = len(dataset) // self.n_sample
        dp_rank = accelerator.process_index
        dp_size = accelerator.num_processes
        if self.k > 1:
            assert self.n_sample >= 80, "MBPP PASS@80 needs n_sample >= 80"
        gpt.eval()
        prompt_indices_split = np.array_split(range(nprompt), dp_size)
        indices = [x * self.n_sample + j for x in prompt_indices_split[dp_rank] for j in range(self.n_sample)]
        if self.limit is not None:
            indices = indices[: self.limit]
        all_num = len(indices)
        processed_num = 0
        log_file = os.path.join(
            self.log_dir,
            f"{self.model_name}_rank{dp_rank}_bs{self.batch_size}_shot_log_{self.language}.json",
        )
        tmpfile = open(log_file, "w", encoding="utf-8")
        start_time = time.time()
        eos_token_id = self._generation_eos_ids()
        prompt_limit = self.max_prompt_tokens
        if prompt_limit is None:
            prompt_limit = max(1, self.max_seq_len - self.max_gen_len)
        prompt_limit = min(prompt_limit, max(1, self.max_seq_len - self.max_gen_len))
        old_truncation_side = self.tokenizer.truncation_side
        self.tokenizer.truncation_side = "left"
        try:
            for idx in range(0, len(indices), self.batch_size):
                prompt_list = []
                add_special_tokens = []
                task_ids = []
                for j in indices[idx:idx + self.batch_size]:
                    data = dataset[j]
                    fprompt, add_special = self._build_eval_prompt(data, dataset, prompt_limit)
                    prompt_list.append(fprompt)
                    add_special_tokens.append(add_special)
                    task_ids.append(data["task_id"])
                if len(set(add_special_tokens)) != 1:
                    raise RuntimeError("Mixed add_special_tokens settings in one batch are unsupported.")
                tokenized = self.tokenizer(
                    prompt_list,
                    padding=True,
                    truncation=True,
                    max_length=prompt_limit,
                    return_tensors="pt",
                    add_special_tokens=add_special_tokens[0],
                )
                raw_lengths = [
                    len(
                        self.tokenizer(
                            item,
                            add_special_tokens=add_special_tokens[0],
                            truncation=False,
                        )["input_ids"]
                    )
                    for item in prompt_list
                ]
                input_ids = tokenized["input_ids"].to(gpt.device)
                attention_mask = tokenized["attention_mask"].to(gpt.device)
                generate_kwargs = dict(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_gen_len,
                    eos_token_id=eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                if self.temperature and self.temperature > 0:
                    generate_kwargs.update(do_sample=True, temperature=self.temperature, top_p=self.top_p)
                else:
                    generate_kwargs.update(do_sample=False)
                decoded = gpt.generate(**generate_kwargs)
                input_width = input_ids.shape[1]
                for local_idx, prediction_ids in enumerate(decoded):
                    response_ids = prediction_ids[input_width:]
                    raw_response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                    processed = postprocess_code_generation(raw_response)
                    processed["postprocess_meta"]["prompt_tokens"] = raw_lengths[local_idx]
                    processed["postprocess_meta"]["prompt_truncated"] = raw_lengths[local_idx] > prompt_limit
                    processed["postprocess_meta"]["prompt_limit"] = prompt_limit
                    res = {
                        "task_id": task_ids[local_idx],
                        "generation": processed["generation"],
                        "raw_response": processed["raw_response"],
                        "postprocess_meta": processed["postprocess_meta"],
                    }
                    tmpfile.write(json.dumps(res) + "\n")
                    tmpfile.flush()
                    processed_num += 1
                self.log_score(dp_rank, processed_num, all_num, start_time, self.batch_size)
        finally:
            self.tokenizer.truncation_side = old_truncation_side
            tmpfile.close()
        accelerator.wait_for_everyone()
        if not self.skip_score:
            self._calculate_final_score(accelerator)

    def log_score(self, dp_rank, processed_num, all_num, start_time, bs):
        mem = torch.cuda.max_memory_allocated() / (1 << 30)
        avg_time = (time.time() - start_time) / processed_num * bs
        print(
            f"DP RANK:{dp_rank} process_num/all_num:{int(processed_num)}/{all_num} "
            f"avg_time_per_batch:{avg_time:.2f} s "
            f"still_need:{((all_num - processed_num) // bs + 1) * avg_time / 60:.2f} m",
            f"mem:{mem:.3f} GiB bs:{bs}",
            flush=True,
        )
        if processed_num == all_num:
            print(f"EVAL DONE! Process time {(time.time() - start_time) / 60:.2f} m", flush=True)

    def _calculate_final_score(self, accelerator):
        if accelerator.is_local_main_process:
            logfilepath = os.path.join(self.log_dir, f"final_{self.model_name}.jsonl")
            with open(logfilepath, "w", encoding="utf-8") as logfile:
                for i in range(accelerator.num_processes):
                    tmplogfile = os.path.join(
                        self.log_dir,
                        f"{self.model_name}_rank{i}_bs{self.batch_size}_shot_log_{self.language}.json",
                    )
                    with open(tmplogfile, encoding="utf-8") as f:
                        logfile.write(f.read().strip() + "\n")
                    os.remove(tmplogfile)
            res = evaluate_functional_correctness(
                input_file=logfilepath,
                problem_file=self.problem_file,
                tmp_dir=self.log_dir,
                timeout=10,
                is_mbpp=True,
                language=self.language,
            )
            score_key = f"pass@{self.k}"
            if score_key in res:
                print("score is", res[score_key])
            else:
                print("score is", res)
            result_file = os.path.join(self.log_dir, f"score_{self.model_name}.json")
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=4)
            print(f"Final score saved to {result_file}")
