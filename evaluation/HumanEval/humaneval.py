import time
import string
import multiprocessing
import os
import numpy as np
import json
import re
import torch
import datetime
import subprocess
import torch.distributed as dist
# from attrdict import AttrDict
from human_eval.evaluation import evaluate_functional_correctness
from transformers import AutoTokenizer
from utils.dataset import HumanEvalDataset
from utils.utils import postprocess_humaneval_generation

class HumanEval:
    """
    HumanEval evaluation class.
    """
    def __init__(self, data_root, max_seq_len=2048,
                language="python", max_gen_len=200, batch_size=512,
                log_dir=None, temperature=0, issft=False, top_p=0.95,
                model_name="", inference_increment=True,
                tokenizer_cfg=None, n_sample=40, k_sample=1,
                use_chat_template=False, chat_system_prompt="",
                max_prompt_tokens=None, use_pad_token_as_eos=False,
                extra_eos_token_ids=None, limit=None, skip_score=False):
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
        self.limit = limit
        self.skip_score = skip_score
        os.makedirs(self.log_dir, exist_ok=True)
        # tokenizer_cls = tokenizer_cfg.pop('cls')
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_cfg["model_path"], trust_remote_code=True)
        except Exception as e:
            print(e)
            assert False
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _build_eval_prompt(self, raw_prompt):
        raw_prompt = raw_prompt.strip()
        if not self.use_chat_template:
            return raw_prompt, True
        user_prompt = (
            "Complete the following HumanEval Python task. Return only valid "
            "Python code, without markdown fences, explanations, tests, or "
            "example calls. You may return either the indented body that "
            "completes the target function, or a complete definition of the "
            "target function and any helper code needed.\n\n"
            f"{raw_prompt}"
        )
        messages = []
        if self.chat_system_prompt:
            messages.append({"role": "system", "content": self.chat_system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        ), False

    def _generation_eos_ids(self):
        eos_ids = []
        if self.tokenizer.eos_token_id is not None:
            eos_ids.append(int(self.tokenizer.eos_token_id))
        if self.use_pad_token_as_eos and self.tokenizer.pad_token_id is not None:
            eos_ids.append(int(self.tokenizer.pad_token_id))
        for value in self.extra_eos_token_ids:
            if value is not None:
                eos_ids.append(int(value))
        deduped = []
        for value in eos_ids:
            if value not in deduped:
                deduped.append(value)
        return deduped[0] if len(deduped) == 1 else deduped

    def _postprocess_generation(self, text, original_prompt):
        if self.language.lower() != "python":
            return {
                "raw_response": text,
                "clean_response": text,
                "completion": text,
                "generation": original_prompt.rstrip() + "\n" + text,
                "postprocess_meta": {"kind": "raw_non_python"},
            }
        return postprocess_humaneval_generation(text, original_prompt)

    @torch.no_grad()
    def eval_model(self, gpt, accelerator):
        """
        Evaluate the model on HumanEval.
        """
        assert self.log_dir is not None, "log_dir should not be None when evaluating humaneval"
        dataset = HumanEvalDataset(self.data_root, sample_num=self.n_sample, language=self.language, issft=self.sft)
        nprompt = len(dataset) // self.n_sample
        dp_rank = accelerator.process_index
        dp_size = accelerator.num_processes
        if self.k > 1:
            assert self.n_sample >= 100, "HumanEval PASS@100 needs n_sample >= 100"
        gpt.eval()
        # each process will process a subset of the dataset
        prompt_indices_split = np.array_split(range(nprompt), dp_size)
        prompt_indices = prompt_indices_split[dp_rank]
        indices = [x * self.n_sample + j for x in prompt_indices for j in range(self.n_sample)]
        if self.limit is not None:
            indices = indices[: self.limit]
        all_num = len(indices)
        processed_num = 0
        log_file = os.path.join(self.log_dir,
                                    f'{self.model_name}_rank{dp_rank}_bs{self.batch_size}_shot_log_{self.language}.json')
        tmpfile = open(log_file, "w")
        start_time = time.time()
        eos_token_id = self._generation_eos_ids()
        prompt_limit = self.max_prompt_tokens
        if prompt_limit is None:
            prompt_limit = max(1, self.max_seq_len - self.max_gen_len)
        prompt_limit = min(prompt_limit, max(1, self.max_seq_len - self.max_gen_len))
        old_truncation_side = self.tokenizer.truncation_side
        self.tokenizer.truncation_side = "left"
        # split the dataset into batches and construct a list of inputs
        try:
            for idx in range(0, len(indices), self.batch_size):
                prompt_list = []
                add_special_tokens = []
                orriginal_prompt_list = []
                taskid = []
                # get the prompts from the dataset
                for j in indices[idx:idx + self.batch_size]:
                    data = dataset[j]
                    fprompt, add_special = self._build_eval_prompt(data["prompt"])
                    prompt_list.append(fprompt)
                    add_special_tokens.append(add_special)
                    orriginal_prompt_list.append(data["original_prompt"])
                    taskid.append(data["task_id"])
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
                input_ids = tokenized["input_ids"].to(accelerator.device)
                attention_mask = tokenized["attention_mask"].to(accelerator.device)
                input_width = input_ids.shape[1]
                generate_kwargs = dict(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_gen_len,
                    eos_token_id=eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                if self.temperature != 0:
                    generate_kwargs.update(do_sample=True, temperature=self.temperature, top_p=self.top_p)
                else:
                    generate_kwargs.update(do_sample=False)
                decoded = gpt.generate(**generate_kwargs)
                # save the results to a file
                for local_idx, text in enumerate(decoded):
                    prediction_ids = decoded[local_idx]
                    response_ids = prediction_ids[input_width:]
                    raw_response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                    processed = self._postprocess_generation(raw_response, orriginal_prompt_list[local_idx])
                    suffixprediction = processed["generation"]
                    processed["postprocess_meta"]["prompt_tokens"] = raw_lengths[local_idx]
                    processed["postprocess_meta"]["prompt_truncated"] = raw_lengths[local_idx] > prompt_limit
                    processed["postprocess_meta"]["prompt_limit"] = prompt_limit
                    wholecode = self.tokenizer.decode(prediction_ids, skip_special_tokens=True)
                    res = {
                        "task_id": taskid[local_idx],
                        "generation": suffixprediction,
                        "completion": processed["completion"],
                        "prompt": orriginal_prompt_list[local_idx],
                        "raw_response": processed["raw_response"],
                        "clean_response": processed["clean_response"],
                        "postprocess_meta": processed["postprocess_meta"],
                        "wholecode": wholecode,
                    }
                    tmpfile.write(json.dumps(res) + "\n")
                    tmpfile.flush()
                    processed_num += 1
                self.log_score(dp_rank, processed_num, all_num, start_time, self.batch_size)
        finally:
            self.tokenizer.truncation_side = old_truncation_side
        tmpfile.close()
        accelerator.wait_for_everyone()
        # calculate the final score of pass@k
        if not self.skip_score:
            self._calculate_final_score(accelerator)
        accelerator.wait_for_everyone()
        return

    def log_score(self, dp_rank, processed_num, all_num, start_time, bs):
        """
        Log the score.
        """
        mem = torch.cuda.max_memory_allocated() / (1 << 30)
        avg_time = (time.time() - start_time) / processed_num * bs
        print(
            f'DP RANK:{dp_rank} process_num/all_num:{int(processed_num)}/{all_num} '
            f'avg_time_per_batch:{avg_time:.2f} s '
            f'still_need:{((all_num - processed_num) // bs + 1) * avg_time / 60:.2f} m',
            f'mem:{mem:.3f} GiB bs:{bs}',
            flush=True
        )
        if processed_num == all_num:
            print(f'EVAL DONE! Process time {(time.time() - start_time) / 60:.2f} m', flush=True)

    def _calculate_final_score(self, accelerator):
        """
        Calculate the final score.
        """
        if accelerator.is_local_main_process:
            logfilepath = os.path.join(self.log_dir, f'final_{self.model_name}.jsonl')
            logfile = open(logfilepath, "w")
            for i in range(accelerator.num_processes):
                tmplogfile = os.path.join(self.log_dir, f'{self.model_name}_rank{i}_bs{self.batch_size}_shot_log_{self.language}.json')
                logfile.write(open(tmplogfile).read().strip() + "\n")
                os.remove(tmplogfile)
            logfile.close()
            timeout = 10
            runlang = self.language
            res = evaluate_functional_correctness(input_file=logfilepath, problem_file=os.path.join(self.data_root, f"humaneval-{self.language}.jsonl"), tmp_dir=self.log_dir, timeout=timeout, language=runlang)
            score_key = 'pass@%d' % self.k
            if score_key in res:
                print("score is", res[score_key])
            else:
                print("score is", res)

            # Save final results to a file
            result_file = os.path.join(self.log_dir, f'score_{self.model_name}.json')
            with open(result_file, 'w') as f:
                json.dump(res, f, indent=4)
            print(f"Final score saved to {result_file}")

            # os.remove(logfilepath)  # Keep the generated code file
        return
