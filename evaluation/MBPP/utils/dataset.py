import json
import os

import numpy as np


class MBPPDataset:
    def __init__(self, root, samplenum=1):
        self.root = root
        self.samplenum = samplenum
        full_path = os.path.join(root, "mbpp.jsonl")
        test_path = os.path.join(root, "mbpp_test.jsonl")
        data_path = full_path if os.path.isfile(full_path) else test_path
        self.data_path = data_path
        self.problem_file = test_path if os.path.isfile(test_path) else data_path
        self.data = open(data_path, encoding="utf-8").readlines()

        self.clean_data = self.get_qa_only_data(self.data)
        self.few_shot_examples = self.clean_data[1:4] if len(self.clean_data) >= 4 else []
        self.testdata = []
        start = 10 if len(self.clean_data) >= 510 else 0
        end = 510 if len(self.clean_data) >= 510 else len(self.clean_data)
        for i in range(start, end):
            for _ in range(samplenum):
                self.testdata.append(self.clean_data[i])
        self._validate_problem_file_alignment()
        np.random.seed(1234)
        print(
            f"Read MBPP from {data_path}, problem_file {self.problem_file}, "
            f"number of samples {len(self.testdata)}"
        )

    def get_qa_only_data(self, data_json):
        ans = []
        for line in data_json:
            line = json.loads(line)
            prompt = line.get("text") or line.get("prompt")
            tests = line.get("test_list") or line.get("test")
            code = line.get("code", "")
            ans.append({"prompt": prompt, "test": tests, "code": code, "task_id": line["task_id"]})
        return ans

    def __len__(self):
        return len(self.testdata)

    def __getitem__(self, index):
        return self.testdata[index]

    def _validate_problem_file_alignment(self):
        if not os.path.isfile(self.problem_file):
            return
        problems = self.get_qa_only_data(open(self.problem_file, encoding="utf-8").readlines())
        if not problems or len(self.testdata) % max(1, self.samplenum) != 0:
            # The fallback case is a custom problem file; the scorer will report
            # subset accuracy instead of pass@1 if the lengths differ.
            return
        samples = self.testdata[:: max(1, self.samplenum)]
        if len(problems) != len(samples):
            return
        for index, (sample, problem) in enumerate(zip(samples, problems)):
            same_prompt = sample["prompt"] == problem["prompt"]
            same_tests = sample["test"] == problem["test"]
            if sample["task_id"] == problem["task_id"] and same_prompt and same_tests:
                continue
            raise RuntimeError(
                "MBPP generation tasks do not align with scoring tasks: "
                f"{self.data_path} vs {self.problem_file} at index {index}"
            )
