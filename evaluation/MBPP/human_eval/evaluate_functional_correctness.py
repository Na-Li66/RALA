import fire
import sys

from .data import HUMAN_EVAL
from .evaluation import evaluate_functional_correctness


def entry_point(
    sample_file: str,
    k: str = "1,10,100",
    n_workers: int = 4,
    timeout: float = 3.0,
    problem_file: str = "",
    is_mbpp: bool = False,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    k = list(map(int, k.split(",")))
    results = evaluate_functional_correctness(
        input_file=sample_file,
        n_workers=n_workers,
        timeout=timeout,
        problem_file=problem_file or HUMAN_EVAL,
        k=k,
        is_mbpp=is_mbpp,
    )
    print(results)


def main():
    fire.Fire(entry_point)


sys.exit(main())
