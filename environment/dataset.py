"""
dataset.py — Loads HumanEval problems from HuggingFace.

WHY HumanEval: It's the standard benchmark for code generation. Each problem
has a docstring (the prompt), a function signature, and canonical unit tests.
This gives us verifiable rewards without needing human labelers.

Each item we return has:
    - problem_id: e.g. "HumanEval/0"
    - prompt: the docstring + function signature the model must complete
    - test_code: assert statements to evaluate correctness
    - entry_point: the function name (used to call it in tests)
"""

from typing import Optional
from datasets import load_dataset


def load_humaneval(split: str = "test", max_problems: Optional[int] = None) -> list[dict]:
    """Load HumanEval problems from HuggingFace datasets.

    WHY: We use the official HuggingFace version so we get the exact same
    benchmark used in GPT-4, Codex, etc. papers. This makes our results
    comparable to published work.

    Args:
        split: Dataset split. HumanEval only has "test" (164 problems).
        max_problems: If set, return only the first N problems. Useful for
                      quick debugging without running all 164.

    Returns:
        List of dicts, each with keys: problem_id, prompt, test_code, entry_point.
    """
    try:
        ds = load_dataset("openai/openai_humaneval", split=split, trust_remote_code=True)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load HumanEval dataset. "
            f"Make sure you have `datasets` installed and internet access.\n"
            f"Original error: {e}"
        )

    problems = []
    for i, item in enumerate(ds):
        if max_problems is not None and i >= max_problems:
            break
        problems.append(
            {
                "problem_id": item["task_id"],
                "prompt": item["prompt"],
                "test_code": item["test"],          # canonical assert statements
                "entry_point": item["entry_point"],  # function name to call
                "canonical_solution": item["canonical_solution"],
            }
        )

    return problems


def format_problem_for_coder(problem: dict) -> str:
    """Format a HumanEval problem into a prompt string for the Coder agent.

    WHY: We give the model the full docstring so it understands what to build.
    The function signature is included so the model knows what to name things.

    Args:
        problem: A problem dict from load_humaneval().

    Returns:
        A string prompt ready to send to the Coder agent.
    """
    return (
        f"Complete the following Python function:\n\n"
        f"```python\n{problem['prompt']}\n```\n\n"
        f"Write a complete, correct implementation."
    )


# ── Quick smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading first 3 HumanEval problems...")
    problems = load_humaneval(max_problems=3)
    for p in problems:
        print(f"\n--- {p['problem_id']} ({p['entry_point']}) ---")
        print("Prompt (first 200 chars):", p["prompt"][:200])
        print("Test code (first 200 chars):", p["test_code"][:200])
    print(f"\nLoaded {len(problems)} problems successfully.")
