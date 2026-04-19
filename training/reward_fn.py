"""
reward_fn.py — Reward function called by GRPOTrainer during training.

WHY: GRPOTrainer needs a reward function with a specific signature:
    reward_fn(completions, prompts, **kwargs) -> list[float]

It calls this function after sampling N completions from the model.
Each completion gets a reward. GRPO then uses these rewards to compute
the policy gradient update — completions with higher reward get reinforced.

The reward is always [0.0, 1.0] = fraction of tests that passed.
This is better than binary (pass/fail) because it gives a gradient signal
even when the solution is partially correct.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.sandbox import execute_code
from utils.parse import extract_code


def reward_fn(
    completions: list[str],
    prompts: list[str],
    test_cases: list[str] | None = None,
    **kwargs,
) -> list[float]:
    """Reward function for GRPOTrainer.

    Called after the model generates `completions` for each `prompt`.
    We parse the code from each completion, run it against tests, and
    return a reward score.

    WHY **kwargs: GRPOTrainer may pass extra keyword arguments we don't need.
    Accepting **kwargs prevents crashes if the TRL version adds new params.

    Args:
        completions: List of raw text completions from the model.
                     Each should contain a Python code block.
        prompts: List of prompts the completions were generated from.
                 Same length as completions.
        test_cases: List of test assertion strings, one per prompt.
                    If None, uses kwargs["test_cases"] as fallback.
        **kwargs: Extra arguments from GRPOTrainer (ignored).

    Returns:
        List of float rewards in [0.0, 1.0], same length as completions.
    """
    # GRPOTrainer sometimes passes test_cases through kwargs
    if test_cases is None:
        test_cases = kwargs.get("test_cases", None)

    rewards = []
    for i, completion in enumerate(completions):
        try:
            code = extract_code(completion)
            if not code:
                # Model produced no code → zero reward
                rewards.append(0.0)
                continue

            # Get the test for this index (or use a default no-op)
            if test_cases and i < len(test_cases):
                test = test_cases[i]
            else:
                # No test available — can't evaluate, give 0
                print(f"[reward_fn] WARNING: No test case for completion {i}")
                rewards.append(0.0)
                continue

            result = execute_code(code=code, test_code=test)
            rewards.append(result["reward"])

        except Exception as e:
            print(f"[reward_fn] ERROR on completion {i}: {e}")
            rewards.append(0.0)

    return rewards


def make_reward_fn_with_tests(test_cases: list[str]):
    """Factory: returns a reward_fn pre-loaded with test cases.

    WHY factory: GRPOTrainer's reward_fn signature only takes completions
    and prompts. We can't pass test_cases directly. A closure captures them.

    Args:
        test_cases: List of test assertion strings, one per problem.

    Returns:
        A reward function with signature (completions, prompts, **kwargs).
    """
    def _reward_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
        """Reward function with pre-loaded test cases."""
        return reward_fn(
            completions=completions,
            prompts=prompts,
            test_cases=test_cases,
            **kwargs,
        )
    return _reward_fn


# ── Quick smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    completions = [
        "Here is my solution:\n```python\ndef add(a, b):\n    return a + b\n```",
        "Here is my solution:\n```python\ndef add(a, b):\n    return 0\n```",
        "I don't know how to solve this.",  # no code block
    ]
    prompts = ["Write add(a, b)"] * 3
    test_cases = ["assert add(2, 3) == 5"] * 3

    rewards = reward_fn(completions, prompts, test_cases=test_cases)
    print("Rewards:", rewards)
    assert rewards[0] == 1.0, "Correct solution should get 1.0"
    assert rewards[1] == 0.0, "Wrong solution should get 0.0"
    assert rewards[2] == 0.0, "No code block should get 0.0"

    # Test factory
    bounded_fn = make_reward_fn_with_tests(test_cases)
    rewards2 = bounded_fn(completions, prompts)
    assert rewards2 == rewards

    print("All reward_fn tests passed!")
