"""
reward_fn.py — Reward function called by GRPOTrainer during training.

GRPOTrainer calls reward_fn(completions, prompts) after sampling completions.
Each completion gets a reward in [0.0, 1.0] = fraction of tests passed.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from environment.sandbox import execute_code
from utils.parse import extract_code


def _get_text(completion) -> str:
    """Extract plain text from a completion regardless of its format.

    GRPOTrainer passes completions in different formats depending on TRL version:
      - Plain string: the raw generated text
      - List of dicts: [{"role": "assistant", "content": "..."}]
      - Single dict:  {"role": "assistant", "content": "..."}

    Args:
        completion: Completion in any of the above formats.

    Returns:
        Plain text string.
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        # chat format — take the last message content
        for msg in reversed(completion):
            if isinstance(msg, dict) and "content" in msg:
                return msg["content"]
        return ""
    if isinstance(completion, dict):
        return completion.get("content", "")
    return str(completion)


def reward_fn(
    completions,
    prompts,
    test_code=None,   # preferred: dataset column, repeated correctly by GRPOTrainer
    test_cases=None,  # legacy fallback (wrong when num_generations > 1)
    **kwargs,
) -> list[float]:
    """Reward function for GRPOTrainer.

    Args:
        completions: Completions from the model (string, list of dicts, etc.)
        prompts: Prompts used to generate completions.
        test_code: Per-completion test string from the dataset column (correct).
                   GRPOTrainer repeats each dataset row num_generations times,
                   so test_code[i] always matches the prompt that generated completion[i].
        test_cases: Legacy list indexed by i — only correct when num_generations=1.
        **kwargs: Extra args from GRPOTrainer (ignored).

    Returns:
        List of float rewards in [0.0, 1.0].
    """
    _debug = os.environ.get("MARL_DEBUG", "0") == "1"

    rewards = []
    for i, completion in enumerate(completions):
        try:
            text = _get_text(completion)

            if _debug and i == 0:
                print(f"\n[DEBUG] completion[0] (first 300 chars):\n{repr(text[:300])}\n")

            # Try extracting code normally
            code = extract_code(text)

            # Fallback: if prompt ended with ```python\n, completion IS the code
            if not code and text.strip():
                prompt_text = _get_text(prompts[i]) if i < len(prompts) else ""
                if "```python" in prompt_text:
                    # Wrap completion in fences and try again
                    code = extract_code(f"```python\n{text}\n```")

            if _debug and i == 0:
                print(f"[DEBUG] extracted code (first 200 chars):\n{repr(code[:200]) if code else 'EMPTY — reward=0'}\n")

            if not code:
                rewards.append(0.0)
                continue

            # Pick the right test: prefer dataset column (correct), fall back to list
            if test_code is not None and i < len(test_code):
                test = test_code[i]
            elif test_cases and i < len(test_cases):
                test = test_cases[i]
            else:
                rewards.append(0.0)
                continue

            result = execute_code(code=code, test_code=test)
            rewards.append(result["reward"])

        except Exception as e:
            print(f"[reward_fn] ERROR on completion {i}: {e}")
            rewards.append(0.0)

    return rewards


def make_reward_fn_with_tests(test_cases: list):
    """Factory that returns a reward_fn.

    test_code now comes from the dataset column (passed via **kwargs by GRPOTrainer).
    test_cases is kept as a fallback for smoke tests / single-generation use.
    """
    def _reward_fn(completions, prompts, **kwargs) -> list[float]:
        # If test_code not in kwargs (e.g. smoke test), inject from test_cases
        if "test_code" not in kwargs:
            kwargs["test_cases"] = test_cases
        return reward_fn(completions=completions, prompts=prompts, **kwargs)
    return _reward_fn


# ── Quick smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    completions = [
        "```python\ndef add(a, b):\n    return a + b\n```",  # correct
        "```python\ndef add(a, b):\n    return 0\n```",      # wrong
        "Here:\n\ndef add(a, b):\n    return a + b\n",       # no fence but has def
        "I don't know.",                                      # no code at all
    ]
    prompts = ["Write add(a, b)"] * 4
    tests   = ["assert add(2, 3) == 5"] * 4

    rewards = reward_fn(completions, prompts, test_cases=tests)
    print("Rewards:", rewards)
    assert rewards[0] == 1.0
    assert rewards[1] == 0.0
    assert rewards[2] == 1.0   # bare def — strategy 4 in parse.py catches it
    assert rewards[3] == 0.0
    print("All reward_fn tests passed!")
