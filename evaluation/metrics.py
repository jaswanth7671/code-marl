"""
metrics.py — Evaluation metrics for code generation.

WHY pass@k: It's the standard metric for code generation (used in Codex,
AlphaCode, GPT-4 papers). pass@k means: if we sample k solutions, what's
the probability that at least one passes all tests?

WHY not just accuracy: A model might generate a correct solution only 30%
of the time (pass@1 = 0.30), but with k=10 samples, the chance of getting
at least one correct is much higher (pass@10 ≈ 0.97). This better reflects
real-world use where you can sample multiple times.

We use the unbiased estimator from the Codex paper (Chen et al. 2021):
    pass@k = 1 - C(n-c, k) / C(n, k)
where n = total samples, c = correct samples.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
from typing import Optional
import numpy as np

from environment.sandbox import execute_code
from utils.parse import extract_code


def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator for pass@k from the Codex paper.

    WHY unbiased estimator: Naively sampling k solutions and checking if any
    pass gives a biased (noisy) estimate. This formula uses all n samples to
    get an unbiased estimate, even when k < n.

    Reference: Chen et al. "Evaluating Large Language Models Trained on Code"
    (2021), Section 2.

    Args:
        n: Total number of samples generated for this problem.
        c: Number of samples that passed all tests.
        k: The k in pass@k.

    Returns:
        Estimated pass@k probability in [0.0, 1.0].

    Raises:
        ValueError: If k > n or inputs are negative.
    """
    if n < 0 or c < 0 or k < 0:
        raise ValueError(f"n, c, k must be non-negative. Got n={n}, c={c}, k={k}")
    if k > n:
        raise ValueError(f"k={k} cannot be greater than n={n}")
    if c > n:
        raise ValueError(f"c={c} cannot be greater than n={n}")
    if n == 0:
        return 0.0
    if n - c < k:
        # All remaining slots are taken by correct solutions → certain to pass
        return 1.0

    # Use log-space computation to avoid overflow for large n
    # pass@k = 1 - C(n-c, k) / C(n, k)
    # In log space: log(C(n-c, k)) - log(C(n, k))
    log_numerator = sum(math.log(n - c - i) for i in range(k))
    log_denominator = sum(math.log(n - i) for i in range(k))
    return 1.0 - math.exp(log_numerator - log_denominator)


def evaluate_pass_at_k(
    problems: list[dict],
    model_completions: list[list[str]],
    k_values: list[int] = [1, 5, 10],
) -> dict:
    """Evaluate pass@k across a set of problems and model completions.

    Args:
        problems: List of problem dicts (from dataset.py), each with "test_code".
        model_completions: List of lists. model_completions[i] is a list of
                           n completions for problems[i].
        k_values: Which k values to compute. Default [1, 5, 10].

    Returns:
        dict with keys like "pass@1", "pass@5", "pass@10", each a float in [0, 1].
    """
    assert len(problems) == len(model_completions), (
        f"problems ({len(problems)}) and completions ({len(model_completions)}) "
        f"must have the same length"
    )

    # For each problem, count how many completions pass
    per_problem_results = []
    for problem, completions in zip(problems, model_completions):
        n = len(completions)
        c = 0
        for completion in completions:
            code = extract_code(completion)
            if code:
                result = execute_code(code=code, test_code=problem["test_code"])
                if result["reward"] == 1.0:
                    c += 1
        per_problem_results.append({"n": n, "c": c})

    # Compute pass@k for each k
    metrics = {}
    for k in k_values:
        valid = [r for r in per_problem_results if r["n"] >= k]
        if not valid:
            metrics[f"pass@{k}"] = 0.0
            continue
        scores = [estimate_pass_at_k(r["n"], r["c"], k) for r in valid]
        metrics[f"pass@{k}"] = round(float(np.mean(scores)), 4)

    return metrics


def compute_debate_improvement(
    reward_before: list[float],
    reward_after: list[float],
) -> dict:
    """Compute debate improvement statistics.

    WHY track this: If debate consistently helps, the Critic is useful.
    If it hurts or has no effect, we need to tune the critic prompt.

    Args:
        reward_before: Rewards of initial Coder solutions (before debate).
        reward_after: Rewards of final solutions (after debate).

    Returns:
        dict with: mean_improvement, pct_improved, pct_hurt, pct_unchanged.
    """
    assert len(reward_before) == len(reward_after)
    deltas = [after - before for before, after in zip(reward_before, reward_after)]
    n = len(deltas)

    return {
        "mean_improvement": round(float(np.mean(deltas)), 4),
        "pct_improved": round(sum(d > 0 for d in deltas) / n, 4),
        "pct_hurt": round(sum(d < 0 for d in deltas) / n, 4),
        "pct_unchanged": round(sum(d == 0 for d in deltas) / n, 4),
    }


# ── Quick smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test pass@k estimator
    print("Testing pass@k estimator...")

    # If all 10 samples pass, pass@1 = 1.0
    assert estimate_pass_at_k(n=10, c=10, k=1) == 1.0

    # If no samples pass, pass@k = 0.0 for all k
    assert estimate_pass_at_k(n=10, c=0, k=1) == 0.0
    assert estimate_pass_at_k(n=10, c=0, k=5) == 0.0

    # pass@k increases with k (more chances to get it right)
    p1 = estimate_pass_at_k(n=10, c=3, k=1)
    p5 = estimate_pass_at_k(n=10, c=3, k=5)
    p10 = estimate_pass_at_k(n=10, c=3, k=10)
    assert p1 <= p5 <= p10, f"pass@k should increase: {p1} <= {p5} <= {p10}"
    print(f"  pass@1={p1:.3f}, pass@5={p5:.3f}, pass@10={p10:.3f} (increasing ✓)")

    # Test debate improvement
    before = [0.5, 0.0, 1.0, 0.5]
    after  = [1.0, 0.5, 0.5, 0.5]
    stats = compute_debate_improvement(before, after)
    print(f"\nDebate improvement stats: {stats}")
    assert stats["pct_improved"] == 0.5   # 2/4 improved
    assert stats["pct_hurt"] == 0.25      # 1/4 hurt
    assert stats["pct_unchanged"] == 0.25 # 1/4 unchanged

    print("\nAll metrics tests passed!")
