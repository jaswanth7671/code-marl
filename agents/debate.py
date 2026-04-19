"""
debate.py — Orchestrates the multi-turn Coder ↔ Critic debate.

WHY multi-turn debate: A single LLM pass often produces suboptimal code.
By having the Critic find bugs and the Coder revise, we get iterative
refinement — similar to how humans do code review.

Flow for N rounds:
  Round 1: Coder writes initial solution
  Round 1: Critic reviews → gives critique + corrected code
  Round 2: Coder revises based on critique
  Round 2: Critic reviews again
  ... repeat N times ...
  Final: Evaluate the last Coder solution and return reward

The "debate improvement" metric = reward_after - reward_before.
If it's positive, the debate helped. If negative, the Critic made it worse
(a signal to tune the critic prompt).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.model import load_model
from agents.coder import CoderAgent
from agents.critic import CriticAgent
from environment.sandbox import execute_code


def run_debate(
    problem: str,
    test_code: str,
    n_rounds: int = 2,
    verbose: bool = True,
) -> dict:
    """Run a full Coder ↔ Critic debate on a coding problem.

    Args:
        problem: The problem statement (docstring + function signature).
        test_code: Assert statements for reward evaluation.
        n_rounds: Number of critique-and-revise cycles. Default 2.
        verbose: If True, print the debate transcript as it happens.

    Returns:
        dict with keys:
            - final_code (str): The last code produced by the Coder.
            - transcript (list[dict]): All turns in the debate.
            - reward_initial (float): Reward of the first Coder solution.
            - reward_final (float): Reward of the last Coder solution.
            - debate_improvement (float): reward_final - reward_initial.
    """
    load_model()  # no-op if already loaded
    coder = CoderAgent()
    critic = CriticAgent()
    transcript = []

    # ── Round 0: Initial solution ─────────────────────────────────────────────
    if verbose:
        print("=" * 60)
        print("DEBATE START")
        print("=" * 60)
        print(f"\nProblem:\n{problem}\n")
        print(f"Running {n_rounds} debate round(s)...\n")

    coder_result = coder.write_solution(problem)
    current_code = coder_result["code"]

    # Evaluate initial code
    initial_eval = execute_code(current_code, test_code)
    reward_initial = initial_eval["reward"]

    transcript.append({
        "turn": "coder_initial",
        "code": current_code,
        "thinking": coder_result["thinking"],
        "reward": reward_initial,
        "raw": coder_result["raw"],
    })

    if verbose:
        print(f"[Coder - Initial Solution] (reward: {reward_initial})")
        print(f"```python\n{current_code}\n```\n")

    # ── Debate rounds ─────────────────────────────────────────────────────────
    for round_num in range(1, n_rounds + 1):
        if verbose:
            print(f"--- Round {round_num} ---")

        # Critic reviews the current code
        critic_result = critic.review(problem=problem, code=current_code)
        critique_text = critic_result["critique"]
        critic_code = critic_result["code"]

        transcript.append({
            "turn": f"critic_round_{round_num}",
            "critique": critique_text,
            "code": critic_code,
            "raw": critic_result["raw"],
        })

        if verbose:
            print(f"\n[Critic - Round {round_num}]")
            print(f"Critique: {critique_text}\n")
            print(f"Critic's corrected code:\n```python\n{critic_code}\n```\n")

        # Coder revises based on critique
        revision_result = coder.revise_solution(
            problem=problem, critique=critique_text
        )
        current_code = revision_result["code"]

        # Evaluate the revision
        revision_eval = execute_code(current_code, test_code)
        round_reward = revision_eval["reward"]

        transcript.append({
            "turn": f"coder_revision_{round_num}",
            "code": current_code,
            "thinking": revision_result["thinking"],
            "reward": round_reward,
            "raw": revision_result["raw"],
        })

        if verbose:
            print(f"[Coder - Revision {round_num}] (reward: {round_reward})")
            print(f"```python\n{current_code}\n```\n")

    # ── Final evaluation ───────────────────────────────────────────────────────
    final_eval = execute_code(current_code, test_code)
    reward_final = final_eval["reward"]
    debate_improvement = round(reward_final - reward_initial, 4)

    if verbose:
        print("=" * 60)
        print(f"DEBATE COMPLETE")
        print(f"  Reward before debate: {reward_initial}")
        print(f"  Reward after debate:  {reward_final}")
        print(f"  Improvement:          {debate_improvement:+.4f}")
        print("=" * 60)

    return {
        "final_code": current_code,
        "transcript": transcript,
        "reward_initial": reward_initial,
        "reward_final": reward_final,
        "debate_improvement": debate_improvement,
    }


# ── Run as script ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test on a simple problem
    test_problem = (
        "def fizzbuzz(n: int) -> list[str]:\n"
        '    """Return a list of strings for numbers 1 to n.\n'
        "    For multiples of 3, use 'Fizz'.\n"
        "    For multiples of 5, use 'Buzz'.\n"
        "    For multiples of both, use 'FizzBuzz'.\n"
        "    Otherwise, use the number as a string.\n"
        '    """\n'
    )
    test_assertions = (
        "result = fizzbuzz(15)\n"
        "assert result[0] == '1'\n"
        "assert result[2] == 'Fizz'\n"
        "assert result[4] == 'Buzz'\n"
        "assert result[14] == 'FizzBuzz'\n"
        "assert len(result) == 15\n"
    )

    outcome = run_debate(
        problem=test_problem,
        test_code=test_assertions,
        n_rounds=2,
        verbose=True,
    )

    print("\nFinal code:")
    print(outcome["final_code"])
