"""
coder.py — The Coder agent (runs on local Qwen model, no API needed).

WHY local model: No API cost, no rate limits, and the model we debate with
is the same one we train — so the debate loop directly reflects what the
trained model can do.

Requires calling agents.model.load_model() before using CoderAgent.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.model import generate
from utils.parse import extract_code, extract_thinking
from utils.prompts import CODER_SYSTEM_PROMPT, format_coder_revision_prompt


class CoderAgent:
    """Agent that writes Python solutions to coding problems.

    Uses the shared local Qwen model (loaded via agents.model.load_model()).
    No external API calls — runs entirely on your GPU/CPU.

    Attributes:
        max_new_tokens: Max tokens to generate per response.
        temperature: Sampling temperature (higher = more varied outputs).
    """

    def __init__(
        self,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> None:
        """Initialize the Coder agent.

        Args:
            max_new_tokens: Max tokens for the generated response.
            temperature: Sampling temperature. 0.7 gives good variety for RL.
        """
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def write_solution(self, problem: str) -> dict:
        """Write an initial solution to a coding problem.

        Args:
            problem: The problem statement (docstring + function signature).

        Returns:
            dict with keys:
                - code (str): Extracted Python code.
                - thinking (str): Extracted chain-of-thought reasoning.
                - raw (str): Full raw model response (for debugging).
        """
        raw = generate(
            system_prompt=CODER_SYSTEM_PROMPT,
            user_message=problem,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        return {
            "code": extract_code(raw),
            "thinking": extract_thinking(raw),
            "raw": raw,
        }

    def revise_solution(self, problem: str, critique: str) -> dict:
        """Revise a solution based on the Critic's feedback.

        Args:
            problem: The original problem statement.
            critique: The Critic's written critique.

        Returns:
            dict with keys:
                - code (str): Extracted revised Python code.
                - thinking (str): Extracted reasoning.
                - raw (str): Full raw model response.
        """
        prompt = format_coder_revision_prompt(critique=critique, problem=problem)
        raw = generate(
            system_prompt=CODER_SYSTEM_PROMPT,
            user_message=prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        return {
            "code": extract_code(raw),
            "thinking": extract_thinking(raw),
            "raw": raw,
        }


# ── Quick smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    from agents.model import load_model
    load_model()

    agent = CoderAgent()
    problem = (
        "def add(a: int, b: int) -> int:\n"
        '    """Return the sum of a and b."""\n'
    )
    print("Writing initial solution...")
    result = agent.write_solution(problem)
    print("Code:\n", result["code"])
    print("Thinking:", result["thinking"][:100] if result["thinking"] else "(none)")

    print("\nRevising with fake critique...")
    revised = agent.revise_solution(
        problem=problem,
        critique="The solution doesn't have type hints on the return value.",
    )
    print("Revised code:\n", revised["code"])
