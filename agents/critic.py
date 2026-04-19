"""
critic.py — The Critic agent (runs on local Qwen model, no API needed).

WHY same model for both agents: We're training the Coder with GRPO.
Using the same model as the Critic means both roles share learned
representations. As training progresses, the Critic gets smarter too —
creating a self-play dynamic that keeps improving the debate quality.

Requires calling agents.model.load_model() before using CriticAgent.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.model import generate
from utils.parse import extract_code, extract_critique
from utils.prompts import CRITIC_SYSTEM_PROMPT, format_critic_prompt


class CriticAgent:
    """Agent that reviews Python solutions and produces corrected versions.

    Uses the shared local Qwen model (loaded via agents.model.load_model()).
    The Critic always finds at least one issue (enforced by system prompt).

    Attributes:
        max_new_tokens: Max tokens to generate per response.
        temperature: Sampling temperature.
    """

    def __init__(
        self,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> None:
        """Initialize the Critic agent.

        Args:
            max_new_tokens: Max tokens for the generated response.
            temperature: Sampling temperature.
        """
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def review(self, problem: str, code: str) -> dict:
        """Review a proposed solution and produce critique + corrected code.

        Args:
            problem: The original problem statement.
            code: The Coder's proposed Python solution.

        Returns:
            dict with keys:
                - critique (str): Written critique text.
                - code (str): Corrected Python code from the Critic.
                - raw (str): Full raw model response (for debugging).
        """
        prompt = format_critic_prompt(problem=problem, code=code)
        raw = generate(
            system_prompt=CRITIC_SYSTEM_PROMPT,
            user_message=prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        return {
            "critique": extract_critique(raw),
            "code": extract_code(raw),
            "raw": raw,
        }


# ── Quick smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    from agents.model import load_model
    load_model()

    agent = CriticAgent()
    problem = (
        "def add(a: int, b: int) -> int:\n"
        '    """Return the sum of a and b."""\n'
    )
    code = "def add(a, b):\n    return a + b"

    print("Reviewing solution...")
    result = agent.review(problem=problem, code=code)
    print("Critique:", result["critique"])
    print("Corrected code:\n", result["code"])
