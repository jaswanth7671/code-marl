"""
parse.py — Extracts structured content from LLM output.

Tries multiple strategies in order. For training, the most important
case is when the model generates code directly (prompt ended with ```python\n).
"""

import re


def extract_code(text: str) -> str:
    """Extract Python code from LLM output.

    Strategies tried in order:
    1. ```python ... ``` fenced block
    2. ``` ... ``` fenced block (no language tag)
    3. <code> ... </code> XML tags
    4. Bare function/class definition (model wrote code without fences)
    5. Returns empty string — caller should handle missing code as reward=0

    Args:
        text: Raw string output from the LLM.

    Returns:
        Extracted code string. Empty string if no code found.
    """
    if not text or not text.strip():
        return ""

    # Strategy 1: ```python ... ```
    match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Strategy 2: ``` ... ``` (no language tag)
    match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Strategy 3: <code> ... </code>
    match = re.search(r"<code>(.*?)</code>", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Strategy 4: bare function/class definition
    # Handles cases where model writes code without fences e.g.:
    #   "Here is my solution:\n\ndef fizzbuzz(n):\n    ..."
    match = re.search(
        r"((?:(?:import|from)\s+\S+[^\n]*\n)*(?:def|class)\s+\w+.*)",
        text,
        re.DOTALL,
    )
    if match:
        return match.group(1).strip()

    # Nothing found — return empty, reward_fn will give 0.0
    return ""


def extract_critique(text: str) -> str:
    """Extract the critique section from the Critic agent's output."""
    if not text or not text.strip():
        return ""
    match = re.search(r"<critique>(.*?)</critique>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"<CRITIQUE>(.*?)</CRITIQUE>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def extract_thinking(text: str) -> str:
    """Extract the thinking/reasoning section from agent output."""
    if not text or not text.strip():
        return ""
    match = re.search(r"<thinking>(.*?)</thinking>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


# ── Quick smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test 1: fenced block
    s1 = "Here:\n```python\ndef add(a, b):\n    return a + b\n```"
    assert "def add" in extract_code(s1), "Test 1 failed"

    # Test 2: no language tag
    s2 = "Solution:\n```\ndef add(a, b):\n    return a + b\n```"
    assert "def add" in extract_code(s2), "Test 2 failed"

    # Test 3: XML tags
    s3 = "<code>def add(a, b):\n    return a + b</code>"
    assert "def add" in extract_code(s3), "Test 3 failed"

    # Test 4: bare function definition (no fences)
    s4 = "Here is the answer:\n\ndef add(a, b):\n    return a + b\n"
    assert "def add" in extract_code(s4), "Test 4 failed"

    # Test 5: empty input
    assert extract_code("") == "", "Test 5 failed"

    # Test 6: no code at all
    assert extract_code("I don't know the answer.") == "", "Test 6 failed"

    print("All parse tests passed!")
