"""
parse.py — Extracts structured content from LLM output.

WHY: LLMs don't return raw code. They return natural language + code blocks.
We need to reliably pull out just the code (or critique) part.

We try multiple extraction strategies in order of preference:
  1. Fenced code block: ```python ... ```  ← most reliable
  2. XML-style tags: <code>...</code>
  3. Bare indented block (fallback, less reliable)

If nothing matches, we return an empty string and the caller should handle it.
"""

import re


def extract_code(text: str) -> str:
    """Extract Python code from LLM output.

    Tries these patterns in order:
    1. ```python ... ``` fenced block
    2. ``` ... ``` fenced block (no language tag)
    3. <code> ... </code> XML tags
    4. Returns the full text as a last resort (caller should validate)

    WHY last resort: If the model ignores formatting instructions, returning
    the raw text is better than returning empty — at least the caller can
    log it and debug.

    Args:
        text: Raw string output from the LLM.

    Returns:
        Extracted code string, stripped of leading/trailing whitespace.
        Empty string if input is empty.
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

    # Strategy 4: last resort — return the whole text
    # This happens when the model ignores formatting. Log a warning.
    print("[parse] WARNING: No code block found. Returning raw text.")
    return text.strip()


def extract_critique(text: str) -> str:
    """Extract the critique section from the Critic agent's output.

    Looks for <critique> ... </critique> XML tags.

    WHY XML tags for critique: We use markdown fences for code (standard),
    so we use XML tags for critique to avoid ambiguity.

    Args:
        text: Raw string output from the Critic LLM.

    Returns:
        The critique text, stripped. Empty string if not found.
    """
    if not text or not text.strip():
        return ""

    match = re.search(r"<critique>(.*?)</critique>", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback: maybe the model used different casing
    match = re.search(r"<CRITIQUE>(.*?)</CRITIQUE>", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    print("[parse] WARNING: No <critique> tag found in critic output.")
    return ""


def extract_thinking(text: str) -> str:
    """Extract the thinking/reasoning section from agent output.

    Looks for <thinking> ... </thinking> tags (used by the Coder agent).

    Args:
        text: Raw LLM output string.

    Returns:
        The thinking text, or empty string if not found.
    """
    if not text or not text.strip():
        return ""

    match = re.search(r"<thinking>(.*?)</thinking>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


# ── Quick smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test extract_code — fenced block
    sample1 = """
Here is my solution:
```python
def add(a, b):
    return a + b
```
I hope this works!
"""
    code = extract_code(sample1)
    print("Test 1 (fenced):", repr(code))
    assert "def add" in code

    # Test extract_code — XML tags
    sample2 = "The solution is: <code>def add(a, b):\n    return a + b</code>"
    code = extract_code(sample2)
    print("Test 2 (XML tags):", repr(code))
    assert "def add" in code

    # Test extract_critique
    sample3 = """
<critique>
The solution doesn't handle negative numbers correctly.
It will fail for edge case add(-1, -1).
</critique>
<code>
def add(a, b):
    return a + b
</code>
"""
    critique = extract_critique(sample3)
    print("Test 3 (critique):", repr(critique))
    assert "negative" in critique

    # Test empty input
    assert extract_code("") == ""
    assert extract_critique("") == ""

    print("\nAll parse tests passed!")
