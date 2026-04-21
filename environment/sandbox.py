"""
sandbox.py — Safe code execution environment.

WHY: We can never trust LLM-generated code. Running it directly with exec()
could delete files, make network calls, or hang forever. Instead, we write
the code to a temp file and run it in a subprocess with a hard timeout.

The reward is the fraction of tests that pass: passed / total.
This gives a smooth signal (0.0 to 1.0) for RL training, not just pass/fail.
"""

import subprocess
import tempfile
import os
import re
from typing import Optional


def count_assertions(test_code: str) -> int:
    """Count the number of assert statements in the test code.

    Args:
        test_code: String containing test assertions.

    Returns:
        Number of assert statements found.
    """
    return len(re.findall(r"^\s*assert\b", test_code, re.MULTILINE))


def execute_code(
    code: str,
    test_code: str,
    timeout: int = 5,
    extra_imports: Optional[str] = None,
) -> dict:
    """Execute generated code against test assertions in a sandboxed subprocess.

    WHY subprocess + tempfile: We never use exec() or eval() directly because
    that runs in our process and can cause damage. A subprocess is isolated —
    if it crashes or hangs, we just kill it. The timeout prevents infinite loops.

    Args:
        code: The Python solution to test (e.g. a function definition).
        test_code: Assert statements that test the solution.
        timeout: Max seconds before we kill the subprocess. Default 5.
        extra_imports: Optional import lines to prepend (e.g. "import math").

    Returns:
        dict with keys:
            - reward (float): passed / total, range [0.0, 1.0]
            - passed (int): number of assertions that passed
            - total (int): total number of assertions
            - output (str): stdout + stderr from the subprocess
            - error (str): exception message if execution failed entirely
    """
    total = count_assertions(test_code)
    if total == 0:
        # No assertions to check — treat as a syntax-only test
        total = 1

    # Build the full script: imports + solution + tests
    script_parts = []
    if extra_imports:
        script_parts.append(extra_imports.strip())
    script_parts.append(code.strip())
    script_parts.append("")
    # Wrap each assertion in its own try/except so one failure doesn't
    # stop us from counting the rest
    script_parts.append("_passed = 0")
    script_parts.append("_total = 0")
    for line in test_code.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue  # skip blank lines and comments
        if stripped.startswith("assert"):
            # wrap each assert in try/except for partial credit
            script_parts.append(f"_total += 1")
            script_parts.append(f"try:")
            script_parts.append(f"    {stripped}")
            script_parts.append(f"    _passed += 1")
            script_parts.append(f"except Exception:")
            script_parts.append(f"    pass")
        else:
            # setup lines like `result = fizzbuzz(15)` — run as-is
            script_parts.append(stripped)
    script_parts.append(f"print(f'RESULT:{{_passed}}:{{_total}}')")

    full_script = "\n".join(script_parts)

    # Write to a temp file and run it
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(full_script)
            tmp_path = f.name

        result = subprocess.run(
            ["python3", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout + result.stderr

        # Parse the RESULT line we printed
        passed, total_seen = _parse_result_line(output, total)
        reward = passed / total_seen if total_seen > 0 else 0.0

        return {
            "reward": round(reward, 4),
            "passed": passed,
            "total": total_seen,
            "output": output.strip(),
            "error": "",
        }

    except subprocess.TimeoutExpired:
        return {
            "reward": 0.0,
            "passed": 0,
            "total": total,
            "output": "",
            "error": f"Execution timed out after {timeout}s (likely infinite loop)",
        }
    except Exception as e:
        return {
            "reward": 0.0,
            "passed": 0,
            "total": total,
            "output": "",
            "error": f"Executor error: {str(e)}",
        }
    finally:
        # Always clean up the temp file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def _parse_result_line(output: str, fallback_total: int) -> tuple[int, int]:
    """Parse the 'RESULT:passed:total' line from subprocess output.

    Args:
        output: Full stdout+stderr string.
        fallback_total: Use this total if parsing fails.

    Returns:
        Tuple of (passed, total).
    """
    for line in output.splitlines():
        if line.startswith("RESULT:"):
            try:
                _, passed_str, total_str = line.split(":")
                return int(passed_str), int(total_str)
            except (ValueError, IndexError):
                pass
    # If we couldn't parse results, likely a syntax error → 0 passed
    return 0, fallback_total


# ── Quick smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Test 1 — correct solution:")
    r = execute_code(
        code="def add(a, b):\n    return a + b",
        test_code="assert add(2, 3) == 5\nassert add(-1, 1) == 0",
    )
    print(r)
    assert r["reward"] == 1.0, "Expected full reward"

    print("\nTest 2 — wrong solution:")
    r = execute_code(
        code="def add(a, b):\n    return 0",
        test_code="assert add(2, 3) == 5",
    )
    print(r)
    assert r["reward"] == 0.0, "Expected zero reward"

    print("\nTest 3 — partial credit (1 of 2 tests pass):")
    # add(2,3)=5 passes; add(0,5) returns 0 (a not > 0) but test expects 5 → fails
    r = execute_code(
        code="def add(a, b):\n    return a + b if a > 0 else 0",
        test_code="assert add(2, 3) == 5\nassert add(0, 5) == 5",
    )
    print(r)
    assert r["reward"] == 0.5, "Expected 0.5 reward"

    print("\nTest 4 — timeout (infinite loop):")
    r = execute_code(
        code="def add(a, b):\n    while True: pass",
        test_code="assert add(2, 3) == 5",
        timeout=2,
    )
    print(r)
    assert r["reward"] == 0.0
    assert "timed out" in r["error"]

    print("\nAll sandbox tests passed!")
