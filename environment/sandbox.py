"""
sandbox.py — Safe code execution environment.

Handles both test code styles:
  1. Simple top-level asserts:  assert add(2,3) == 5
  2. HumanEval def check style: def check(candidate): assert ...
"""

import subprocess
import tempfile
import os
import re
from typing import Optional


def count_assertions(test_code: str) -> int:
    """Count assert statements in test code."""
    return len(re.findall(r"^\s*assert\b", test_code, re.MULTILINE))


def _build_script(code: str, test_code: str, extra_imports: Optional[str]) -> str:
    """Build the full test script with pass/fail counting.

    KEY FIX: We preserve the original indentation of each line.
    This keeps 'def check(candidate):' bodies intact.
    We use _counts = [0, 0] (a list) so inner functions can modify
    the counter without Python's scoping issues with plain variables.
    """
    lines = ["# === solution ==="]
    if extra_imports:
        lines.append(extra_imports.strip())
    lines.append(code.strip())
    lines.append("")
    lines.append("# === test harness ===")
    lines.append("_counts = [0, 0]  # [passed, total]")
    lines.append("")

    for line in test_code.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            # keep blank lines and comments to preserve structure
            lines.append(line)
            continue

        if stripped.startswith("assert"):
            # Preserve original indentation of the assert line
            indent = line[: len(line) - len(line.lstrip())]
            lines.append(f"{indent}_counts[1] += 1  # total")
            lines.append(f"{indent}try:")
            lines.append(f"{indent}    {stripped}")
            lines.append(f"{indent}    _counts[0] += 1  # passed")
            lines.append(f"{indent}except Exception:")
            lines.append(f"{indent}    pass")
        else:
            # Non-assert line: keep exactly as-is (preserves def check, calls, etc.)
            lines.append(line)

    lines.append("")
    lines.append("print(f'RESULT:{_counts[0]}:{_counts[1]}')")
    return "\n".join(lines)


def execute_code(
    code: str,
    test_code: str,
    timeout: int = 5,
    extra_imports: Optional[str] = None,
) -> dict:
    """Execute generated code against tests in a sandboxed subprocess.

    Args:
        code: Python solution (function definition).
        test_code: Test code — either simple asserts or HumanEval def check style.
        timeout: Kill subprocess after this many seconds.
        extra_imports: Optional import lines to prepend.

    Returns:
        dict: reward (0.0-1.0), passed, total, output, error.
    """
    total = count_assertions(test_code)
    if total == 0:
        total = 1

    script = _build_script(code, test_code, extra_imports)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script)
            tmp_path = f.name

        result = subprocess.run(
            ["python3", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout + result.stderr
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
            "error": f"Timed out after {timeout}s (likely infinite loop)",
        }
    except Exception as e:
        return {
            "reward": 0.0,
            "passed": 0,
            "total": total,
            "output": "",
            "error": str(e),
        }
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


def _parse_result_line(output: str, fallback_total: int) -> tuple:
    for line in output.splitlines():
        if line.startswith("RESULT:"):
            try:
                _, p, t = line.split(":")
                return int(p), int(t)
            except Exception:
                pass
    return 0, fallback_total


# ── Smoke tests ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test 1: simple top-level assert
    r = execute_code("def add(a,b): return a+b", "assert add(2,3)==5")
    assert r["reward"] == 1.0, f"Test 1 failed: {r}"
    print("Test 1 (simple assert) ✅")

    # Test 2: wrong answer
    r = execute_code("def add(a,b): return 0", "assert add(2,3)==5")
    assert r["reward"] == 0.0, f"Test 2 failed: {r}"
    print("Test 2 (wrong answer) ✅")

    # Test 3: partial credit
    r = execute_code(
        "def add(a,b): return a+b if a>0 else 0",
        "assert add(2,3)==5\nassert add(0,5)==5"
    )
    assert r["reward"] == 0.5, f"Test 3 failed: {r}"
    print("Test 3 (partial credit) ✅")

    # Test 4: HumanEval def check style (THE KEY FIX)
    humaneval_test = """
METADATA = {'author': 'jt', 'dataset': 'test'}

def check(candidate):
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False
    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True

check(has_close_elements)
"""
    correct = "def has_close_elements(numbers, threshold):\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i]-numbers[j]) < threshold:\n                return True\n    return False"
    r = execute_code(correct, humaneval_test)
    assert r["reward"] == 1.0, f"Test 4 failed: {r}"
    print("Test 4 (HumanEval def check, correct) ✅")

    # Test 5: HumanEval style, wrong answer
    wrong = "def has_close_elements(numbers, threshold): return False"
    r = execute_code(wrong, humaneval_test)
    assert 0.0 <= r["reward"] < 1.0, f"Test 5 failed: {r}"
    print(f"Test 5 (HumanEval def check, wrong) ✅  reward={r['reward']}")

    # Test 6: setup line before assert
    r = execute_code("def add(a,b): return a+b", "x=add(2,3)\nassert x==5")
    assert r["reward"] == 1.0, f"Test 6 failed: {r}"
    print("Test 6 (setup line) ✅")

    # Test 7: timeout
    r = execute_code("def add(a,b):\n while True: pass", "assert add(2,3)==5", timeout=2)
    assert r["reward"] == 0.0 and "timed out" in r["error"].lower(), f"Test 7 failed: {r}"
    print("Test 7 (timeout) ✅")

    print("\nAll 7 sandbox tests passed! ✅")
