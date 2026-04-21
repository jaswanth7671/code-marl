import subprocess, tempfile, os, re
from typing import Optional


def count_assertions(test_code: str) -> int:
    return len(re.findall(r"^\s*assert\b", test_code, re.MULTILINE))


def execute_code(
    code: str,
    test_code: str,
    timeout: int = 5,
    extra_imports: Optional[str] = None,
) -> dict:
    """Execute generated code against test assertions in a sandboxed subprocess.

    Returns dict with: reward (0.0-1.0), passed, total, output, error.
    """
    total = count_assertions(test_code)
    if total == 0:
        total = 1

    script_parts = []
    if extra_imports:
        script_parts.append(extra_imports.strip())
    script_parts.append(code.strip())
    script_parts.append("")
    script_parts.append("_passed = 0")
    script_parts.append("_total = 0")

    for line in test_code.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("assert"):
            # wrap each assert so one failure doesn't stop the rest
            script_parts.append("_total += 1")
            script_parts.append("try:")
            script_parts.append(f"    {stripped}")
            script_parts.append("    _passed += 1")
            script_parts.append("except Exception:")
            script_parts.append("    pass")
        else:
            # setup lines like `result = fizzbuzz(15)` — run as-is
            script_parts.append(stripped)

    script_parts.append("print(f'RESULT:{_passed}:{_total}')")
    full_script = "\n".join(script_parts)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(full_script)
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


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test 1: correct solution
    r = execute_code("def add(a,b): return a+b", "assert add(2,3)==5")
    assert r["reward"] == 1.0, r
    print("Test 1 passed ✅")

    # Test 2: wrong solution
    r = execute_code("def add(a,b): return 0", "assert add(2,3)==5")
    assert r["reward"] == 0.0, r
    print("Test 2 passed ✅")

    # Test 3: partial credit
    r = execute_code("def add(a,b): return a+b if a>0 else 0",
                     "assert add(2,3)==5\nassert add(0,5)==5")
    assert r["reward"] == 0.5, r
    print("Test 3 passed ✅")

    # Test 4: setup line (result = func()) before asserts
    r = execute_code("def add(a,b): return a+b",
                     "x = add(2,3)\nassert x == 5")
    assert r["reward"] == 1.0, r
    print("Test 4 passed ✅")

    # Test 5: timeout
    r = execute_code("def add(a,b):\n while True: pass",
                     "assert add(2,3)==5", timeout=2)
    assert r["reward"] == 0.0
    assert "timed out" in r["error"].lower()
    print("Test 5 passed ✅")

    print("\nAll sandbox tests passed!")
