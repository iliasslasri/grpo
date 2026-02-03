import os
import re
import subprocess
import tempfile


def _structure_bonus(code):
    bonus = 0.0
    # Encourage reusable structure and clarity
    if re.search(r"^\s*def\s+\w+\s*\(", code, re.MULTILINE):
        bonus += 0.10
        # Docstring right after a def line
        if re.search(r"^\s*def\s+\w+\s*\(.*\):\s*\n\s*[ruRU]{0,2}['\"]{3}", code, re.MULTILINE):
            bonus += 0.05
        # Simple signal for type hints in defs
        if re.search(r"def\s+\w+\s*\([^)]*:\s*[^)]+\)\s*->\s*[^:]+:", code):
            bonus += 0.05
        elif re.search(r"def\s+\w+\s*\([^)]*\)", code):
            # print("has args", 0.03)
            bonus += 0.03
    if re.search(r"^\s*if\s+__name__\s*==\s*['\"]__main__['\"]\s*:", code, re.MULTILINE):
        bonus += 0.05
    # Gentle nudge for minimal explanations
    if re.search(r"^\s*#.+", code, re.MULTILINE):
        bonus += 0.02
    return bonus

def reward_function(completion_text, timeout=2):
    """
    Staircase Reward:
    0.0 -> No python code found
    0.2 -> Code found but has syntax errors
    0.4 -> Code runs but returns an error/crash
    0.6 -> Code runs successfully (Exit Code 0)
    +    -> Structured bonus up to 0.4 total (capped at 1.0 for now #TODO)
    """
    # Extraction (Regex for Markdown Python blocks)
    code_match = re.search(r"```python\n(.*?)\n```", completion_text, re.DOTALL)
    if not code_match:
        # Give a tiny reward if they at least attempted backticks without 'python' label
        code_match = re.search(r"```\n(.*?)\n```", completion_text, re.DOTALL)
        if not code_match: return 0.0
    
    code = code_match.group(1).strip()
    if not code: return 0.0

    # Syntax Check (Static Analysis)
    try:
        compile(code, "<string>", "exec")
    except Exception:
        return 0.2 # "At least you tried to write code"

    # Execution Check (Dynamic Analysis)
    with tempfile.NamedTemporaryFile(suffix=".py", mode='w', delete=False) as tmp:
        tmp.write(code)
        tmp_path = tmp.name

    try:
        # Run in a separate process for safety
        result = subprocess.run(
            ["python3", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.returncode == 0:
            base_reward = 0.6
        else:
            # It compiled, but crashed during logic (e.g., NameError, TypeError)
            base_reward = 0.4
        
        bonus = _structure_bonus(code)
        return min(1.0, base_reward + bonus)

    except subprocess.TimeoutExpired:
        return 0.1 # Heavily penalize infinite loops
    except Exception:
        return 0.0
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
