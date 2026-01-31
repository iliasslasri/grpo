import re
import subprocess
import tempfile
import os

def reward_function(completion_text, timeout=2):
    """
    Staircase Reward:
    0.0 -> No python code found
    0.2 -> Code found but has syntax errors
    0.5 -> Code runs but returns an error/crash
    1.0 -> Code runs successfully (Exit Code 0)
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
            # TODO 
            return 1.0 
        else:
            # It compiled, but crashed during logic (e.g., NameError, TypeError)
            return 0.5

    except subprocess.TimeoutExpired:
        return 0.1 # Heavily penalize infinite loops
    except Exception:
        return 0.0
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)