import os
import sys

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.rewards import reward_function


def test_no_code():
    text = "Just an explanation with no code block."
    assert reward_function(text) == 0.0


def test_empty_code_block():
    text = "```python\n\n```"
    assert reward_function(text) == 0.0


def test_syntax_error():
    text = "```python\nfor\n```"
    assert reward_function(text) == 0.2


def test_runtime_error():
    text = "```python\nraise ValueError('boom')\n```"
    assert reward_function(text) == pytest.approx(0.4, abs=1e-6)


def test_success_minimum():
    text = "```python\nprint('ok')\n```"
    assert reward_function(text) == pytest.approx(0.6, abs=1e-6)


def test_success_with_structure_bonus():
    text = (
        "```python\n"
        "def add(a: int, b: int) -> int:\n"
        "    \"\"\"Add two numbers.\"\"\"\n"
        "    # simple example\n"
        "    return a + b\n"
        "\n"
        "if __name__ == \"__main__\":\n"
        "    print(add(1, 2))\n"
        "```\n"
    )
    # base reward: 0.6
    # bonus: 0.22
        # def 0.1
        # type hints in defs 0.05
        # add examples 0.05
        # explanations 0.02

    assert reward_function(text) == pytest.approx(0.82, abs=1e-6)


def test_timeout():
    text = "```python\nwhile True:\n    pass\n```"
    assert reward_function(text, timeout=0.05) == pytest.approx(0.1, abs=1e-6)
