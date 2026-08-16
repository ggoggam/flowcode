"""Tests for pulling code out of a completion.

The cases below are transcribed from the shapes real chat models actually produce: fences
with and without a language tag, two attempts in one answer, a fence the model forgot to
close because it hit the token limit, prose wrapped around the code, the prompt restated
back verbatim.
"""

from __future__ import annotations

import pytest

from flowcode.envs.extract import extract_code, extract_code_blocks, is_parseable


def _defines(code: str, name: str) -> bool:
    return f"def {name}" in code


def test_plain_python_fence() -> None:
    completion = (
        "Here is the solution:\n\n"
        "```python\n"
        "def add(a, b):\n"
        "    return a + b\n"
        "```\n\n"
        "Let me know if you need anything else!"
    )
    code = extract_code(completion)
    assert code == "def add(a, b):\n    return a + b"
    assert "Here is the solution" not in code


def test_untagged_fence() -> None:
    completion = "```\ndef add(a, b):\n    return a + b\n```"
    assert extract_code(completion) == "def add(a, b):\n    return a + b"


def test_last_complete_python_block_wins() -> None:
    """Models write a wrong attempt, then correct themselves. The correction is the answer."""
    completion = (
        "First attempt:\n\n"
        "```python\n"
        "def add(a, b):\n"
        "    return a - b\n"
        "```\n\n"
        "Wait, that's subtraction. Here is the fix:\n\n"
        "```python\n"
        "def add(a, b):\n"
        "    return a + b\n"
        "```\n"
    )
    assert extract_code(completion) == "def add(a, b):\n    return a + b"


def test_python_tagged_block_beats_a_later_shell_block() -> None:
    completion = (
        "```python\n"
        "def add(a, b):\n"
        "    return a + b\n"
        "```\n\n"
        "Run it with:\n\n"
        "```bash\n"
        "python add.py\n"
        "```\n"
    )
    code = extract_code(completion)
    assert _defines(code, "add")
    assert "python add.py" not in code


def test_unterminated_fence_from_a_truncated_completion() -> None:
    completion = (
        "Sure! Here you go:\n\n```python\ndef add(a, b):\n    total = a + b\n    return total"
    )
    code = extract_code(completion)
    assert _defines(code, "add")
    assert "return total" in code
    assert "Sure!" not in code


def test_unfenced_code_with_prose_on_both_sides() -> None:
    completion = (
        "I think the cleanest way to do this is a simple loop.\n"
        "def total(xs):\n"
        "    out = 0\n"
        "    for x in xs:\n"
        "        out += x\n"
        "    return out\n"
        "This runs in linear time and uses constant extra space.\n"
    )
    code = extract_code(completion)
    assert is_parseable(code)
    assert _defines(code, "total")
    assert "linear time" not in code


def test_prose_after_code_inside_a_fence_is_trimmed() -> None:
    completion = (
        "```python\n"
        "def add(a, b):\n"
        "    return a + b\n"
        "\n"
        "This function adds two numbers together.\n"
        "```\n"
    )
    code = extract_code(completion)
    assert is_parseable(code)
    assert "adds two numbers" not in code


def test_model_restating_the_prompt_then_answering() -> None:
    completion = (
        "You asked: Write a function to add two numbers. Your code should pass "
        "these tests: assert add(1, 2) == 3\n\n"
        "```python\n"
        "def add(a, b):\n"
        "    return a + b\n"
        "```\n"
    )
    code = extract_code(completion)
    assert code.startswith("def add")
    assert "You asked" not in code


def test_imports_are_kept() -> None:
    completion = (
        "```python\n"
        "import math\n"
        "from collections import Counter\n"
        "\n"
        "def area(r):\n"
        "    return math.pi * r ** 2\n"
        "```\n"
    )
    code = extract_code(completion)
    assert "import math" in code
    assert "from collections import Counter" in code


def test_indented_fence_inside_a_list_item() -> None:
    completion = "1. Define it:\n\n   ```python\n   def add(a, b):\n       return a + b\n   ```\n"
    code = extract_code(completion)
    assert is_parseable(code)
    assert _defines(code, "add")


def test_repl_transcript_is_normalised() -> None:
    completion = "```\n>>> def add(a, b):\n...     return a + b\n>>> add(1, 2)\n```\n"
    code = extract_code(completion)
    assert is_parseable(code)
    assert _defines(code, "add")


def test_tilde_fences() -> None:
    completion = "~~~python\ndef add(a, b):\n    return a + b\n~~~\n"
    assert _defines(extract_code(completion), "add")


def test_language_tag_variants() -> None:
    for tag in ("python", "py", "Python", "python3"):
        completion = f"```{tag}\ndef add(a, b):\n    return a + b\n```"
        assert _defines(extract_code(completion), "add"), tag


def test_pure_prose_yields_empty_string() -> None:
    """An apology is not a program: it must score as `empty`, not as a syntax error."""
    completion = (
        "I'm sorry, but I can't help with that request. Perhaps you could try a "
        "different approach or consult the documentation."
    )
    assert extract_code(completion) == ""


def test_empty_completion_yields_empty_string() -> None:
    assert extract_code("") == ""
    assert extract_code("   \n\n  ") == ""
    assert extract_code("```python\n```") == ""


def test_broken_python_is_returned_not_discarded() -> None:
    """Code that is clearly meant to be code must reach the syntax-error bucket."""
    completion = "```python\ndef add(a, b)\n    return a + b\n```"
    code = extract_code(completion)
    assert code.strip() != ""
    assert not is_parseable(code)


def test_body_only_continuation_survives() -> None:
    """HumanEval-style: the model continues the given signature instead of restating it."""
    completion = (
        "```python\n    total = 0\n    for x in xs:\n        total += x\n    return total\n```"
    )
    code = extract_code(completion)
    assert code.startswith("    total = 0")
    assert "return total" in code


def test_extract_code_blocks_reports_closure() -> None:
    completion = "```python\na = 1\n```\ntext\n```python\nb = 2"
    blocks = extract_code_blocks(completion)
    assert [(info, closed) for info, _body, closed in blocks] == [
        ("python", True),
        ("python", False),
    ]


def test_extraction_is_deterministic() -> None:
    completion = "```python\ndef add(a, b):\n    return a + b\n```\nDone.\n"
    assert extract_code(completion) == extract_code(completion)


@pytest.mark.parametrize(
    "completion",
    [
        "```python\ndef f():\n    return 1\n```",
        "def f():\n    return 1\n",
        "Here:\n```\ndef f():\n    return 1\n```\nEnjoy.",
        "```py\ndef f():\n    return 1",
    ],
)
def test_every_shape_round_trips_to_runnable_code(completion: str) -> None:
    code = extract_code(completion)
    assert is_parseable(code)
    namespace: dict[str, object] = {}
    exec(compile(code, "<test>", "exec"), namespace)
    assert callable(namespace["f"])
