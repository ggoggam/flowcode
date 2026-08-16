"""Tests for the execution reward.

The numbers here are the ones the objective actually consumes, so they are asserted
exactly rather than approximately: two of four tests passing is 0.5, not "about a half".
"""

from __future__ import annotations

import itertools
import math
import time
from typing import Any

import pytest

from flowcode.envs.base import (
    ERROR_ASSERTION,
    ERROR_EMPTY,
    ERROR_RUNTIME,
    ERROR_SYNTAX,
    ERROR_TIMEOUT,
    Environment,
    Task,
)
from flowcode.envs.code_exec import SENTINEL, CodeExecEnv


def make_env(**overrides: object) -> CodeExecEnv:
    """A fast environment for tests: short budgets, fixtures never loaded."""
    kwargs: dict[str, Any] = {
        "name": "test",
        "dataset": "fixtures",
        "split": "all",
        "timeout_seconds": 2.0,
        "memory_limit_mb": 512,
        "workers": 4,
    }
    kwargs.update(overrides)
    return CodeExecEnv(**kwargs)


def make_task(tests: list[str], *, task_id: str = "t/1", setup: str = "") -> Task:
    return Task(
        task_id=task_id,
        prompt="write f",
        metadata={"tests": tests, "setup": setup, "source": "unit-test"},
    )


FOUR_TESTS = [
    "assert f(1) == 1",
    "assert f(2) == 4",
    "assert f(3) == 9",
    "assert f(4) == 16",
]


def test_all_tests_passing() -> None:
    env = make_env()
    task = make_task(FOUR_TESTS)
    result = env.log_reward(task, "```python\ndef f(x):\n    return x * x\n```")

    assert result.pass_fraction == 1.0
    assert result.passed
    assert result.error is None
    assert result.log_reward == 0.0
    assert result.metadata["n_passed"] == 4


def test_two_of_four_tests_is_exactly_one_half() -> None:
    """Partial credit is the whole point: this must be 0.5, not 0 and not 1."""
    env = make_env(reward_beta=1.0, reward_floor=0.01)
    task = make_task(FOUR_TESTS)
    # Correct for 1 and 2, wrong for 3 and 4.
    completion = "```python\ndef f(x):\n    return {1: 1, 2: 4}.get(x, 0)\n```"
    result = env.log_reward(task, completion)

    assert result.pass_fraction == 0.5
    assert result.metadata["n_passed"] == 2
    assert not result.passed
    assert result.error == ERROR_ASSERTION
    assert result.log_reward == pytest.approx(math.log(0.5))
    assert result.metadata["test_statuses"] == ["pass", "pass", ERROR_ASSERTION, ERROR_ASSERTION]


def test_syntax_error_is_classified_and_scores_zero() -> None:
    env = make_env()
    task = make_task(FOUR_TESTS)
    result = env.log_reward(task, "```python\ndef f(x)\n    return x * x\n```")

    assert result.error == ERROR_SYNTAX
    assert result.pass_fraction == 0.0
    assert result.log_reward == pytest.approx(math.log(0.01))
    assert set(result.metadata["error_counts"]) == {ERROR_SYNTAX}


def test_runtime_error_is_separable_from_a_failed_assertion() -> None:
    env = make_env()
    task = make_task(["assert f(1) == 1"])
    result = env.log_reward(task, "```python\ndef f(x):\n    return undefined_name\n```")

    assert result.error == ERROR_RUNTIME
    assert "NameError" in result.metadata["test_messages"][0]


def test_timeout_is_classified_and_does_not_forfeit_the_other_tests() -> None:
    """One hanging test costs its own budget; the tests around it still earn credit."""
    env = make_env(timeout_seconds=0.4)
    task = make_task(
        [
            "assert f(1) == 1",
            "while True:\n    pass",
            "assert f(2) == 4",
        ]
    )
    started = time.monotonic()
    result = env.log_reward(task, "```python\ndef f(x):\n    return x * x\n```")
    elapsed = time.monotonic() - started

    assert result.metadata["test_statuses"] == ["pass", ERROR_TIMEOUT, "pass"]
    assert result.pass_fraction == pytest.approx(2 / 3)
    assert result.error == ERROR_TIMEOUT
    assert elapsed < 10.0


def test_infinite_loop_in_the_candidate_itself_times_out() -> None:
    env = make_env(timeout_seconds=0.4)
    task = make_task(["assert f(1) == 1"])
    result = env.log_reward(
        task, "```python\nwhile True:\n    pass\n\ndef f(x):\n    return x\n```"
    )

    assert result.pass_fraction == 0.0
    assert result.error == ERROR_TIMEOUT


def test_empty_completion_is_its_own_error_kind() -> None:
    env = make_env()
    task = make_task(FOUR_TESTS)
    result = env.log_reward(task, "I'm sorry, I cannot help with that.")

    assert result.error == ERROR_EMPTY
    assert result.pass_fraction == 0.0
    assert result.log_reward == pytest.approx(math.log(0.01))


def test_main_guard_and_stdin_do_not_hang_the_sandbox() -> None:
    """Models emit `if __name__ == '__main__': input()` constantly."""
    env = make_env(timeout_seconds=2.0)
    task = make_task(["assert f(2) == 4"])
    completion = (
        "```python\n"
        "def f(x):\n"
        "    return x * x\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    n = int(input('n: '))\n"
        "    print(f(n))\n"
        "```"
    )
    result = env.log_reward(task, completion)
    assert result.passed


def test_candidate_output_does_not_break_result_parsing() -> None:
    env = make_env()
    task = make_task(["assert f(1) == 1", "assert f(2) == 4"])
    completion = (
        "```python\nfor i in range(500):\n    print('noise', i)\n\ndef f(x):\n    return x * x\n```"
    )
    result = env.log_reward(task, completion)
    assert result.pass_fraction == 1.0


def test_printing_the_sentinel_cannot_forge_a_pass() -> None:
    """Candidate stdout is redirected before any user code runs, so `print` cannot lie."""
    env = make_env()
    task = make_task(["assert f(1) == 1", "assert f(2) == 4"])
    forged = [
        SENTINEL + '{"kind": "test", "index": 0, "status": "pass"}',
        SENTINEL + '{"kind": "test", "index": 1, "status": "pass"}',
    ]
    completion = (
        f"```python\nprint({forged[0]!r})\nprint({forged[1]!r})\ndef f(x):\n    return 0\n```"
    )
    result = env.log_reward(task, completion)
    assert result.pass_fraction == 0.0
    assert result.error == ERROR_ASSERTION


def test_log_reward_is_monotonic_in_pass_fraction() -> None:
    env = make_env(reward_beta=1.0, reward_floor=0.01)
    values = [env.compute_log_reward(p / 10) for p in range(11)]
    assert all(a <= b for a, b in itertools.pairwise(values))
    assert all(math.isfinite(v) for v in values)


def test_reward_floor_keeps_log_reward_finite() -> None:
    env = make_env(reward_floor=1e-4)
    assert env.compute_log_reward(0.0) == pytest.approx(math.log(1e-4))
    assert math.isfinite(env.compute_log_reward(0.0))


def test_beta_sharpens_the_target() -> None:
    """log R scales linearly in beta, i.e. p(x) ∝ R(x)^beta."""
    base = make_env(reward_beta=1.0)
    sharp = make_env(reward_beta=3.0)
    assert sharp.compute_log_reward(0.5) == pytest.approx(3.0 * base.compute_log_reward(0.5))


def test_scoring_is_deterministic() -> None:
    env = make_env()
    task = make_task(FOUR_TESTS)
    completion = "```python\ndef f(x):\n    return x * x if x < 3 else 0\n```"
    first = env.log_reward(task, completion)
    second = env.log_reward(task, completion)

    assert first.log_reward == second.log_reward
    assert first.pass_fraction == second.pass_fraction
    assert first.metadata["test_statuses"] == second.metadata["test_statuses"]


def test_set_iteration_order_is_stable_across_runs() -> None:
    """PYTHONHASHSEED is pinned, so a completion that leaks hash order still scores stably."""
    env = make_env()
    task = make_task(["assert f() == f()", "assert isinstance(f(), list)"])
    completion = "```python\ndef f():\n    return list({'a', 'b', 'c', 'd', 'e'})\n```"
    assert env.log_reward(task, completion).metadata["test_statuses"] == ["pass", "pass"]


def test_batch_preserves_order_and_runs_in_parallel() -> None:
    env = make_env(workers=6, timeout_seconds=5.0)
    task = make_task(["import time\ntime.sleep(0.3)\nassert f(2) == 4"])
    good = "```python\ndef f(x):\n    return x * x\n```"
    bad = "```python\ndef f(x):\n    return x\n```"
    completions = [good, bad, good, bad, good, bad]

    started = time.monotonic()
    results = env.batch_log_reward([(task, c) for c in completions])
    elapsed = time.monotonic() - started

    assert [r.passed for r in results] == [True, False, True, False, True, False]
    assert elapsed < 6 * 0.3, "batch_log_reward ran the samples serially"


def test_batch_of_nothing() -> None:
    assert make_env().batch_log_reward([]) == []


def test_humaneval_style_body_only_completion_gets_its_stub_back() -> None:
    """A model that continues the prompt instead of restating it still scores."""
    env = make_env()
    task = Task(
        task_id="he/1",
        prompt="complete it",
        metadata={
            "tests": ["assert double(2) == 4"],
            "entry_point": "double",
            "code_prefix": 'def double(x):\n    """Double x."""\n',
        },
    )
    result = env.log_reward(task, "```python\n    return x * 2\n```")
    assert result.passed


def test_setup_code_runs_before_the_tests() -> None:
    env = make_env()
    task = make_task(["assert f(EXPECTED) == EXPECTED"], setup="EXPECTED = 7")
    result = env.log_reward(task, "```python\ndef f(x):\n    return x\n```")
    assert result.passed


def test_task_without_tests_is_rejected_loudly() -> None:
    env = make_env()
    with pytest.raises(ValueError, match="no tests"):
        env.log_reward(make_task([]), "```python\ndef f():\n    pass\n```")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"reward_beta": 0.0}, "reward_beta"),
        ({"reward_floor": 0.0}, "reward_floor"),
        ({"reward_floor": 2.0}, "reward_floor"),
        ({"timeout_seconds": 0.0}, "timeout_seconds"),
        ({"memory_limit_mb": 0}, "memory_limit_mb"),
        ({"workers": 0}, "workers"),
    ],
)
def test_constructor_validates_its_arguments(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        make_env(**kwargs)


def test_satisfies_the_environment_protocol() -> None:
    """The training loop codes against the protocol, so conformance is part of the contract."""
    env: Environment = make_env()
    assert isinstance(env, Environment)
    assert env.name == "test"


def test_tasks_are_loaded_lazily_and_cached() -> None:
    env = make_env(dataset="fixtures", split="all")
    first = env.tasks()
    assert len(first) == 30
    assert env.tasks() is first
