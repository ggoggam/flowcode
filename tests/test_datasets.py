"""Tests for the task loaders.

Everything here runs offline against the bundled fixtures. The two tests that reach the
HuggingFace hub are marked ``network`` and are deselected by default (see the ``addopts``
in pyproject.toml), so a laptop on a plane still gets a green suite.
"""

from __future__ import annotations

import builtins
from collections.abc import Sequence
from typing import Any

import pytest

from flowcode.envs.base import Task
from flowcode.envs.code_exec import CodeExecEnv
from flowcode.envs.datasets import (
    DATA_DIR,
    FIXTURE_FILES,
    iter_reference_solutions,
    load_fixture_tasks,
    load_tasks,
    mbpp_prompt,
    split_humaneval_tests,
    split_tasks,
)

EXPECTED_FIXTURE_COUNT = 30


def test_fixture_files_are_present_in_the_package() -> None:
    for filename in FIXTURE_FILES:
        assert (DATA_DIR / filename).exists(), f"{filename} must ship next to the package"


def test_fixtures_load_offline() -> None:
    tasks = load_fixture_tasks()
    assert len(tasks) == EXPECTED_FIXTURE_COUNT
    assert len({t.task_id for t in tasks}) == len(tasks)
    assert tasks == sorted(tasks, key=lambda t: t.task_id)
    for task in tasks:
        assert task.tests, f"{task.task_id} has no tests"
        assert task.prompt.strip()
        assert task.metadata["reference_solution"].strip()


def test_fixtures_include_both_shapes() -> None:
    sources = {str(t.metadata["source"]) for t in load_fixture_tasks()}
    assert sources == {"fixtures/fixtures_mbpp", "fixtures/fixtures_humaneval"}
    mbpp = [t for t in load_fixture_tasks() if "mbpp" in str(t.metadata["source"])]
    humaneval = [t for t in load_fixture_tasks() if "humaneval" in str(t.metadata["source"])]
    assert len(mbpp) == 20
    assert len(humaneval) == 10
    # HumanEval-shaped tasks must be decomposed into individually scorable assertions,
    # otherwise partial credit on them is a lie.
    assert all(len(t.tests) >= 3 for t in humaneval)


def test_every_fixture_reference_solution_passes_its_own_tests() -> None:
    """The fixtures are only useful if they are *correct*, and the only proof is running them.

    A fixture whose reference solution fails its own tests would silently cap the maximum
    achievable reward below 1.0, which in a GFlowNet means training toward a target
    distribution that has no mode where the correct answers are.
    """
    env = CodeExecEnv(name="fixtures", dataset="fixtures", split="all", timeout_seconds=5.0)
    pairs = iter_reference_solutions(load_fixture_tasks())
    assert len(pairs) == EXPECTED_FIXTURE_COUNT

    results = env.batch_log_reward(pairs)
    broken = [
        (task.task_id, result.error, result.metadata["test_statuses"])
        for (task, _), result in zip(pairs, results, strict=True)
        if not result.passed
    ]
    assert not broken, f"fixtures whose reference solution fails: {broken}"


def test_splits_are_disjoint_and_cover_everything() -> None:
    train = {t.task_id for t in load_tasks(dataset="fixtures", split="train", seed=0)}
    val = {t.task_id for t in load_tasks(dataset="fixtures", split="val", seed=0)}
    every = {t.task_id for t in load_tasks(dataset="fixtures", split="all", seed=0)}

    assert train & val == set()
    assert train | val == every
    assert len(every) == EXPECTED_FIXTURE_COUNT
    assert len(val) == round(EXPECTED_FIXTURE_COUNT * 0.2)


def test_split_is_deterministic_under_a_seed() -> None:
    first = [t.task_id for t in load_tasks(dataset="fixtures", split="train", seed=7)]
    second = [t.task_id for t in load_tasks(dataset="fixtures", split="train", seed=7)]
    assert first == second
    assert first == sorted(first), "the returned order must not depend on the shuffle"


def test_different_seeds_give_different_splits() -> None:
    by_seed = [
        tuple(t.task_id for t in load_tasks(dataset="fixtures", split="val", seed=seed))
        for seed in (0, 1, 2, 3)
    ]
    assert len(set(by_seed)) > 1


def test_test_split_aliases_val_for_fixtures() -> None:
    val = [t.task_id for t in load_tasks(dataset="fixtures", split="val", seed=3)]
    test = [t.task_id for t in load_tasks(dataset="fixtures", split="test", seed=3)]
    assert val == test


def test_limit_caps_the_task_set_deterministically() -> None:
    first = [t.task_id for t in load_tasks(dataset="fixtures", split="all", limit=5)]
    second = [t.task_id for t in load_tasks(dataset="fixtures", split="all", limit=5)]
    every = [t.task_id for t in load_tasks(dataset="fixtures", split="all")]

    assert len(first) == 5
    assert first == second
    assert first == every[:5]


def test_limit_larger_than_the_split_is_harmless() -> None:
    tasks = load_tasks(dataset="fixtures", split="all", limit=10_000)
    assert len(tasks) == EXPECTED_FIXTURE_COUNT


def test_unknown_dataset_and_split_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown dataset"):
        load_tasks(dataset="nonexistent")
    with pytest.raises(ValueError, match="unknown split"):
        load_tasks(dataset="fixtures", split="holdout")
    with pytest.raises(ValueError, match="limit"):
        load_tasks(dataset="fixtures", limit=-1)


def test_split_tasks_rejects_a_degenerate_fraction() -> None:
    tasks = load_fixture_tasks()
    with pytest.raises(ValueError, match="val_fraction"):
        split_tasks(tasks, split="train", val_fraction=0.0)


def test_mbpp_prompt_shows_the_tests() -> None:
    """Without a visible assert the function name is unguessable and the reward is noise."""
    prompt = mbpp_prompt("Add two numbers.", ["assert add(1, 2) == 3"])
    assert "Add two numbers." in prompt
    assert "assert add(1, 2) == 3" in prompt
    assert "```python" in prompt


def test_humaneval_check_is_split_into_one_snippet_per_assert() -> None:
    source = (
        "METADATA = {'author': 'nobody'}\n"
        "\n"
        "def check(candidate):\n"
        "    assert candidate(1) == 1\n"
        "    assert candidate(2) == 4\n"
        "    assert candidate(3) == 9\n"
    )
    snippets = split_humaneval_tests(source, "square")
    assert len(snippets) == 3

    namespace: dict[str, Any] = {"square": lambda x: x * x}
    for snippet in snippets:
        exec(compile(snippet, "<snippet>", "exec"), dict(namespace))

    failing: dict[str, Any] = {"square": lambda x: 0}
    outcomes = []
    for snippet in snippets:
        try:
            exec(compile(snippet, "<snippet>", "exec"), dict(failing))
            outcomes.append(True)
        except AssertionError:
            outcomes.append(False)
    assert outcomes == [False, False, False]


def test_humaneval_setup_statements_are_replicated_into_each_snippet() -> None:
    source = (
        "def check(candidate):\n"
        "    cases = [(1, 1), (2, 4)]\n"
        "    assert candidate(cases[0][0]) == cases[0][1]\n"
        "    assert candidate(cases[1][0]) == cases[1][1]\n"
    )
    snippets = split_humaneval_tests(source, "square")
    assert len(snippets) == 2
    for snippet in snippets:
        assert "cases = [(1, 1), (2, 4)]" in snippet
        exec(compile(snippet, "<snippet>", "exec"), {"square": lambda x: x * x})


def test_humaneval_loop_based_check_falls_back_to_one_snippet() -> None:
    """A check whose assertions live inside a loop cannot be decomposed statically."""
    source = (
        "def check(candidate):\n"
        "    for value in [1, 2, 3]:\n"
        "        assert candidate(value) == value * value\n"
    )
    snippets = split_humaneval_tests(source, "square")
    assert len(snippets) == 1
    exec(compile(snippets[0], "<snippet>", "exec"), {"square": lambda x: x * x})


def test_humaneval_unparseable_check_falls_back() -> None:
    snippets = split_humaneval_tests("def check(candidate:\n", "square")
    assert len(snippets) == 1


def test_fixture_tasks_are_hashable_and_frozen() -> None:
    task = load_fixture_tasks()[0]
    assert isinstance(task, Task)
    with pytest.raises(Exception):  # noqa: B017 - dataclasses raise FrozenInstanceError
        task.prompt = "mutated"  # ty: ignore[invalid-assignment]


def test_missing_datasets_extra_gives_an_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "datasets" or name.startswith("datasets."):
            raise ImportError("No module named 'datasets'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"uv sync --extra data"):
        load_tasks(dataset="mbpp", split="train", limit=1)


def _assert_hub_tasks_are_well_formed(tasks: Sequence[Task]) -> None:
    assert tasks
    for task in tasks:
        assert task.tests
        assert task.prompt.strip()


@pytest.mark.network
def test_mbpp_loads_from_the_hub() -> None:
    tasks = load_tasks(dataset="mbpp", split="train", limit=5)
    _assert_hub_tasks_are_well_formed(tasks)
    assert len(tasks) == 5


@pytest.mark.network
def test_humaneval_loads_from_the_hub() -> None:
    tasks = load_tasks(dataset="humaneval", split="test", limit=5)
    _assert_hub_tasks_are_well_formed(tasks)
    assert all(task.metadata["entry_point"] for task in tasks)
