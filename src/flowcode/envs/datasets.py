"""Task loaders: MBPP, HumanEval, and the bundled offline fixtures.

Three sources, one output type (:class:`flowcode.envs.base.Task`), so the reward and the
training loop never learn which benchmark they are looking at.

The ``datasets`` library is an optional extra (``uv sync --extra data``). It is imported
lazily *inside* the loader that needs it, never at module import, so that the unit tests
— and anyone smoke-testing the loop on fixtures — do not pay for the HuggingFace stack or
need a network. The bundled fixtures exist for the same reason: ``dataset: fixtures``
gives a complete, runnable task set with zero external dependencies, which is what makes
it possible to test the reward path offline.

Splitting is seeded and deterministic. A GFlowNet run compares train and held-out pass
rates across many hours; a split that shifts between processes silently invalidates that
comparison, so the shuffle is driven by an explicit ``random.Random(seed)`` and the
selected tasks come back in sorted order regardless of how the shuffle landed.

Partial credit needs individually scorable tests, which the two upstream formats provide
very differently. MBPP hands over ``test_list``, already a list of asserts — one test
each, nothing to do. HumanEval hands over a single ``check(candidate)`` function
containing every assertion, which all-or-nothing scoring would collapse into one bit;
:func:`split_humaneval_tests` takes it apart with :mod:`ast` so each assertion is scored
on its own.
"""

from __future__ import annotations

import ast
import copy
import json
import random
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Final

from flowcode.envs.base import Task

__all__ = [
    "DATA_DIR",
    "FIXTURE_FILES",
    "humaneval_prompt",
    "iter_reference_solutions",
    "load_fixture_tasks",
    "load_tasks",
    "mbpp_prompt",
    "split_humaneval_tests",
    "split_tasks",
]

DATA_DIR: Final[Path] = Path(__file__).resolve().parent.parent / "data"
"""Where the offline fixtures live. Ships inside the wheel next to the package."""

FIXTURE_FILES: Final[tuple[str, ...]] = ("fixtures_mbpp.jsonl", "fixtures_humaneval.jsonl")
"""Fixture shards, in load order. The first is MBPP-shaped, the second HumanEval-shaped."""

DEFAULT_VAL_FRACTION: Final[float] = 0.2
"""Held-out share when a split has to be derived rather than taken from the hub."""

_MBPP_HUB_ID: Final[str] = "google-research-datasets/mbpp"
_HUMANEVAL_HUB_ID: Final[str] = "openai_humaneval"
_MBPP_NATIVE_SPLITS: Final[frozenset[str]] = frozenset({"train", "test", "validation", "prompt"})
_DERIVED_SPLITS: Final[frozenset[str]] = frozenset({"train", "val", "validation", "test", "all"})

_MISSING_DATASETS_HINT: Final[str] = (
    "the `datasets` package is required to load {name} from the HuggingFace hub but is not "
    "installed. Install the optional extra with `uv sync --extra data`, or use "
    "`env.dataset=fixtures` to run against the bundled offline task set."
)


def mbpp_prompt(text: str, tests: Sequence[str]) -> str:
    """Render an MBPP task as a user message.

    The tests go into the prompt on purpose. An MBPP description ("Write a function to
    find the shared elements from two lists") does not say what the function is called or
    what shape it returns, so without a visible assert the task is unanswerable and the
    reward measures name-guessing. Showing the asserts is the standard MBPP protocol and
    it is what makes the pass rate mean anything.
    """
    rendered = "\n".join(tests)
    return (
        "You are an expert Python programmer. Write a Python function for the following "
        "task.\n\n"
        f"Task: {text.strip()}\n\n"
        "Your code must pass these tests:\n\n"
        f"{rendered}\n\n"
        "Reply with the complete function in a single ```python code block."
    )


def humaneval_prompt(stub: str) -> str:
    """Render a HumanEval stub (signature + docstring) as a user message."""
    return (
        "Complete the following Python function. Reply with the complete function, "
        "including its signature, in a single ```python code block.\n\n"
        f"```python\n{stub.rstrip()}\n```"
    )


def split_humaneval_tests(test_source: str, entry_point: str) -> list[str]:
    """Break a HumanEval ``check`` function into one snippet per assertion.

    Each returned snippet is a standalone program: any module-level setup from the
    original (``METADATA = ...``), a one-assertion ``check``, and the call
    ``check(<entry_point>)``. Non-assert statements inside the original body (fixture
    lists, helper definitions) are replicated into every snippet, since a later assertion
    may depend on them.

    Args:
        test_source: The dataset's ``test`` field.
        entry_point: The function the tests exercise.

    Returns:
        One snippet per top-level assertion, in source order. Falls back to a single
        snippet holding the whole check — coarse but correct — when the source does not
        parse, has no ``check`` function, or hides its assertions inside a loop, which is
        the one HumanEval shape that cannot be decomposed statically.
    """
    whole = f"{test_source.rstrip()}\ncheck({entry_point})\n"
    try:
        module = ast.parse(test_source)
    except SyntaxError:
        return [whole]

    check_fn: ast.FunctionDef | None = None
    prelude: list[ast.stmt] = []
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == "check" and check_fn is None:
            check_fn = node
        else:
            prelude.append(node)
    if check_fn is None:
        return [whole]

    asserts = [stmt for stmt in check_fn.body if isinstance(stmt, ast.Assert)]
    others = [stmt for stmt in check_fn.body if not isinstance(stmt, ast.Assert)]
    if len(asserts) < 2:
        return [whole]

    call = ast.parse(f"check({entry_point})").body
    snippets: list[str] = []
    for assertion in asserts:
        # Copy the original node rather than constructing a FunctionDef: its field list
        # grew a `type_params` entry in 3.12, and a copy is version-agnostic.
        single = copy.copy(check_fn)
        single.body = [*others, assertion]
        tree = ast.Module(body=[*prelude, single, *call], type_ignores=[])
        ast.fix_missing_locations(tree)
        snippets.append(ast.unparse(tree) + "\n")
    return snippets


def _fixture_rows(path: Path) -> list[dict[str, Any]]:
    """Read one JSONL fixture shard."""
    if not path.exists():
        raise FileNotFoundError(
            f"fixture file {path} is missing; the offline task set ships with the package "
            "and should sit next to flowcode/envs"
        )
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} is not valid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{lineno} must be a JSON object, got {type(row).__name__}")
            rows.append(row)
    return rows


def _task_from_mbpp_row(row: dict[str, Any], *, source: str) -> Task:
    """Build a task from an MBPP-shaped row (``text``, ``code``, ``test_list``)."""
    tests = [str(t) for t in row.get("test_list", [])]
    if not tests:
        raise ValueError(f"MBPP-shaped row {row.get('task_id')!r} has an empty test_list")
    setup = str(row.get("test_setup_code", "") or "")
    return Task(
        task_id=str(row["task_id"]),
        prompt=mbpp_prompt(str(row["text"]), tests),
        metadata={
            "source": source,
            "tests": tests,
            "setup": setup,
            "entry_point": row.get("entry_point"),
            "reference_solution": str(row.get("code", "")),
        },
    )


def _task_from_humaneval_row(row: dict[str, Any], *, source: str) -> Task:
    """Build a task from a HumanEval-shaped row."""
    entry_point = str(row["entry_point"])
    stub = str(row["prompt"])
    tests = split_humaneval_tests(str(row["test"]), entry_point)
    return Task(
        task_id=str(row["task_id"]),
        prompt=humaneval_prompt(stub),
        metadata={
            "source": source,
            "tests": tests,
            "setup": "",
            "entry_point": entry_point,
            "code_prefix": stub,
            "reference_solution": stub + str(row.get("canonical_solution", "")),
        },
    )


def load_fixture_tasks() -> list[Task]:
    """Load the bundled offline tasks.

    Roughly twenty MBPP-shaped and ten HumanEval-shaped problems, written for this
    repository rather than copied from either benchmark — the upstream rows carry their
    own licences and are not redistributable here. They are deliberately easy; their job
    is to exercise the reward path, not to measure a model.

    Returns:
        Every fixture task, sorted by ``task_id``.
    """
    tasks: list[Task] = []
    for filename in FIXTURE_FILES:
        path = DATA_DIR / filename
        source = f"fixtures/{path.stem}"
        for row in _fixture_rows(path):
            if "test_list" in row:
                tasks.append(_task_from_mbpp_row(row, source=source))
            else:
                tasks.append(_task_from_humaneval_row(row, source=source))
    return sorted(tasks, key=lambda t: t.task_id)


def split_tasks(
    tasks: Sequence[Task],
    *,
    split: str,
    seed: int = 0,
    val_fraction: float = DEFAULT_VAL_FRACTION,
) -> list[Task]:
    """Deterministically carve a train/val split out of a flat task list.

    The shuffle is seeded and local (``random.Random(seed)``, never the global RNG, which
    the training loop also uses), and the chosen tasks are returned in ``task_id`` order
    so the result does not depend on how the shuffle permuted them.

    Args:
        tasks: The full set.
        split: ``"train"``, ``"val"`` / ``"validation"``, ``"test"`` (an alias of val for
            sources with no third split), or ``"all"``.
        seed: Shuffle seed.
        val_fraction: Held-out share, in ``(0, 1)``.

    Returns:
        The selected tasks, sorted by ``task_id``.

    Raises:
        ValueError: On an unknown split name or an out-of-range fraction.
    """
    normalised = split.strip().lower()
    if normalised not in _DERIVED_SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {sorted(_DERIVED_SPLITS)}")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    ordered = sorted(tasks, key=lambda t: t.task_id)
    if normalised == "all":
        return ordered

    shuffled = list(ordered)
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_fraction)) if shuffled else 0
    held_out = shuffled[:n_val]
    train = shuffled[n_val:]
    selected = train if normalised == "train" else held_out
    return sorted(selected, key=lambda t: t.task_id)


def _require_datasets(name: str) -> Any:
    """Import ``datasets`` lazily with an actionable error if the extra is missing."""
    try:
        import datasets as hf_datasets
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(_MISSING_DATASETS_HINT.format(name=name)) from exc
    return hf_datasets


def _load_mbpp(split: str, seed: int) -> list[Task]:
    """Load MBPP from the hub. Requires network on first call; cached after."""
    hf_datasets = _require_datasets("MBPP")
    normalised = "validation" if split.strip().lower() == "val" else split.strip().lower()
    hub_split = normalised if normalised in _MBPP_NATIVE_SPLITS else "train"
    raw = hf_datasets.load_dataset(_MBPP_HUB_ID, "full", split=hub_split)
    tasks = [_task_from_mbpp_row(dict(row), source="mbpp") for row in raw]
    if normalised in _MBPP_NATIVE_SPLITS:
        return sorted(tasks, key=lambda t: t.task_id)
    return split_tasks(tasks, split=normalised, seed=seed)


def _load_humaneval(split: str, seed: int) -> list[Task]:
    """Load HumanEval from the hub. Only a ``test`` split exists upstream."""
    hf_datasets = _require_datasets("HumanEval")
    raw = hf_datasets.load_dataset(_HUMANEVAL_HUB_ID, split="test")
    tasks = [_task_from_humaneval_row(dict(row), source="humaneval") for row in raw]
    normalised = split.strip().lower()
    if normalised in {"test", "all"}:
        return sorted(tasks, key=lambda t: t.task_id)
    return split_tasks(tasks, split=normalised, seed=seed)


def load_tasks(
    *,
    dataset: str,
    split: str = "train",
    limit: int | None = None,
    seed: int = 0,
) -> list[Task]:
    """Load a task set by name.

    Args:
        dataset: ``"mbpp"``, ``"humaneval"`` or ``"fixtures"``.
        split: Split name. MBPP passes native names (``train``/``test``/``validation``/
            ``prompt``) straight through to the hub; HumanEval's only native split is
            ``test``, and anything else is derived with a seeded shuffle; fixtures are
            always derived.
        limit: Keep at most this many tasks, taken from the front of the sorted split so
            the cap is itself deterministic. ``None`` keeps everything.
        seed: Seed for derived splits.

    Returns:
        Tasks in ``task_id`` order.

    Raises:
        ValueError: On an unknown dataset name or a negative limit.
        ImportError: If a hub dataset is requested without the ``data`` extra installed.
    """
    if limit is not None and limit < 0:
        raise ValueError(f"limit must be non-negative or None, got {limit}")

    key = dataset.strip().lower()
    if key == "fixtures":
        tasks = split_tasks(load_fixture_tasks(), split=split, seed=seed)
    elif key == "mbpp":
        tasks = _load_mbpp(split, seed)
    elif key == "humaneval":
        tasks = _load_humaneval(split, seed)
    else:
        raise ValueError(f"unknown dataset {dataset!r}; expected 'mbpp', 'humaneval' or 'fixtures'")

    if not tasks:
        raise ValueError(f"dataset {dataset!r} split {split!r} produced no tasks")
    return tasks if limit is None else tasks[:limit]


def iter_reference_solutions(tasks: Iterable[Task]) -> list[tuple[Task, str]]:
    """Pair each task with its reference solution, skipping tasks that have none.

    Used by the fixture self-check: a fixture whose own reference solution does not pass
    its own tests is a broken reward signal, and the only way to know is to run it.
    """
    pairs: list[tuple[Task, str]] = []
    for task in tasks:
        reference = str(task.metadata.get("reference_solution", "") or "")
        if reference.strip():
            pairs.append((task, reference))
    return pairs
