"""The environment contract the training loop codes against.

An environment is two things and nothing more: a set of :class:`Task` objects to
condition on, and a map from ``(task, completion)`` to a :class:`RewardResult` carrying
``log R(x)``.

Why log space, and why determinism matters here more than in PPO: a GFlowNet is trained
to sample proportionally to the reward, ``p(x) ∝ R(x)``. The reward is therefore not a
scalar advantage signal that gets averaged away — it *is* the target distribution. A
reward that returns 0.75 for one sample and 0.5 for the same sample a second time does
not merely add gradient variance, it makes the thing being fit ill-defined. Everything
downstream of this module (sandbox limits, per-test scoring, fixed hash seed, sorted test
order) exists to keep ``log_reward`` a pure function of ``(task, completion)``.

``batch_log_reward`` is separate from ``log_reward`` because it is the throughput
bottleneck of the whole loop: one subprocess per sampled completion, ``groups *
group_size`` of them per optimiser step. Implementations must actually run them
concurrently.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "ERROR_ASSERTION",
    "ERROR_EMPTY",
    "ERROR_HARNESS",
    "ERROR_KINDS",
    "ERROR_MEMORY",
    "ERROR_RUNTIME",
    "ERROR_SYNTAX",
    "ERROR_TIMEOUT",
    "Environment",
    "RewardResult",
    "Task",
]

ERROR_EMPTY = "empty"
"""No code could be extracted from the completion at all."""

ERROR_SYNTAX = "syntax"
"""The extracted code does not parse."""

ERROR_TIMEOUT = "timeout"
"""A test (or the candidate's module-level code) exceeded its wall-clock budget."""

ERROR_MEMORY = "memory"
"""The candidate hit ``MemoryError`` under the sandbox's address-space limit."""

ERROR_RUNTIME = "runtime"
"""A non-assertion exception escaped: NameError, TypeError, ZeroDivisionError, ..."""

ERROR_ASSERTION = "assertion"
"""The code ran and produced the wrong answer. The only *interesting* failure."""

ERROR_HARNESS = "harness"
"""The scoring harness itself produced nothing parseable. A bug in flowcode, not the model."""

ERROR_KINDS: tuple[str, ...] = (
    ERROR_EMPTY,
    ERROR_SYNTAX,
    ERROR_TIMEOUT,
    ERROR_MEMORY,
    ERROR_RUNTIME,
    ERROR_ASSERTION,
    ERROR_HARNESS,
)
"""Every value ``RewardResult.error`` can take, in rough precedence order.

Kept as a closed set so training metrics can bucket failures without string-matching on
free-form messages: "38% syntax, 4% timeout, 51% assertion" is a diagnosis, "51% of
samples failed" is not.
"""


@dataclass(frozen=True)
class Task:
    """One prompt to condition on, plus whatever the reward needs to score it.

    Args:
        task_id: Stable identifier. Objectives with a per-prompt ``log Z`` group
            trajectories by this, so it must be identical across every completion
            sampled for the same prompt, and stable across processes and runs.
        prompt: The user-message content shown to the model. Contains no hidden tests.
        metadata: Everything the reward needs and the model must not see — notably
            ``tests`` (the hidden test snippets), ``setup`` (code run before them), and
            ``entry_point``. Also carries provenance (``source``, ``split``).
    """

    task_id: str
    prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def tests(self) -> list[str]:
        """The hidden test snippets, each independently scorable. May be empty."""
        raw = self.metadata.get("tests", [])
        if not isinstance(raw, list):
            raise TypeError(
                f"Task {self.task_id!r}: metadata['tests'] must be a list of source "
                f"strings, got {type(raw).__name__}"
            )
        return [str(t) for t in raw]

    @property
    def setup(self) -> str:
        """Code executed once after the candidate and before the tests (imports, data)."""
        return str(self.metadata.get("setup", "") or "")

    @property
    def entry_point(self) -> str | None:
        """Name of the function the tests call, when the dataset names one."""
        value = self.metadata.get("entry_point")
        return None if value is None else str(value)


@dataclass(frozen=True)
class RewardResult:
    """The outcome of scoring one completion.

    Args:
        log_reward: ``beta * log(max(pass_fraction, reward_floor))``. Always finite —
            the floor is what keeps a totally broken sample from contributing ``-inf``
            and taking the objective's gradient with it.
        pass_fraction: Fraction of the task's tests that passed, in ``[0, 1]``.
        passed: True iff every test passed. Not derivable from ``pass_fraction == 1.0``
            for a task with no tests, which is why it is stored.
        error: One of :data:`ERROR_KINDS`, or ``None`` when everything passed.
        metadata: Per-test statuses, counts by error kind, captured output tails,
            timings. Diagnostics only; never read by the objectives.
    """

    log_reward: float
    pass_fraction: float
    passed: bool
    error: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.pass_fraction <= 1.0:
            raise ValueError(f"pass_fraction must be in [0, 1], got {self.pass_fraction}")
        if self.error is not None and self.error not in ERROR_KINDS:
            raise ValueError(
                f"error must be one of {ERROR_KINDS} or None, got {self.error!r}; "
                "free-form error strings belong in metadata"
            )
        if self.passed and self.error is not None:
            raise ValueError(f"passed=True but error={self.error!r}")


@runtime_checkable
class Environment(Protocol):
    """What the training loop needs from a task source.

    Deliberately tiny: the loop asks for tasks, samples completions elsewhere, and hands
    ``(task, completion)`` pairs back for scoring.
    """

    name: str

    def tasks(self) -> Sequence[Task]:
        """The task set, in a fixed order. Repeated calls return the same sequence."""
        ...

    def log_reward(self, task: Task, completion: str) -> RewardResult:
        """Score one completion. Pure: same inputs, same output."""
        ...

    def batch_log_reward(self, pairs: Sequence[tuple[Task, str]]) -> list[RewardResult]:
        """Score a batch concurrently, returning results in the input order."""
        ...
