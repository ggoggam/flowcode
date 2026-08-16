"""The reward: run the model's code against the task's hidden tests.

Reward shape
------------
``pass_fraction = passed / total`` over the task's individual tests, and

    ``log R(x) = beta * log(max(pass_fraction, reward_floor))``

Two decisions in that line matter more than they look.

**Partial credit is not a nicety.** With an all-or-nothing reward, every sample that gets
one assertion wrong lands on the floor, so the target distribution ``p(x) ∝ R(x)^beta``
becomes uniform over a huge set of near-identical failures with a thin spike on the rare
success. The gradient signal that survives is the same one PPO would get from a sparse
binary reward, minus PPO's advantage normalisation. Scoring each test separately turns
that into a graded landscape the sampler can actually climb, which is why the harness
below reports per-test results instead of a single exit status.

**beta sharpens the target.** ``p(x) ∝ R(x)^beta``: at ``beta = 1`` the policy samples in
proportion to the pass rate; as ``beta`` grows the distribution collapses toward the
argmax and the mode-covering behaviour that motivates using a GFlowNet in the first place
is thrown away. ``reward_floor`` keeps ``log R`` finite for a completion that passes
nothing — without it those samples contribute ``-inf`` and the objective degenerates.

Determinism
-----------
The reward must be a pure function of ``(task, completion)``:

* tests run in a fixed order (the order the loader produced, never a set or dict
  iteration);
* ``PYTHONHASHSEED=0`` in the child, so ``set``/``dict`` ordering inside the model's own
  code is stable across processes;
* nothing wall-clock-dependent enters the score — durations are recorded in metadata and
  never read back;
* the sandbox working directory is fresh, so nothing leaks between samples.

The one unavoidable exception is the timeout: a test that sits near the wall-clock budget
can pass on one run and time out on the next. That is inherent to executing arbitrary
code, and the mitigation is a budget generous enough that no legitimate solution is near
it. Timeouts are reported as their own error kind so their rate can be watched.
"""

from __future__ import annotations

import json
import math
import signal
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Final

from flowcode.envs.base import (
    ERROR_ASSERTION,
    ERROR_EMPTY,
    ERROR_HARNESS,
    ERROR_KINDS,
    ERROR_MEMORY,
    ERROR_RUNTIME,
    ERROR_SYNTAX,
    ERROR_TIMEOUT,
    RewardResult,
    Task,
)
from flowcode.envs.datasets import load_tasks
from flowcode.envs.extract import extract_code
from flowcode.envs.sandbox import SandboxResult, run_python

__all__ = ["SENTINEL", "CodeExecEnv", "build_harness"]

SENTINEL: Final[str] = "@@FLOWCODE@@"
"""Prefix marking a structured result line on the child's stdout.

Fixed rather than random so the harness stays reproducible and testable. The candidate's
own ``print`` output is redirected away from the real stdout before any user code runs,
so ordinary printing cannot forge a line — but a program that writes to file descriptor 1
directly still can. That is reward hacking, not an accident, and it belongs to the same
threat model the sandbox docstring declines to defend against.
"""

_ERROR_PRECEDENCE: Final[tuple[str, ...]] = (
    ERROR_HARNESS,
    ERROR_SYNTAX,
    ERROR_TIMEOUT,
    ERROR_MEMORY,
    ERROR_RUNTIME,
    ERROR_ASSERTION,
)
"""Which failure to name when a task fails several ways at once. Most-diagnostic first:
"one test asserted and one timed out" is reported as a timeout, because the timeout is
the thing to go look at."""

_MESSAGE_CHARS: Final[int] = 400
_OUTPUT_TAIL_CHARS: Final[int] = 2000

# Everything below the header is static; the header is generated per call. Keeping them
# separate avoids brace-escaping a large chunk of Python through str.format.
_HARNESS_BODY: Final[str] = '''
import io
import json
import signal
import sys
import traceback

_REAL_STDOUT = sys.stdout


class _Sink(io.TextIOBase):
    """Swallow candidate output past a small cap so it cannot flood the capture file."""

    def __init__(self, limit=65536):
        self._written = 0
        self._limit = limit

    def write(self, s):
        n = len(s)
        self._written += n
        return n

    def writable(self):
        return True

    def flush(self):
        return None


class _TestTimeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise _TestTimeout("exceeded per-test time budget")


def _emit(payload):
    _REAL_STDOUT.write(_SENTINEL + json.dumps(payload, default=str) + "\\n")
    _REAL_STDOUT.flush()


def _short(text):
    text = str(text)
    return text if len(text) <= 400 else text[:400] + "..."


def _classify(exc):
    if isinstance(exc, _TestTimeout):
        return "timeout", "timeout after %.3fs" % _PER_TEST_TIMEOUT
    if isinstance(exc, AssertionError):
        return "assertion", _short(str(exc) or "assertion failed")
    if isinstance(exc, MemoryError):
        return "memory", "MemoryError"
    if isinstance(exc, SyntaxError):
        return "syntax", _short("%s: %s" % (type(exc).__name__, exc))
    if isinstance(exc, SystemExit):
        return "runtime", "SystemExit(%s)" % (exc.code,)
    if isinstance(exc, RecursionError):
        return "runtime", "RecursionError"
    return "runtime", _short("%s: %s" % (type(exc).__name__, exc))


def _guarded(source, filename, namespace):
    """Run one snippet under the per-test alarm. Returns (status, message)."""
    try:
        compiled = compile(source, filename, "exec")
    except SyntaxError as exc:
        return _classify(exc)
    signal.setitimer(signal.ITIMER_REAL, _PER_TEST_TIMEOUT)
    try:
        exec(compiled, namespace)
    except BaseException as exc:  # noqa: BLE001 - the candidate may raise anything
        return _classify(exc)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
    return "pass", ""


def main():
    signal.signal(signal.SIGALRM, _on_alarm)
    sys.stdout = _Sink()
    sys.stderr = _Sink()
    sys.setrecursionlimit(4000)

    _emit({"kind": "start", "n_tests": len(_TESTS)})

    namespace = {"__name__": "flowcode_candidate", "__file__": "<candidate>"}
    status, message = _guarded(_CANDIDATE, "<candidate>", namespace)
    if status != "pass":
        _emit({"kind": "candidate", "status": status, "message": message})
        return
    if _SETUP.strip():
        status, message = _guarded(_SETUP, "<setup>", namespace)
        if status != "pass":
            _emit({"kind": "setup", "status": status, "message": message})
            return

    for index, source in enumerate(_TESTS):
        status, message = _guarded(source, "<test%d>" % index, namespace)
        _emit({"kind": "test", "index": index, "status": status, "message": message})

    _emit({"kind": "done"})


try:
    main()
except BaseException:  # noqa: BLE001 - never let the harness itself go unreported
    sys.stdout = _REAL_STDOUT
    _emit({"kind": "harness", "status": "runtime", "message": traceback.format_exc()[-400:]})
'''


def build_harness(code: str, tests: Sequence[str], setup: str, per_test_timeout: float) -> str:
    """Assemble the script that runs ``code`` against ``tests`` and reports per test.

    The candidate is executed with ``__name__ = "flowcode_candidate"`` rather than
    ``"__main__"``, so a ``if __name__ == "__main__":`` block — which models emit
    constantly, often containing an ``input()`` call — does not run and does not hang the
    sandbox waiting on stdin.

    Each test is guarded by ``signal.setitimer``, so one hanging test costs its own
    budget instead of the whole sample's: the tests after it still run and still earn
    partial credit. The alarm only interrupts between bytecodes, so a completion stuck
    inside a single long C call (``sum(range(10 ** 12))``) escapes it — the sandbox's
    wall-clock kill and ``RLIMIT_CPU`` are the backstop, and everything unreported is
    then scored as a timeout.

    Args:
        code: The extracted candidate source.
        tests: Test snippets, executed in this exact order in a shared namespace.
        setup: Code run after the candidate and before the tests.
        per_test_timeout: Seconds allowed per snippet.

    Returns:
        A self-contained Python program.
    """
    header = "\n".join(
        [
            f"_SENTINEL = {SENTINEL!r}",
            f"_PER_TEST_TIMEOUT = {float(per_test_timeout)!r}",
            f"_CANDIDATE = {code!r}",
            f"_SETUP = {setup!r}",
            f"_TESTS = {list(tests)!r}",
        ]
    )
    return header + "\n" + _HARNESS_BODY


def _parse_reports(stdout: str) -> list[dict[str, Any]]:
    """Pull the sentinel-prefixed JSON lines out of the child's stdout."""
    reports: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        if not line.startswith(SENTINEL):
            continue
        try:
            payload = json.loads(line[len(SENTINEL) :])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            reports.append(payload)
    return reports


def _signal_error_kind(signum: int) -> str:
    """Best-effort classification of a child killed by a signal."""
    if signum in (signal.SIGXCPU, signal.SIGALRM):
        return ERROR_TIMEOUT
    if signum == signal.SIGKILL:
        # Under a memory limit this is usually the kernel OOM killer; there is no more
        # specific evidence available to the parent.
        return ERROR_MEMORY
    return ERROR_RUNTIME


class CodeExecEnv:
    """Execution-based reward over a code dataset.

    Instantiated by Hydra from ``conf/env/*.yaml``; the keyword names here are the config
    keys and must not drift from them.

    Args:
        name: Environment name, used in logs and metric keys.
        dataset: ``"mbpp"``, ``"humaneval"`` or ``"fixtures"``.
        split: Split name. See :func:`flowcode.envs.datasets.load_tasks`.
        limit: Cap on the number of tasks, applied after splitting. ``None`` for all.
        seed: Seed for the deterministic train/val split.
        reward_beta: Exponent on the reward, ``p(x) ∝ R(x)^beta``.
        reward_floor: Lower clamp on ``pass_fraction`` before the log. Must be positive.
        timeout_seconds: Per-test wall-clock budget in the sandbox.
        memory_limit_mb: ``RLIMIT_AS`` for the child. Ignored on macOS; see
            :mod:`flowcode.envs.sandbox`.
        workers: Concurrency of :meth:`batch_log_reward`.

    Raises:
        ValueError: If any numeric parameter is out of range.
    """

    def __init__(
        self,
        name: str = "code_exec",
        dataset: str = "fixtures",
        split: str = "train",
        limit: int | None = None,
        seed: int = 0,
        reward_beta: float = 1.0,
        reward_floor: float = 0.01,
        timeout_seconds: float = 10.0,
        memory_limit_mb: int = 1024,
        workers: int = 8,
    ) -> None:
        if reward_beta <= 0:
            raise ValueError(f"reward_beta must be positive, got {reward_beta}")
        if not 0.0 < reward_floor <= 1.0:
            raise ValueError(
                f"reward_floor must be in (0, 1], got {reward_floor}; it exists to keep "
                "log R finite, so zero defeats the purpose"
            )
        if timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be positive, got {timeout_seconds}")
        if memory_limit_mb <= 0:
            raise ValueError(f"memory_limit_mb must be positive, got {memory_limit_mb}")
        if workers < 1:
            raise ValueError(f"workers must be at least 1, got {workers}")

        self.name = name
        self.dataset = dataset
        self.split = split
        self.limit = limit
        self.seed = seed
        self.reward_beta = float(reward_beta)
        self.reward_floor = float(reward_floor)
        self.timeout_seconds = float(timeout_seconds)
        self.memory_limit_mb = int(memory_limit_mb)
        self.workers = int(workers)
        self._tasks: tuple[Task, ...] | None = None

    def tasks(self) -> Sequence[Task]:
        """The task set, loaded on first use and cached.

        Loading is deferred out of ``__init__`` so that constructing the environment —
        which Hydra does while composing the config — never touches the network or the
        HuggingFace cache.
        """
        if self._tasks is None:
            self._tasks = tuple(
                load_tasks(dataset=self.dataset, split=self.split, limit=self.limit, seed=self.seed)
            )
        return self._tasks

    def compute_log_reward(self, pass_fraction: float) -> float:
        """``beta * log(max(pass_fraction, floor))``.

        Monotonically increasing in ``pass_fraction`` and finite everywhere, including at
        zero. Exposed separately so tests and the cost estimator can reason about the
        reward curve without executing anything.
        """
        return self.reward_beta * math.log(max(pass_fraction, self.reward_floor))

    def log_reward(self, task: Task, completion: str) -> RewardResult:
        """Score one completion against one task's hidden tests."""
        code = extract_code(completion)
        tests = task.tests
        if not code.strip():
            return RewardResult(
                log_reward=self.compute_log_reward(0.0),
                pass_fraction=0.0,
                passed=False,
                error=ERROR_EMPTY,
                metadata={
                    "task_id": task.task_id,
                    "n_tests": len(tests),
                    "n_passed": 0,
                    "error_counts": {ERROR_EMPTY: 1},
                    "completion_chars": len(completion),
                },
            )

        candidate = self._assemble(task, code)
        if not tests:
            raise ValueError(
                f"Task {task.task_id!r} has no tests; a task with nothing to check cannot "
                "produce a reward signal and must be filtered out by the loader"
            )

        harness = build_harness(candidate, tests, task.setup, self.timeout_seconds)
        # The wall-clock budget covers every test plus the candidate's own module-level
        # code. The per-test alarm normally fires long before this; it is the backstop for
        # code the alarm cannot interrupt.
        wall_timeout = self.timeout_seconds * (len(tests) + 1) + 2.0
        result = run_python(
            harness,
            timeout_seconds=wall_timeout,
            memory_limit_mb=self.memory_limit_mb,
            extra_env={"PYTHONHASHSEED": "0"},
        )
        return self._score(task, code, tests, result)

    def batch_log_reward(self, pairs: Sequence[tuple[Task, str]]) -> list[RewardResult]:
        """Score a batch in parallel, preserving input order.

        This is the loop's throughput floor: ``groups * group_size`` subprocesses per
        optimiser step, each paying interpreter startup. Threads rather than processes
        because every worker spends its life blocked in ``waitpid`` — the GIL is released
        throughout, and threads avoid pickling the task set into a pool of children.
        """
        if not pairs:
            return []
        if self.workers == 1 or len(pairs) == 1:
            return [self.log_reward(task, completion) for task, completion in pairs]
        with ThreadPoolExecutor(
            max_workers=min(self.workers, len(pairs)), thread_name_prefix=f"{self.name}-reward"
        ) as pool:
            return list(pool.map(lambda pair: self.log_reward(*pair), pairs))

    @staticmethod
    def _assemble(task: Task, code: str) -> str:
        """Prepend the task's stub when the model wrote only a function body.

        HumanEval prompts end mid-definition, and a model that continues rather than
        restates produces an indented block that cannot stand alone. Prepending the stub
        is only correct in exactly that case, so the trigger is narrow and syntactic:
        there is a stub, the tests name an entry point, the extracted code never defines
        it, and the code is indented.
        """
        stub = task.metadata.get("code_prefix")
        entry_point = task.entry_point
        if not stub or not entry_point:
            return code
        if f"def {entry_point}" in code:
            return code
        first = next((ln for ln in code.splitlines() if ln.strip()), "")
        if not first.startswith((" ", "\t")):
            return code
        return str(stub).rstrip("\n") + "\n" + code

    def _score(
        self, task: Task, code: str, tests: Sequence[str], result: SandboxResult
    ) -> RewardResult:
        """Turn the sandbox's stdout into a :class:`RewardResult`."""
        reports = _parse_reports(result.stdout)
        by_index: dict[int, dict[str, Any]] = {}
        fatal: dict[str, Any] | None = None
        for report in reports:
            kind = report.get("kind")
            if kind == "test":
                index = report.get("index")
                if isinstance(index, int):
                    by_index[index] = report
            elif kind in {"candidate", "setup", "harness"} and fatal is None:
                fatal = report

        statuses: list[str] = []
        messages: list[str] = []
        for position in range(len(tests)):
            reported = by_index.get(position)
            if reported is not None:
                statuses.append(str(reported.get("status", ERROR_RUNTIME)))
                messages.append(str(reported.get("message", ""))[:_MESSAGE_CHARS])
                continue
            # Unreported: the child died before reaching this test.
            if fatal is not None:
                statuses.append(str(fatal.get("status", ERROR_RUNTIME)))
                messages.append(str(fatal.get("message", ""))[:_MESSAGE_CHARS])
            elif result.timed_out:
                statuses.append(ERROR_TIMEOUT)
                messages.append(f"sandbox wall-clock timeout after {result.duration_seconds:.1f}s")
            elif result.killed_by_signal is not None:
                signum = result.killed_by_signal
                statuses.append(_signal_error_kind(signum))
                messages.append(f"child killed by signal {signum}")
            elif not reports:
                statuses.append(ERROR_HARNESS)
                messages.append(
                    result.stderr[-_MESSAGE_CHARS:] or "no output from the scoring harness"
                )
            else:
                statuses.append(ERROR_RUNTIME)
                messages.append("test did not report a result")

        n_passed = sum(1 for status in statuses if status == "pass")
        pass_fraction = n_passed / len(tests)
        error_counts: dict[str, int] = {}
        for status in statuses:
            if status == "pass":
                continue
            key = status if status in ERROR_KINDS else ERROR_RUNTIME
            error_counts[key] = error_counts.get(key, 0) + 1

        error: str | None = None
        if error_counts:
            error = next(kind for kind in _ERROR_PRECEDENCE if kind in error_counts)

        return RewardResult(
            log_reward=self.compute_log_reward(pass_fraction),
            pass_fraction=pass_fraction,
            passed=error is None,
            error=error,
            metadata={
                "task_id": task.task_id,
                "n_tests": len(tests),
                "n_passed": n_passed,
                "test_statuses": statuses,
                "test_messages": messages,
                "error_counts": error_counts,
                "timed_out": result.timed_out,
                "returncode": result.returncode,
                "duration_seconds": result.duration_seconds,
                "code_chars": len(code),
                "stdout_tail": result.stdout[-_OUTPUT_TAIL_CHARS:],
                "stderr_tail": result.stderr[-_OUTPUT_TAIL_CHARS:],
                "output_truncated": result.stdout_truncated or result.stderr_truncated,
                "limits_applied": list(result.limits_applied),
            },
        )
