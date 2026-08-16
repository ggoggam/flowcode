"""Environments: task sets and the execution-based reward.

Import surface for the training loop::

    from flowcode.envs import CodeExecEnv, Environment, RewardResult, Task

:mod:`flowcode.envs.code_exec` is what Hydra instantiates from ``conf/env/*.yaml``;
everything else here is a piece of it — :mod:`flowcode.envs.datasets` supplies tasks,
:mod:`flowcode.envs.extract` turns a completion into a program, and
:mod:`flowcode.envs.sandbox` runs that program without taking the trainer down with it.
Read the sandbox module docstring before pointing any of this at code you did not write:
it is a guard against accidents, not a security boundary.
"""

from __future__ import annotations

from flowcode.envs.base import (
    ERROR_ASSERTION,
    ERROR_EMPTY,
    ERROR_HARNESS,
    ERROR_KINDS,
    ERROR_MEMORY,
    ERROR_RUNTIME,
    ERROR_SYNTAX,
    ERROR_TIMEOUT,
    Environment,
    RewardResult,
    Task,
)
from flowcode.envs.code_exec import CodeExecEnv
from flowcode.envs.datasets import load_fixture_tasks, load_tasks, split_tasks
from flowcode.envs.extract import extract_code
from flowcode.envs.sandbox import MEMORY_LIMIT_SUPPORTED, SandboxResult, run_python

__all__ = [
    "ERROR_ASSERTION",
    "ERROR_EMPTY",
    "ERROR_HARNESS",
    "ERROR_KINDS",
    "ERROR_MEMORY",
    "ERROR_RUNTIME",
    "ERROR_SYNTAX",
    "ERROR_TIMEOUT",
    "MEMORY_LIMIT_SUPPORTED",
    "CodeExecEnv",
    "Environment",
    "RewardResult",
    "SandboxResult",
    "Task",
    "extract_code",
    "load_fixture_tasks",
    "load_tasks",
    "run_python",
    "split_tasks",
]
