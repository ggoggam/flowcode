"""Sampling and scoring decoupled from the training step.

The problem
-----------
:meth:`flowcode.train.Trainer.step` runs sample → score → train in series. With the
shipped defaults that is 64 sequences per step (``groups_per_step: 8`` times
``group_size: 8``), which is far too few concurrent requests to keep an accelerator busy
during decode, followed by 64 sandboxed test executions through ``env.workers``, during
which the accelerator does nothing whatsoever. Both halves idle waiting for the other.

Why this is allowed here and not in PPO
---------------------------------------
Decoupling means training on trajectories drawn from a policy that is several updates
behind. For a ratio-clipped policy-gradient method that is a correctness problem needing
importance correction. For a GFlowNet it is not: the balance conditions the objectives
regress on are properties of the policy and flow *functions*, not expectations under the
sampling distribution, so a stale trajectory is an equally valid constraint —
:mod:`flowcode.objectives.base` says so at length, and
:class:`~flowcode.replay.ReplayBuffer` already exploits it. This module takes the same
licence one step further: instead of a buffer topping up an otherwise on-policy batch, the
sampler simply never stops.

The shape
---------
``concurrency`` workers each hold **one task's group** in flight at a time, so
``concurrency * group_size`` sequences are outstanding rather than the 64 a step happens to
want. Completed groups land in a bounded queue; the trainer pulls whole groups from it and
never awaits a rollout. The bound is what stops a fast sampler from running arbitrarily far
ahead of a slow trainer, and it is expressed in groups because a group is the unit
:class:`~flowcode.objectives.vargrad.VarGrad` and every other group-baseline estimator
needs intact.

Two knobs interact and it is not obvious
----------------------------------------
Each worker's scoring pass fans out over ``env.workers`` subprocesses, so the peak
subprocess count is ``concurrency * env.workers``, not ``env.workers``. Raising producer
concurrency without lowering ``env.workers`` is how a host ends up with hundreds of
concurrent test executions. :func:`validate_producer_config` warns about it rather than
letting the machine find out.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from flowcode.envs.base import Task
from flowcode.render import ChatTokenizer
from flowcode.rollout import RolloutStats, rollout
from flowcode.types import Trajectory

__all__ = [
    "ProducedGroup",
    "ProducerStats",
    "TrajectoryProducer",
    "validate_producer_config",
]

logger = logging.getLogger(__name__)

MAX_REASONABLE_SUBPROCESSES = 64
"""Above this many concurrent sandboxed executions, warn. Each is a forked interpreter
running untrusted generated code; the host, not the trainer, is what falls over."""


@dataclass(frozen=True)
class ProducedGroup:
    """One task's completions, kept together.

    Args:
        task_id: The task these completions answer.
        trajectories: The group. Shorter than ``group_size`` when degenerate samples were
            dropped, and possibly empty, in which case the group is discarded rather than
            queued.
        policy_version: How many sampler syncs had happened when this was drawn. The
            trainer's current version minus this is the group's staleness.
    """

    task_id: str
    trajectories: list[Trajectory]
    policy_version: int


@dataclass
class ProducerStats:
    """What the producer has done, cumulatively.

    Args:
        groups_produced: Groups pushed onto the queue.
        groups_dropped_stale: Groups discarded at drain time for being too far behind.
        groups_dropped_empty: Rollouts that yielded no usable trajectory at all.
        trajectories_produced: Trajectories inside the produced groups.
        rollout_errors: Rollouts that raised. Counted, logged and retried rather than
            killing the run: one bad batch should not end a job that has been going for
            hours.
        waits: How many times the trainer had to block because the queue was empty. This
            is the number that says whether the sampler is keeping up; a healthy
            decoupled run should see it near zero after warmup.
    """

    groups_produced: int = 0
    groups_dropped_stale: int = 0
    groups_dropped_empty: int = 0
    trajectories_produced: int = 0
    rollout_errors: int = 0
    waits: int = 0
    rollout: RolloutStats = field(default_factory=RolloutStats)

    def as_metrics(self, prefix: str = "producer") -> dict[str, float]:
        """Flatten to ``{f"{prefix}/{field}": value}`` for the logger."""
        metrics = {
            f"{prefix}/groups_produced": float(self.groups_produced),
            f"{prefix}/groups_dropped_stale": float(self.groups_dropped_stale),
            f"{prefix}/groups_dropped_empty": float(self.groups_dropped_empty),
            f"{prefix}/trajectories_produced": float(self.trajectories_produced),
            f"{prefix}/rollout_errors": float(self.rollout_errors),
            f"{prefix}/waits": float(self.waits),
        }
        metrics.update(self.rollout.as_metrics(f"{prefix}/rollout"))
        return metrics


def validate_producer_config(concurrency: int, queue_size: int, env_workers: int) -> None:
    """Reject or warn about producer settings that cannot mean what they say.

    Args:
        concurrency: Workers sampling in parallel.
        queue_size: Bound on queued groups.
        env_workers: The environment's own scoring fan-out.

    Raises:
        ValueError: If ``concurrency`` or ``queue_size`` is not positive.
    """
    if concurrency <= 0:
        raise ValueError(f"producer concurrency must be positive, got {concurrency}")
    if queue_size <= 0:
        raise ValueError(
            f"producer queue_size must be positive, got {queue_size}; an unbounded queue "
            "would let a fast sampler run arbitrarily far ahead of the trainer"
        )
    subprocesses = concurrency * env_workers
    if subprocesses > MAX_REASONABLE_SUBPROCESSES:
        logger.warning(
            "producer.concurrency=%d x env.workers=%d means up to %d concurrent sandboxed "
            "executions. Each is a forked interpreter running generated code; consider "
            "lowering env.workers now that scoring is fanned out by the producer instead.",
            concurrency,
            env_workers,
            subprocesses,
        )


class TrajectoryProducer:
    """Keeps ``concurrency`` groups in flight and queues the finished ones.

    Start it with :meth:`start`, take batches with :meth:`drain`, and shut it down with
    :meth:`stop` — or use it as an async context manager, which does both.
    """

    def __init__(
        self,
        backend: Any,
        env: Any,
        tokenizer: ChatTokenizer,
        cfg: Any,
        tasks: Sequence[Task],
        *,
        concurrency: int = 8,
        queue_size: int = 32,
        max_staleness: int = 0,
        rng: random.Random | None = None,
        env_workers: int = 8,
        empty_backoff_seconds: float = 0.05,
    ) -> None:
        """Configure the producer without starting it.

        Args:
            backend: Anything with :meth:`~flowcode.rollout.SamplerLike.sample`. Read for
                ``policy_version`` too, when it has one.
            env: The scoring environment.
            tokenizer: The model's tokenizer, for rendering and decoding.
            cfg: The composed run config; supplies the ``train.*`` sampling knobs.
            tasks: Tasks to sample from, drawn uniformly at random with replacement. Unlike
                the synchronous loop there is no per-step "pick k distinct tasks": workers
                are independent and a step's batch is whatever finished first.
            concurrency: Workers sampling in parallel. Each holds one group, so sequences
                in flight is ``concurrency * train.group_size``.
            queue_size: Bound on queued groups; backpressure once full.
            max_staleness: Discard groups more than this many policy versions behind at
                drain time. ``0`` disables the check and keeps everything, which is the
                correct default — the objectives do not need the bound, it exists for
                ablations and for capping how far behind the batch can drift.
            rng: Seeded RNG for task selection.
            env_workers: The environment's scoring fan-out, for the warning in
                :func:`validate_producer_config`.
            empty_backoff_seconds: How long a worker pauses after a rollout that produced
                nothing usable. Small but non-zero: nothing on that path is guaranteed to
                await, so without it a sampler returning instant garbage starves the loop.

        Raises:
            ValueError: On an incoherent configuration.
        """
        validate_producer_config(concurrency, queue_size, env_workers)
        if not tasks:
            raise ValueError("TrajectoryProducer needs at least one task to sample from")

        self.backend = backend
        self.env = env
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.tasks = list(tasks)
        self.concurrency = concurrency
        self.max_staleness = max_staleness
        self.rng = random.Random(0) if rng is None else rng
        self.stats = ProducerStats()
        self._empty_backoff_seconds = empty_backoff_seconds

        self._queue: asyncio.Queue[ProducedGroup] = asyncio.Queue(maxsize=queue_size)
        self._workers: list[asyncio.Task[None]] = []
        self._stopping = asyncio.Event()

    @property
    def policy_version(self) -> int:
        """The backend's current policy revision, or 0 for a backend without one."""
        return int(getattr(self.backend, "policy_version", 0))

    @property
    def queued_groups(self) -> int:
        """Groups waiting to be drained. A healthy run keeps this off zero."""
        return self._queue.qsize()

    async def start(self) -> None:
        """Spawn the workers. Idempotent."""
        if self._workers:
            return
        self._stopping.clear()
        self._workers = [
            asyncio.create_task(self._worker(i), name=f"flowcode-producer-{i}")
            for i in range(self.concurrency)
        ]
        logger.info(
            "producer started: %d workers, %d sequences in flight",
            self.concurrency,
            self.concurrency * self.cfg.train.group_size,
        )

    async def stop(self) -> None:
        """Cancel the workers and wait for them to unwind. Idempotent."""
        if not self._workers:
            return
        self._stopping.set()
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers = []
        logger.info("producer stopped after %d groups", self.stats.groups_produced)

    async def __aenter__(self) -> TrajectoryProducer:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    async def drain(self, num_groups: int) -> list[Trajectory]:
        """Take ``num_groups`` finished groups, waiting only if the queue is short.

        Blocks on the first group and then takes whatever else is already queued, so a
        trainer that has outrun the sampler waits for one group rather than for a whole
        batch, and a trainer that has not outrun it never waits at all.

        Args:
            num_groups: How many groups the step wants — ``train.groups_per_step``.

        Returns:
            The groups' trajectories, flattened, group-contiguous. Fewer than
            ``num_groups`` worth if the queue ran dry after the first.

        Raises:
            ValueError: If ``num_groups`` is not positive.
            RuntimeError: If the producer was never started.
        """
        if num_groups <= 0:
            raise ValueError(f"num_groups must be positive, got {num_groups}")
        if not self._workers:
            raise RuntimeError("TrajectoryProducer.drain() called before start()")

        collected: list[Trajectory] = []
        taken = 0
        while taken < num_groups:
            if collected and self._queue.empty():
                break
            if self._queue.empty():
                self.stats.waits += 1
            group = await self._queue.get()
            if self._is_stale(group):
                self.stats.groups_dropped_stale += 1
                continue
            collected.extend(group.trajectories)
            taken += 1
        return collected

    def _is_stale(self, group: ProducedGroup) -> bool:
        if self.max_staleness <= 0:
            return False
        return (self.policy_version - group.policy_version) > self.max_staleness

    async def _worker(self, index: int) -> None:
        """Sample and score one task at a time, forever, until cancelled."""
        while not self._stopping.is_set():
            task = self.rng.choice(self.tasks)
            version = self.policy_version
            try:
                trajectories = await rollout(
                    self.backend,
                    self.env,
                    self.tokenizer,
                    [task],
                    self.cfg,
                    num_samples=self.cfg.train.group_size,
                    stats=self.stats.rollout,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                # One bad rollout should not end a run that has been going for hours.
                self.stats.rollout_errors += 1
                logger.exception("producer worker %d: rollout failed for %s", index, task.task_id)
                await asyncio.sleep(1.0)
                continue

            if not trajectories:
                self.stats.groups_dropped_empty += 1
                # Yield before retrying. Nothing on this path is guaranteed to await —
                # a sampler that returns instantly with nothing usable would otherwise
                # spin this worker at 100% CPU and starve the event loop, which is
                # exactly the state a misconfigured max_tokens or a broken stop sequence
                # puts it in.
                await asyncio.sleep(self._empty_backoff_seconds)
                continue

            for trajectory in trajectories:
                trajectory.metadata["policy_version"] = version
            # Blocks when the queue is full: this is the backpressure that stops the
            # sampler running arbitrarily far ahead of the trainer.
            await self._queue.put(
                ProducedGroup(
                    task_id=task.task_id, trajectories=trajectories, policy_version=version
                )
            )
            self.stats.groups_produced += 1
            self.stats.trajectories_produced += len(trajectories)

    async def prefill(self, num_groups: int, timeout: float | None = None) -> int:
        """Wait until ``num_groups`` are queued, so the first step does not start empty.

        Args:
            num_groups: Groups to wait for.
            timeout: Seconds to wait, or ``None`` for no limit.

        Returns:
            How many groups are queued when this returns — fewer than asked for if the
            timeout fired.
        """

        async def wait() -> None:
            while self._queue.qsize() < num_groups:
                await asyncio.sleep(0.05)

        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(wait(), timeout)
        return self._queue.qsize()

    def __repr__(self) -> str:
        return (
            f"TrajectoryProducer(concurrency={self.concurrency}, "
            f"queued={self.queued_groups}, produced={self.stats.groups_produced})"
        )
