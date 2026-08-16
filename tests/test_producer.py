"""The decoupled sampler: group integrity, backpressure, staleness and failure handling.

The producer's whole justification is that it can run ahead of the trainer, so the tests
that matter are about what happens at the seams — a queue that fills, a trainer that
outruns the sampler, a rollout that raises, a policy that moves on while groups are still
in flight. None of that needs a model; the sampler is a stub.
"""

from __future__ import annotations

import asyncio
import random
from typing import Any

import pytest

from flowcode.config import ModelConfig, RootConfig, TrainConfig
from flowcode.envs.base import RewardResult, Task
from flowcode.producer import (
    ProducedGroup,
    ProducerStats,
    TrajectoryProducer,
    validate_producer_config,
)
from flowcode.types import SampledSequence, SampleResponse, SamplingParams, Segment, Trajectory

TASKS = [
    Task(task_id="a", prompt="add two numbers"),
    Task(task_id="b", prompt="reverse a string"),
]

CODE_BLOCK = "```python\ndef solve(x):\n    return x\n```"


class FakeTokenizer:
    chat_template = "{{ messages }}"
    eos_token_id = 7
    unk_token_id = 0

    def apply_chat_template(self, conversation: Any, **kwargs: Any) -> list[int]:
        return [11, 12, 13]

    def encode(self, text: Any, **kwargs: Any) -> list[int]:
        return [1, 2, 3]

    def decode(self, token_ids: Any, **kwargs: Any) -> str:
        return CODE_BLOCK

    def convert_tokens_to_ids(self, tokens: Any) -> int:
        return 7


class FakeBackend:
    """Returns canned groups; ``policy_version`` is settable to simulate a sync."""

    def __init__(self, completion_length: int = 3, delay: float = 0.0) -> None:
        self.completion_length = completion_length
        self.delay = delay
        self.policy_version = 0
        self.calls = 0

    async def sample(
        self, prompts: Any, num_samples: int, sampling_params: SamplingParams
    ) -> list[SampleResponse]:
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        return [
            SampleResponse(
                sequences=[
                    SampledSequence(
                        tokens=[100 + i for i in range(self.completion_length)],
                        logprobs=[-0.3] * self.completion_length,
                        stop_reason="stop",
                    )
                    for _ in range(num_samples)
                ],
                prompt_cache_hit_tokens=1,
            )
            for _ in prompts
        ]


class EmptyBackend(FakeBackend):
    """Every sample is degenerate — the group is unusable."""

    async def sample(
        self, prompts: Any, num_samples: int, sampling_params: SamplingParams
    ) -> list[SampleResponse]:
        self.calls += 1
        return [
            SampleResponse(
                sequences=[
                    SampledSequence(tokens=[], logprobs=[], stop_reason="stop")
                    for _ in range(num_samples)
                ]
            )
            for _ in prompts
        ]


class ExplodingBackend(FakeBackend):
    """Fails the first ``failures`` calls, then behaves."""

    def __init__(self, failures: int = 2) -> None:
        super().__init__()
        self.failures = failures

    async def sample(
        self, prompts: Any, num_samples: int, sampling_params: SamplingParams
    ) -> list[SampleResponse]:
        if self.calls < self.failures:
            self.calls += 1
            raise RuntimeError("the engine fell over")
        return await super().sample(prompts, num_samples, sampling_params)


class RecordingEnv:
    name = "recording"

    def tasks(self) -> list[Task]:
        return TASKS

    def batch_log_reward(self, pairs: Any) -> list[RewardResult]:
        return [
            RewardResult(log_reward=-0.5, pass_fraction=1.0, passed=True, error=None) for _ in pairs
        ]


def make_cfg(**train_kwargs: Any) -> RootConfig:
    cfg = RootConfig()
    cfg.model = ModelConfig(name="Qwen/Qwen3-8B", renderer="qwen3")
    train = TrainConfig(group_size=4, max_tokens=64, temperature=0.8, top_p=0.95)
    for key, value in train_kwargs.items():
        assert hasattr(train, key), f"TrainConfig has no field {key!r}"
        setattr(train, key, value)
    cfg.train = train
    return cfg


def make_producer(backend: Any = None, env: Any = None, **kwargs: Any) -> TrajectoryProducer:
    return TrajectoryProducer(
        backend if backend is not None else FakeBackend(),
        env if env is not None else RecordingEnv(),
        FakeTokenizer(),
        kwargs.pop("cfg", make_cfg()),
        TASKS,
        rng=random.Random(0),
        **kwargs,
    )


def make_trajectory(task_id: str = "a", policy_version: int = 0) -> Trajectory:
    return Trajectory(
        task_id=task_id,
        prompt_tokens=[1, 2],
        completion_tokens=[3, 4],
        sampling_logprobs=[-0.1, -0.2],
        log_reward=-0.5,
        segments=[Segment(0, 2)],
        metadata={"policy_version": policy_version},
    )


# ------------------------------------------------------------------------- validation


class TestValidateProducerConfig:
    def test_rejects_zero_concurrency(self) -> None:
        with pytest.raises(ValueError, match="concurrency must be positive"):
            validate_producer_config(0, 8, 4)

    def test_rejects_unbounded_queue(self) -> None:
        # An unbounded queue lets a fast sampler run arbitrarily far ahead, which is a
        # memory leak dressed up as throughput.
        with pytest.raises(ValueError, match="queue_size must be positive"):
            validate_producer_config(4, 0, 4)

    def test_warns_when_subprocess_count_explodes(self, caplog: pytest.LogCaptureFixture) -> None:
        # concurrency x env.workers is the real subprocess count, and it is not obvious.
        with caplog.at_level("WARNING"):
            validate_producer_config(32, 8, 8)
        assert "concurrent sandboxed executions" in caplog.text

    def test_quiet_at_sane_settings(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("WARNING"):
            validate_producer_config(8, 32, 4)
        assert caplog.text == ""

    def test_no_tasks_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one task"):
            TrajectoryProducer(FakeBackend(), RecordingEnv(), FakeTokenizer(), make_cfg(), [])


# ---------------------------------------------------------------------------- producing


class TestProducing:
    async def test_drain_returns_whole_groups(self) -> None:
        # group_size=4, so a group of 4 trajectories sharing a task_id. Splitting a group
        # across steps would break every group-baseline estimator silently.
        async with make_producer(concurrency=2) as producer:
            await producer.prefill(2, timeout=5.0)
            trajectories = await producer.drain(2)
        assert len(trajectories) % 4 == 0
        assert len(trajectories) >= 4

    async def test_a_drained_group_shares_one_task_id(self) -> None:
        async with make_producer(concurrency=1) as producer:
            await producer.prefill(1, timeout=5.0)
            trajectories = await producer.drain(1)
        assert len({t.task_id for t in trajectories}) == 1

    async def test_stamps_the_policy_version(self) -> None:
        backend = FakeBackend()
        backend.policy_version = 7
        async with make_producer(backend, concurrency=1) as producer:
            await producer.prefill(1, timeout=5.0)
            trajectories = await producer.drain(1)
        assert all(t.metadata["policy_version"] == 7 for t in trajectories)

    async def test_keeps_producing_without_being_asked(self) -> None:
        # The whole point: the sampler does not wait for the trainer to come back.
        async with make_producer(concurrency=4, queue_size=16) as producer:
            queued = await producer.prefill(8, timeout=5.0)
        assert queued >= 8

    async def test_counts_what_it_produced(self) -> None:
        async with make_producer(concurrency=2) as producer:
            await producer.prefill(3, timeout=5.0)
            await producer.drain(3)
            stats = producer.stats
        assert stats.groups_produced >= 3
        assert stats.trajectories_produced >= 12

    async def test_drain_before_start_is_an_error(self) -> None:
        producer = make_producer()
        with pytest.raises(RuntimeError, match="before start"):
            await producer.drain(1)

    async def test_non_positive_drain_rejected(self) -> None:
        async with make_producer() as producer:
            with pytest.raises(ValueError, match="num_groups must be positive"):
                await producer.drain(0)

    async def test_start_and_stop_are_idempotent(self) -> None:
        producer = make_producer(concurrency=1)
        await producer.start()
        await producer.start()
        await producer.stop()
        await producer.stop()
        assert producer.queued_groups >= 0


# -------------------------------------------------------------------------- backpressure


class TestBackpressure:
    async def test_queue_does_not_grow_past_its_bound(self) -> None:
        async with make_producer(concurrency=4, queue_size=3) as producer:
            await producer.prefill(3, timeout=5.0)
            await asyncio.sleep(0.2)  # let the workers pile up against the bound
            assert producer.queued_groups <= 3

    async def test_drain_takes_what_is_there_without_waiting_for_more(self) -> None:
        # A trainer that has outrun the sampler should wait for one group, not a batch.
        async with make_producer(concurrency=1, queue_size=8) as producer:
            await producer.prefill(1, timeout=5.0)
            trajectories = await producer.drain(100)
        assert trajectories  # got something rather than blocking for all 100

    async def test_counts_waits_when_the_queue_is_dry(self) -> None:
        # This is the number that says whether the sampler is keeping up.
        async with make_producer(FakeBackend(delay=0.05), concurrency=1) as producer:
            await producer.drain(1)
            assert producer.stats.waits >= 1


# ------------------------------------------------------------------------------ staleness


class TestStaleness:
    def test_disabled_by_default(self) -> None:
        producer = make_producer()
        assert producer.max_staleness == 0
        assert not producer._is_stale(ProducedGroup("a", [make_trajectory()], policy_version=0))

    def test_old_groups_are_stale_once_a_bound_is_set(self) -> None:
        backend = FakeBackend()
        backend.policy_version = 10
        producer = make_producer(backend, max_staleness=2)
        assert producer._is_stale(ProducedGroup("a", [make_trajectory()], policy_version=7))
        assert not producer._is_stale(ProducedGroup("a", [make_trajectory()], policy_version=9))

    async def test_stale_groups_are_dropped_at_drain(self) -> None:
        # Groups queued at version 0 while the policy races ahead to 5 are past a
        # max_staleness of 1, so drain discards them and waits for fresh ones instead.
        backend = FakeBackend()
        async with make_producer(backend, concurrency=1, max_staleness=1) as producer:
            await producer.prefill(3, timeout=5.0)
            backend.policy_version = 5
            trajectories = await producer.drain(1)
        assert producer.stats.groups_dropped_stale >= 1
        # What comes back is from the current policy, not the discarded revision.
        assert all(t.metadata["policy_version"] == 5 for t in trajectories)

    async def test_nothing_is_dropped_when_the_bound_is_off(self) -> None:
        backend = FakeBackend()
        async with make_producer(backend, concurrency=1) as producer:
            await producer.prefill(2, timeout=5.0)
            backend.policy_version = 99
            await producer.drain(1)
        assert producer.stats.groups_dropped_stale == 0


# ------------------------------------------------------------------------ failure modes


class TestFailureModes:
    async def test_a_failing_rollout_is_counted_not_fatal(self) -> None:
        # A run that has been going for hours should not die on one bad batch.
        backend = ExplodingBackend(failures=2)
        async with make_producer(backend, concurrency=1) as producer:
            await producer.prefill(1, timeout=10.0)
            trajectories = await producer.drain(1)
        assert producer.stats.rollout_errors == 2
        assert trajectories

    async def test_all_degenerate_groups_are_dropped_not_queued(self) -> None:
        async with make_producer(EmptyBackend(), concurrency=1) as producer:
            await asyncio.sleep(0.2)
            assert producer.queued_groups == 0
            assert producer.stats.groups_dropped_empty > 0

    async def test_stop_cancels_in_flight_work(self) -> None:
        producer = make_producer(FakeBackend(delay=5.0), concurrency=2)
        await producer.start()
        await asyncio.sleep(0.05)
        await producer.stop()
        assert producer._workers == []


class TestProducerStats:
    def test_metrics_are_flat_floats(self) -> None:
        metrics = ProducerStats(groups_produced=3, waits=1).as_metrics()
        assert metrics["producer/groups_produced"] == 3.0
        assert metrics["producer/waits"] == 1.0
        assert all(isinstance(v, float) for v in metrics.values())

    def test_rollout_stats_are_nested_under_the_prefix(self) -> None:
        metrics = ProducerStats().as_metrics()
        assert "producer/rollout/num_requested" in metrics
