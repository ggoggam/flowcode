"""Rollouts against a fake sampler: shape, ordering, and the two throughput properties.

No network and no API key. The sampler is a stub returning canned
:class:`~flowcode.types.SampleResponse` objects — the backend-neutral type, so this file
never imports an engine SDK — and the environment is either a recording stub or the *real*
``dataset: fixtures`` environment, which executes code in a subprocess and is the only way
to prove the reward path is wired up end to end.

Two non-obvious properties are asserted here because nothing else would catch them:

* scoring runs on a worker thread, not the event loop — ``batch_log_reward`` blocks in
  ``waitpid`` for the whole batch, and doing that inline stalls every other request in
  flight;
* the segments of every produced trajectory tile the completion exactly, which
  :class:`~flowcode.types.Trajectory` enforces but only for the layout it is handed.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Literal

import pytest

from flowcode.config import ModelConfig, RootConfig, TrainConfig
from flowcode.envs.base import RewardResult, Task
from flowcode.envs.code_exec import CodeExecEnv
from flowcode.envs.datasets import load_fixture_tasks
from flowcode.rollout import RolloutStats, rollout, solution_fingerprint
from flowcode.types import SampledSequence, SampleResponse, SamplingParams

CODE_BLOCK = "```python\ndef solve(x):\n    return x\n```"


# ------------------------------------------------------------------------------- fakes


class FakeTokenizer:
    """Renders prompts as short token lists and decodes completions to fixed text."""

    chat_template = "{{ messages }}"
    eos_token_id = 7
    unk_token_id = 0

    def __init__(self, decoded: str = CODE_BLOCK, decode_map: dict[int, str] | None = None) -> None:
        self.decoded = decoded
        self.decode_map = decode_map or {}
        self.rendered: list[Any] = []

    def apply_chat_template(self, conversation: Any, **kwargs: Any) -> list[int]:
        self.rendered.append(conversation)
        return [11, 12, 13, len(self.rendered)]

    def encode(self, text: Any, **kwargs: Any) -> list[int]:
        return [1, 2, 3]

    def decode(self, token_ids: Any, **kwargs: Any) -> str:
        tokens = list(token_ids)
        if tokens and tokens[0] in self.decode_map:
            return self.decode_map[tokens[0]]
        return self.decoded

    def convert_tokens_to_ids(self, tokens: Any) -> int:
        return 7


class FakeSampler:
    """Returns ``num_samples`` canned sequences per prompt and records the request."""

    def __init__(
        self,
        completion_length: int = 4,
        empty_indices: tuple[int, ...] = (),
        stop_reason: Literal["length", "stop"] = "stop",
        first_tokens: tuple[int, ...] | None = None,
    ) -> None:
        self.completion_length = completion_length
        self.empty_indices = empty_indices
        self.stop_reason = stop_reason
        self.first_tokens = first_tokens
        self.calls: list[tuple[list[list[int]], int, SamplingParams]] = []

    async def sample(
        self,
        prompts: Any,
        num_samples: int,
        sampling_params: SamplingParams,
    ) -> list[SampleResponse]:
        self.calls.append(([list(p) for p in prompts], num_samples, sampling_params))
        responses: list[SampleResponse] = []
        index = 0
        for _prompt in prompts:
            sequences: list[SampledSequence] = []
            for sample in range(num_samples):
                length = 0 if index in self.empty_indices else self.completion_length
                first = 100 + index if self.first_tokens is None else self.first_tokens[index]
                sequences.append(
                    SampledSequence(
                        tokens=[first + j for j in range(length)],
                        logprobs=[-0.25 - 0.01 * sample] * length,
                        stop_reason=self.stop_reason,
                    )
                )
                index += 1
            responses.append(SampleResponse(sequences=sequences, prompt_cache_hit_tokens=3))
        return responses


class RecordingEnv:
    """Scores everything the same way and remembers which thread it ran on."""

    name = "recording"

    def __init__(self, log_reward: float = -0.5, passed: bool = True) -> None:
        self.log_reward = log_reward
        self.passed = passed
        self.pairs: list[tuple[str, str]] = []
        self.threads: list[str] = []

    def tasks(self) -> list[Task]:
        return TASKS

    def log_reward_for(self, task: Task, completion: str) -> RewardResult:
        return RewardResult(
            log_reward=self.log_reward,
            pass_fraction=1.0 if self.passed else 0.0,
            passed=self.passed,
            error=None if self.passed else "assertion",
        )

    def batch_log_reward(self, pairs: Any) -> list[RewardResult]:
        self.threads.append(threading.current_thread().name)
        self.pairs.extend((task.task_id, text) for task, text in pairs)
        return [self.log_reward_for(task, text) for task, text in pairs]


TASKS = [
    Task(task_id="a", prompt="add two numbers"),
    Task(task_id="b", prompt="reverse a string"),
]


def make_cfg(**train_kwargs: Any) -> RootConfig:
    """A RootConfig instance, bypassing Hydra — rollout only reads model/train."""
    cfg = RootConfig()
    cfg.model = ModelConfig(name="Qwen/Qwen3-8B", renderer="qwen3")
    train = TrainConfig(group_size=2, max_tokens=64, temperature=0.8, top_p=0.95)
    for key, value in train_kwargs.items():
        assert hasattr(train, key), f"TrainConfig has no field {key!r}"
        setattr(train, key, value)
    cfg.train = train
    return cfg


# ------------------------------------------------------------------------------- tests


class TestTrajectoryShape:
    async def test_one_trajectory_per_sample(self) -> None:
        trajectories = await rollout(FakeSampler(), RecordingEnv(), FakeTokenizer(), TASKS, cfg())
        assert len(trajectories) == len(TASKS) * 2

    async def test_trajectories_are_grouped_by_task_in_task_order(self) -> None:
        # Group-baseline objectives (VarGrad) group by task_id, so a scrambled order would
        # not break anything visibly — it would just quietly shrink the groups.
        trajectories = await rollout(FakeSampler(), RecordingEnv(), FakeTokenizer(), TASKS, cfg())
        assert [t.task_id for t in trajectories] == ["a", "a", "b", "b"]

    async def test_segments_tile_the_completion_at_token_level(self) -> None:
        trajectories = await rollout(
            FakeSampler(completion_length=5), RecordingEnv(), FakeTokenizer(), TASKS, cfg()
        )
        for trajectory in trajectories:
            segments = trajectory.segments
            assert len(segments) == trajectory.num_completion_tokens
            assert segments[0].start == 0
            assert segments[-1].end == trajectory.num_completion_tokens
            assert all(len(s) == 1 for s in segments)

    async def test_sampling_logprobs_come_from_the_sampler(self) -> None:
        trajectories = await rollout(FakeSampler(), RecordingEnv(), FakeTokenizer(), TASKS, cfg())
        first, second = trajectories[0], trajectories[1]
        assert first.sampling_logprobs == pytest.approx([-0.25] * 4)
        assert second.sampling_logprobs == pytest.approx([-0.26] * 4)
        assert len(first.sampling_logprobs) == first.num_completion_tokens

    async def test_log_reward_comes_from_the_environment(self) -> None:
        env = RecordingEnv(log_reward=-1.75)
        trajectories = await rollout(FakeSampler(), env, FakeTokenizer(), TASKS, cfg())
        assert all(t.log_reward == -1.75 for t in trajectories)

    async def test_prompt_tokens_are_the_rendered_prompt(self) -> None:
        tokenizer = FakeTokenizer()
        trajectories = await rollout(FakeSampler(), RecordingEnv(), tokenizer, TASKS, cfg())
        assert trajectories[0].prompt_tokens == [11, 12, 13, 1]
        assert trajectories[-1].prompt_tokens == [11, 12, 13, 2]

    async def test_metadata_carries_the_metrics_the_loop_reports(self) -> None:
        trajectories = await rollout(FakeSampler(), RecordingEnv(), FakeTokenizer(), TASKS, cfg())
        metadata = trajectories[0].metadata
        assert metadata["pass_fraction"] == 1.0
        assert metadata["passed"] is True
        assert metadata["error"] is None
        assert metadata["stop_reason"] == "stop"
        assert metadata["solution_fingerprint"]


class TestSamplingParameters:
    async def test_config_knobs_reach_the_sampler(self) -> None:
        sampler = FakeSampler()
        await rollout(sampler, RecordingEnv(), FakeTokenizer(), TASKS, cfg())
        _prompts, num_samples, params = sampler.calls[0]
        assert num_samples == 2
        assert params.max_tokens == 64
        assert params.temperature == pytest.approx(0.8)
        assert params.top_p == pytest.approx(0.95)

    async def test_stop_sequences_are_attached(self) -> None:
        sampler = FakeSampler()
        await rollout(sampler, RecordingEnv(), FakeTokenizer(), TASKS, cfg())
        assert sampler.calls[0][2].stop == [7]

    async def test_overrides_win_over_the_config(self) -> None:
        # The eval path relies on this: one greedy sample per task.
        sampler = FakeSampler()
        await rollout(
            sampler,
            RecordingEnv(),
            FakeTokenizer(),
            TASKS,
            cfg(),
            num_samples=1,
            temperature=0.0,
            max_tokens=16,
            seed=99,
        )
        _prompts, num_samples, params = sampler.calls[0]
        assert num_samples == 1
        assert params.temperature == 0.0
        assert params.max_tokens == 16
        assert params.seed == 99

    async def test_one_sample_call_for_all_prompts(self) -> None:
        # The backend gathers over prompts internally; a per-prompt loop here would
        # serialise what it deliberately parallelised.
        sampler = FakeSampler()
        await rollout(sampler, RecordingEnv(), FakeTokenizer(), TASKS, cfg())
        assert len(sampler.calls) == 1
        assert len(sampler.calls[0][0]) == 2


class TestScoring:
    async def test_scoring_happens_off_the_event_loop(self) -> None:
        env = RecordingEnv()
        loop_thread = threading.current_thread().name
        await rollout(FakeSampler(), env, FakeTokenizer(), TASKS, cfg())
        assert env.threads and all(name != loop_thread for name in env.threads)

    async def test_the_event_loop_keeps_running_during_scoring(self) -> None:
        class SlowEnv(RecordingEnv):
            def batch_log_reward(self, pairs: Any) -> list[RewardResult]:
                import time

                time.sleep(0.15)
                return super().batch_log_reward(pairs)

        ticks = 0

        async def tick() -> None:
            nonlocal ticks
            for _ in range(10):
                await asyncio.sleep(0.01)
                ticks += 1

        await asyncio.gather(
            rollout(FakeSampler(), SlowEnv(), FakeTokenizer(), TASKS, cfg()), tick()
        )
        assert ticks == 10

    async def test_pairs_are_task_and_decoded_text_in_order(self) -> None:
        env = RecordingEnv()
        await rollout(FakeSampler(), env, FakeTokenizer(decoded="hello"), TASKS, cfg())
        assert env.pairs == [("a", "hello"), ("a", "hello"), ("b", "hello"), ("b", "hello")]

    async def test_misaligned_results_are_rejected(self) -> None:
        class ShortEnv(RecordingEnv):
            def batch_log_reward(self, pairs: Any) -> list[RewardResult]:
                return super().batch_log_reward(pairs)[:1]

        with pytest.raises(ValueError, match="one-to-one"):
            await rollout(FakeSampler(), ShortEnv(), FakeTokenizer(), TASKS, cfg())


class TestDegenerateSamples:
    async def test_empty_completions_are_dropped_not_raised(self) -> None:
        stats = RolloutStats()
        trajectories = await rollout(
            FakeSampler(empty_indices=(0, 3)),
            RecordingEnv(),
            FakeTokenizer(),
            TASKS,
            cfg(),
            stats=stats,
        )
        assert len(trajectories) == 2
        assert stats.num_empty == 2
        assert stats.num_requested == 4
        assert stats.num_returned == 4

    async def test_all_empty_returns_an_empty_batch(self) -> None:
        trajectories = await rollout(
            FakeSampler(empty_indices=(0, 1, 2, 3)), RecordingEnv(), FakeTokenizer(), TASKS, cfg()
        )
        assert trajectories == []

    async def test_truncation_is_counted(self) -> None:
        stats = RolloutStats()
        await rollout(
            FakeSampler(stop_reason="length"),
            RecordingEnv(),
            FakeTokenizer(),
            TASKS,
            cfg(),
            stats=stats,
        )
        assert stats.num_truncated == 4

    async def test_empty_task_list_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one task"):
            await rollout(FakeSampler(), RecordingEnv(), FakeTokenizer(), [], cfg())

    async def test_non_positive_num_samples_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            await rollout(
                FakeSampler(), RecordingEnv(), FakeTokenizer(), TASKS, cfg(), num_samples=0
            )


class TestStats:
    async def test_stats_are_reported_as_metrics(self) -> None:
        stats = RolloutStats()
        await rollout(FakeSampler(), RecordingEnv(), FakeTokenizer(), TASKS, cfg(), stats=stats)
        metrics = stats.as_metrics()
        assert metrics["rollout/num_returned"] == 4.0
        assert metrics["rollout/completion_tokens"] == 16.0
        assert metrics["rollout/prompt_cache_hit_tokens"] == 6.0


class TestSolutionFingerprint:
    def test_identical_code_hashes_identically(self) -> None:
        a = "here you go:\n```python\ndef f():\n    return 1\n```"
        b = "sure!\n```python\ndef f():\n    return 1\n```\nHope that helps."
        assert solution_fingerprint(a) == solution_fingerprint(b)

    def test_different_code_hashes_differently(self) -> None:
        a = "```python\ndef f():\n    return 1\n```"
        b = "```python\ndef f():\n    return 2\n```"
        assert solution_fingerprint(a) != solution_fingerprint(b)

    def test_prose_only_has_no_fingerprint(self) -> None:
        assert solution_fingerprint("I am afraid I cannot help with that.") == ""


class TestAgainstTheRealFixturesEnvironment:
    """The reward path for real: fixtures, a sandbox subprocess, an actual pass/fail."""

    async def test_reference_solutions_score_as_passing(self) -> None:
        task = next(
            t for t in load_fixture_tasks() if str(t.metadata.get("reference_solution", "")).strip()
        )
        solution = str(task.metadata["reference_solution"])
        tokenizer = FakeTokenizer(decoded=f"```python\n{solution}\n```")
        env = CodeExecEnv(name="fixtures", dataset="fixtures", split="all", workers=2)

        trajectories = await rollout(
            FakeSampler(completion_length=3), env, tokenizer, [task], cfg(), num_samples=1
        )
        assert len(trajectories) == 1
        assert trajectories[0].metadata["passed"] is True
        assert trajectories[0].metadata["pass_fraction"] == 1.0
        assert trajectories[0].log_reward == pytest.approx(0.0)

    async def test_garbage_completions_score_as_empty(self) -> None:
        task = load_fixture_tasks()[0]
        tokenizer = FakeTokenizer(decoded="I would rather not.")
        env = CodeExecEnv(name="fixtures", dataset="fixtures", split="all", workers=2)

        trajectories = await rollout(
            FakeSampler(completion_length=3), env, tokenizer, [task], cfg(), num_samples=1
        )
        assert trajectories[0].metadata["error"] == "empty"
        assert trajectories[0].log_reward < 0.0


def cfg(**train_kwargs: Any) -> RootConfig:
    """Shorthand used by the tests above."""
    return make_cfg(**train_kwargs)
