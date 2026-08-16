"""The training loop against fakes: order, arithmetic, and the config that must be refused.

Everything here runs offline. The backend is a stub that records the order of its calls and
returns canned logprobs; the environment scores from a table; the objectives are the real
ones, because the point is to check that the loop drives them correctly, not to re-test the
losses.

Three things are asserted that nothing else in the suite can catch:

* ``on_policy_only`` together with replay is **refused**, loudly, before anything is
  created. Allowing it would train the balance conditions against a stale policy's
  logprobs — no exception, no wrong-looking metric, just a run that optimises the wrong
  thing.
* ``apply_gradient(defer=True)`` is submitted **before** ``optim_step``, so both land in
  one Tinker clock cycle. Swap them and the run still trains, at twice the latency.
* the client-side flow parameters actually move. They have their own optimiser at their own
  learning rate, and nothing else in the pipeline would notice if ``step()`` were never
  called on it.
"""

from __future__ import annotations

import io
import math
import random
import sys
from typing import Any, cast

import numpy as np
import pytest
import torch
from rich.console import Console
from tinker.types import SampledSequence, SampleResponse, SamplingParams

from flowcode.config import CostConfig, ModelConfig, ReplayConfig, RootConfig, TrainConfig
from flowcode.envs.base import RewardResult, Task
from flowcode.logging import LOGGER_KINDS, RichLogger, make_logger
from flowcode.objectives.tb import TrajectoryBalance
from flowcode.objectives.vargrad import VarGrad
from flowcode.train import (
    LR_SCHEDULES,
    Trainer,
    _instantiate_env,
    _instantiate_objective,
    _tokenizer_of,
    learning_rate,
    validate_train_config,
)
from flowcode.types import TokenUsage

MODEL_CFG = ModelConfig(name="Qwen/Qwen3-8B", renderer="qwen3")

TASKS = [Task(task_id=f"task-{i}", prompt=f"solve problem {i}") for i in range(6)]


# ------------------------------------------------------------------------------- fakes


class FakeTokenizer:
    chat_template = "{{ messages }}"
    eos_token_id = 7
    unk_token_id = 0

    def __init__(self) -> None:
        self.prompt_calls = 0

    def apply_chat_template(self, conversation: Any, **kwargs: Any) -> list[int]:
        self.prompt_calls += 1
        return [11, 12, 13, 14]

    def encode(self, text: Any, **kwargs: Any) -> list[int]:
        return [1, 2, 3]

    def decode(self, token_ids: Any, **kwargs: Any) -> str:
        # Distinct completions decode to distinct programs, so the mode-diversity metric
        # has something real to count.
        first = next(iter(token_ids))
        return f"```python\ndef solve():\n    return {first}\n```"

    def convert_tokens_to_ids(self, tokens: Any) -> int:
        return 7


class FakeTrainingClient:
    """Just enough of a Tinker training client for the checkpoint and tokenizer paths."""

    def __init__(self, path: str = "tinker://ckpt/abc") -> None:
        self.path = path
        self.saved: list[str] = []

    async def save_weights_for_sampler_async(
        self, name: str, ttl_seconds: int | None = None
    ) -> Any:
        self.saved.append(name)

        class _Response:
            path = self.path

        class _Future:
            async def result_async(self, timeout: float | None = None) -> Any:
                return _Response()

        return _Future()

    def get_tokenizer(self) -> FakeTokenizer:
        return FakeTokenizer()


class FakeBackend:
    """Records call order, returns canned logprobs, spends fake tokens."""

    def __init__(
        self,
        completion_length: int = 4,
        training_client: FakeTrainingClient | None = None,
        fail_at_step: int | None = None,
    ) -> None:
        self.model_cfg = MODEL_CFG
        self.completion_length = completion_length
        self.log: list[str] = []
        self.sample_calls: list[tuple[int, SamplingParams]] = []
        self.apply_calls: list[dict[str, Any]] = []
        self.optim_calls: list[dict[str, Any]] = []
        self.usage = TokenUsage()
        self.fail_at_step = fail_at_step
        self.steps_seen = 0
        self._training_client = training_client

    async def sample(
        self, prompts: Any, num_samples: int, sampling_params: SamplingParams
    ) -> list[SampleResponse]:
        self.log.append("sample")
        self.sample_calls.append((num_samples, sampling_params))
        self.steps_seen += 1
        if self.fail_at_step is not None and self.steps_seen > self.fail_at_step:
            raise RuntimeError("the API fell over")
        responses: list[SampleResponse] = []
        token = 100
        for _prompt in prompts:
            sequences = []
            for _ in range(num_samples):
                tokens = np.arange(token, token + self.completion_length, dtype=np.int32)
                sequences.append(
                    SampledSequence(
                        stop_reason="stop",
                        tokens_np=tokens,
                        logprobs_np=np.full(self.completion_length, -0.3, dtype=np.float32),
                    )
                )
                token += 1
            responses.append(SampleResponse(sequences=sequences, prompt_cache_hit_tokens=2))
        self.usage += TokenUsage(sample_tokens=1000, prompt_cache_hit_tokens=200)
        return responses

    async def compute_logprobs(self, trajectories: Any) -> list[torch.Tensor]:
        self.log.append("compute_logprobs")
        self.usage += TokenUsage(train_tokens=500, num_forward_passes=1)
        return [
            torch.full((t.num_completion_tokens,), -0.4, dtype=torch.float32, requires_grad=True)
            for t in trajectories
        ]

    async def apply_gradient(
        self,
        trajectories: Any,
        grads: Any,
        *,
        grad_scale: float = 1.0,
        defer: bool = False,
    ) -> dict[str, float]:
        self.log.append("apply_gradient")
        self.apply_calls.append(
            {
                "batch_size": len(list(trajectories)),
                "grad_scale": grad_scale,
                "defer": defer,
                "grads": [g.clone() for g in grads],
            }
        )
        self.usage += TokenUsage(train_tokens=500, num_backward_passes=1)
        return {"weight_abs_sum": 1.0}

    async def optim_step(self, lr: float, *, grad_clip_norm: float = 0.0) -> dict[str, float]:
        self.log.append("optim_step")
        self.optim_calls.append({"lr": lr, "grad_clip_norm": grad_clip_norm})
        self.usage += TokenUsage(num_optim_steps=1)
        return {"lr": lr}

    async def sync_sampler(self) -> None:
        self.log.append("sync_sampler")

    def token_usage(self) -> TokenUsage:
        return self.usage


class FakeEnv:
    """Scores by a table keyed on the code the completion contains."""

    name = "fake"

    def __init__(self, passing_tokens: tuple[int, ...] = (100, 101)) -> None:
        self.passing_tokens = passing_tokens
        self.batches: list[int] = []

    def tasks(self) -> list[Task]:
        return TASKS

    def log_reward(self, task: Task, completion: str) -> RewardResult:
        passed = any(f"return {t}\n" in completion + "\n" for t in self.passing_tokens)
        return RewardResult(
            log_reward=0.0 if passed else -2.0,
            pass_fraction=1.0 if passed else 0.25,
            passed=passed,
            error=None if passed else "assertion",
        )

    def batch_log_reward(self, pairs: Any) -> list[RewardResult]:
        pairs = list(pairs)
        self.batches.append(len(pairs))
        return [self.log_reward(task, text) for task, text in pairs]


class RecordingLogger:
    def __init__(self) -> None:
        self.steps: list[tuple[int, dict[str, float]]] = []
        self.texts: list[str] = []
        self.objects: list[Any] = []
        self.closed = 0

    def log(self, step: int, metrics: Any) -> None:
        self.steps.append((step, dict(metrics)))

    def log_text(self, message: str) -> None:
        self.texts.append(message)

    def log_object(self, renderable: Any) -> None:
        self.objects.append(renderable)

    def close(self) -> None:
        self.closed += 1


def make_cfg(**train_kwargs: Any) -> RootConfig:
    cfg = RootConfig()
    cfg.seed = 0
    cfg.model = MODEL_CFG
    cfg.cost = CostConfig()
    train = TrainConfig(
        steps=2,
        groups_per_step=2,
        group_size=2,
        max_tokens=32,
        policy_lr=1e-4,
        flow_lr=1e-1,
        lr_schedule="constant",
        warmup_steps=0,
        eval_every=0,
        checkpoint_every=0,
        on_policy_only=False,
        replay=ReplayConfig(enabled=False, fraction=0.0),
    )
    for key, value in train_kwargs.items():
        assert hasattr(train, key), f"TrainConfig has no field {key!r}"
        setattr(train, key, value)
    cfg.train = train
    return cfg


def make_trainer(
    cfg: RootConfig | None = None,
    backend: FakeBackend | None = None,
    objective: Any = None,
    env: FakeEnv | None = None,
    logger: RecordingLogger | None = None,
) -> tuple[Trainer, FakeBackend, RecordingLogger]:
    cfg = make_cfg() if cfg is None else cfg
    backend = FakeBackend() if backend is None else backend
    env = FakeEnv() if env is None else env
    logger = RecordingLogger() if logger is None else logger
    if objective is None:
        objective = VarGrad(name="vargrad")
    objective.register_tasks([t.task_id for t in TASKS])
    trainer = Trainer(
        cfg,
        backend,
        env,
        objective,
        FakeTokenizer(),
        run_logger=logger,
        rng=random.Random(0),
    )
    return trainer, backend, logger


# ------------------------------------------------------------------------------- tests


class TestConfigValidation:
    def test_on_policy_only_with_replay_is_refused(self) -> None:
        cfg = make_cfg(on_policy_only=True, replay=ReplayConfig(enabled=True, fraction=0.25))
        with pytest.raises(ValueError) as excinfo:
            validate_train_config(cfg)
        message = str(excinfo.value)
        assert "on_policy_only" in message
        assert "replay" in message
        # Both escape hatches must be spelled out; a bare "incompatible" is not a fix.
        assert "train.replay.enabled=false" in message
        assert "train.on_policy_only=false" in message

    def test_the_refusal_happens_before_a_trainer_exists(self) -> None:
        cfg = make_cfg(on_policy_only=True, replay=ReplayConfig(enabled=True, fraction=0.25))
        with pytest.raises(ValueError):
            Trainer(cfg, FakeBackend(), FakeEnv(), VarGrad(), FakeTokenizer())

    def test_either_flag_alone_is_fine(self) -> None:
        validate_train_config(make_cfg(on_policy_only=True))
        validate_train_config(make_cfg(replay=ReplayConfig(enabled=True, fraction=0.25)))

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"steps": 0}, "steps"),
            ({"groups_per_step": 0}, "groups_per_step"),
            ({"group_size": 0}, "group_size"),
            ({"lr_schedule": "exponential"}, "lr_schedule"),
            ({"warmup_steps": -1}, "warmup_steps"),
            ({"policy_lr": 0.0}, "policy_lr"),
        ],
    )
    def test_out_of_range_values_are_rejected(self, kwargs: dict[str, Any], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            validate_train_config(make_cfg(**kwargs))

    def test_a_fully_replayed_batch_is_rejected(self) -> None:
        cfg = make_cfg(replay=ReplayConfig(enabled=True, fraction=1.0))
        with pytest.raises(ValueError, match=r"replay\.fraction"):
            validate_train_config(cfg)

    def test_the_shipped_profiles_validate(self) -> None:
        # conf/train/default.yaml and conf/train/smoke.yaml, transcribed.
        validate_train_config(make_cfg(replay=ReplayConfig(enabled=True, fraction=0.25)))
        validate_train_config(
            make_cfg(on_policy_only=True, replay=ReplayConfig(enabled=False, fraction=0.0))
        )


class TestLearningRateSchedule:
    def test_warmup_ramps_linearly_to_the_base_lr(self) -> None:
        values = [learning_rate(s, 1.0, "cosine", warmup_steps=4, steps=100) for s in range(4)]
        assert values == pytest.approx([0.25, 0.5, 0.75, 1.0])

    def test_warmup_never_returns_zero(self) -> None:
        # Tinker rejects a non-positive learning rate outright.
        assert learning_rate(0, 1e-5, "linear", warmup_steps=10, steps=100) > 0.0

    def test_constant_is_constant(self) -> None:
        values = [learning_rate(s, 3e-5, "constant", 0, 10) for s in range(10)]
        assert values == pytest.approx([3e-5] * 10)

    def test_linear_decays_to_zero_at_the_end(self) -> None:
        assert learning_rate(0, 1.0, "linear", 0, 10) == pytest.approx(1.0)
        assert learning_rate(5, 1.0, "linear", 0, 10) == pytest.approx(0.5)
        assert learning_rate(9, 1.0, "linear", 0, 10) == pytest.approx(0.1)

    def test_cosine_halves_at_the_midpoint(self) -> None:
        assert learning_rate(0, 1.0, "cosine", 0, 10) == pytest.approx(1.0)
        assert learning_rate(5, 1.0, "cosine", 0, 10) == pytest.approx(0.5)
        assert learning_rate(9, 1.0, "cosine", 0, 10) == pytest.approx(
            0.5 * (1 + math.cos(math.pi * 0.9))
        )

    def test_decay_starts_after_warmup_not_at_step_zero(self) -> None:
        assert learning_rate(10, 1.0, "linear", warmup_steps=10, steps=20) == pytest.approx(1.0)
        assert learning_rate(15, 1.0, "linear", warmup_steps=10, steps=20) == pytest.approx(0.5)

    @pytest.mark.parametrize("schedule", LR_SCHEDULES)
    def test_every_schedule_is_positive_on_every_step(self, schedule: str) -> None:
        values = [learning_rate(s, 1e-5, schedule, 3, 20) for s in range(20)]
        assert all(v > 0.0 for v in values)

    def test_unknown_schedule_raises(self) -> None:
        with pytest.raises(ValueError, match="lr_schedule"):
            learning_rate(0, 1.0, "sqrt", 0, 10)


class TestOneStep:
    async def test_a_step_runs_end_to_end(self) -> None:
        trainer, backend, _logger = make_trainer()
        metrics = await trainer.step()
        assert "loss" in metrics
        assert metrics["batch/size"] == 4.0
        assert metrics["lr"] == pytest.approx(1e-4)
        assert metrics["tokens_total"] > 0.0
        assert "usd_total" in metrics
        assert backend.log == [
            "sync_sampler",
            "sample",
            "compute_logprobs",
            "apply_gradient",
            "optim_step",
        ]

    async def test_gradient_is_pushed_before_the_optimiser_steps(self) -> None:
        # Both must land in one server clock cycle, which is what defer=True buys.
        trainer, backend, _logger = make_trainer()
        await trainer.step()
        assert backend.log.index("apply_gradient") < backend.log.index("optim_step")
        assert backend.apply_calls[0]["defer"] is True

    async def test_grad_scale_is_one_over_the_batch(self) -> None:
        # Tinker sums per-datum losses; this is the only place a denominator appears.
        trainer, backend, _logger = make_trainer()
        await trainer.step()
        call = backend.apply_calls[0]
        assert call["grad_scale"] == pytest.approx(1.0 / call["batch_size"])

    async def test_gradients_are_finite_and_correctly_shaped(self) -> None:
        trainer, backend, _logger = make_trainer()
        await trainer.step()
        grads = backend.apply_calls[0]["grads"]
        assert len(grads) == backend.apply_calls[0]["batch_size"]
        assert all(torch.isfinite(g).all() for g in grads)
        assert all(g.ndim == 1 for g in grads)

    async def test_the_learning_rate_schedule_reaches_optim_step(self) -> None:
        cfg = make_cfg(steps=10, lr_schedule="linear", warmup_steps=0, policy_lr=1.0)
        trainer, backend, _logger = make_trainer(cfg=cfg)
        await trainer.step()
        await trainer.step()
        assert [c["lr"] for c in backend.optim_calls] == pytest.approx([1.0, 0.9])

    async def test_grad_clip_norm_is_forwarded(self) -> None:
        trainer, backend, _logger = make_trainer(cfg=make_cfg(grad_clip_norm=0.5))
        await trainer.step()
        assert backend.optim_calls[0]["grad_clip_norm"] == pytest.approx(0.5)

    async def test_sampler_sync_respects_its_period(self) -> None:
        cfg = make_cfg(sync_sampler_every=2, steps=4)
        trainer, backend, _logger = make_trainer(cfg=cfg)
        for _ in range(4):
            await trainer.step()
        assert backend.log.count("sync_sampler") == 2

    async def test_batch_metrics_report_pass_rate_and_diversity(self) -> None:
        # The fake env passes two of the four completions, and each decodes to a different
        # program, so both should show up as distinct modes.
        trainer, _backend, _logger = make_trainer()
        metrics = await trainer.step()
        assert metrics["pass_rate"] == pytest.approx(0.5)
        assert metrics["diversity/distinct_passing_solutions"] == 2.0
        assert metrics["error/assertion"] == pytest.approx(0.5)

    async def test_an_all_degenerate_step_is_skipped_not_fatal(self) -> None:
        class EmptyBackend(FakeBackend):
            async def sample(
                self, prompts: Any, num_samples: int, sampling_params: SamplingParams
            ) -> list[SampleResponse]:
                self.log.append("sample")
                return [
                    SampleResponse(
                        sequences=[
                            SampledSequence(
                                stop_reason="stop",
                                tokens_np=np.array([], dtype=np.int32),
                                logprobs_np=np.array([], dtype=np.float32),
                            )
                            for _ in range(num_samples)
                        ],
                        prompt_cache_hit_tokens=0,
                    )
                    for _ in prompts
                ]

        trainer, backend, _logger = make_trainer(backend=EmptyBackend())
        metrics = await trainer.step()
        assert metrics["skipped"] == 1.0
        assert "apply_gradient" not in backend.log


class TestOnPolicyOnly:
    async def test_the_oracle_forward_is_skipped(self) -> None:
        trainer, backend, _logger = make_trainer(cfg=make_cfg(on_policy_only=True))
        await trainer.step()
        assert "compute_logprobs" not in backend.log
        assert backend.usage.num_forward_passes == 0

    async def test_the_gradient_is_still_pushed(self) -> None:
        trainer, backend, _logger = make_trainer(cfg=make_cfg(on_policy_only=True))
        await trainer.step()
        assert backend.apply_calls[0]["defer"] is True
        assert backend.optim_calls


class TestFlowParameters:
    async def test_flow_parameters_are_actually_stepped(self) -> None:
        objective = TrajectoryBalance(name="tb")
        objective.register_tasks([t.task_id for t in TASKS])
        before = [p.detach().clone() for p in objective.parameters()]
        assert before, "TrajectoryBalance is supposed to own client-side log Z parameters"

        trainer, _backend, _logger = make_trainer(objective=objective)
        await trainer.step()

        after = list(objective.parameters())
        assert any(not torch.equal(a, b) for a, b in zip(after, before, strict=True))

    async def test_the_flow_optimiser_uses_flow_lr(self) -> None:
        objective = TrajectoryBalance(name="tb")
        objective.register_tasks([t.task_id for t in TASKS])
        cfg = make_cfg(flow_lr=0.123)
        trainer, _backend, _logger = make_trainer(cfg=cfg, objective=objective)
        assert trainer.flow_optimizer is not None
        assert trainer.flow_optimizer.param_groups[0]["lr"] == pytest.approx(0.123)

    async def test_flow_gradients_do_not_accumulate_across_steps(self) -> None:
        objective = TrajectoryBalance(name="tb")
        objective.register_tasks([t.task_id for t in TASKS])
        trainer, _backend, _logger = make_trainer(objective=objective)
        await trainer.step()
        first = [
            None if p.grad is None else p.grad.detach().clone() for p in objective.parameters()
        ]
        await trainer.step()
        second = [
            None if p.grad is None else p.grad.detach().clone() for p in objective.parameters()
        ]
        # Not a strict inequality on every tensor; the point is the optimiser zeroes grads,
        # so step two's gradient is not step one's plus something.
        assert any(
            a is None or b is None or not torch.equal(a, b)
            for a, b in zip(first, second, strict=True)
        )

    async def test_an_objective_without_parameters_has_no_optimiser(self) -> None:
        trainer, _backend, _logger = make_trainer(objective=VarGrad())
        assert trainer.flow_optimizer is None
        metrics = await trainer.step()
        assert metrics["flow/num_params"] == 0.0

    async def test_flow_grad_norm_is_reported_when_clipping(self) -> None:
        objective = TrajectoryBalance(name="tb")
        objective.register_tasks([t.task_id for t in TASKS])
        trainer, _backend, _logger = make_trainer(
            cfg=make_cfg(grad_clip_norm=1.0), objective=objective
        )
        metrics = await trainer.step()
        assert metrics["flow/grad_norm"] >= 0.0


class TestReplayMixing:
    def _cfg(self, fraction: float = 0.5) -> RootConfig:
        return make_cfg(replay=ReplayConfig(enabled=True, capacity=100, fraction=fraction))

    async def test_the_first_step_has_nothing_to_replay(self) -> None:
        # The buffer is sampled BEFORE the fresh rollouts are banked, so step 0 is pure
        # on-policy data rather than a batch containing copies of itself.
        trainer, _backend, _logger = make_trainer(cfg=self._cfg())
        metrics = await trainer.step()
        assert metrics["batch/num_replayed"] == 0.0
        assert trainer.replay is not None
        assert len(trainer.replay) == 4

    async def test_later_steps_mix_in_replayed_trajectories(self) -> None:
        trainer, _backend, _logger = make_trainer(cfg=self._cfg(fraction=0.5))
        await trainer.step()
        metrics = await trainer.step()
        # f=0.5 => replayed == fresh, so the batch doubles.
        assert metrics["batch/num_fresh"] == 4.0
        assert metrics["batch/num_replayed"] == 4.0
        assert metrics["batch/size"] == 8.0

    async def test_the_replayed_share_matches_the_configured_fraction(self) -> None:
        trainer, _backend, _logger = make_trainer(cfg=self._cfg(fraction=0.25))
        await trainer.step()
        metrics = await trainer.step()
        share = metrics["batch/num_replayed"] / metrics["batch/size"]
        assert share == pytest.approx(0.25, abs=0.05)

    async def test_replay_stats_are_logged(self) -> None:
        trainer, _backend, _logger = make_trainer(cfg=self._cfg())
        metrics = await trainer.step()
        assert metrics["replay/size"] == 4.0
        assert metrics["replay/capacity"] == 100.0

    async def test_replayed_trajectories_are_rescored_by_the_current_policy(self) -> None:
        # The whole reason replay is legal here: compute_logprobs is called on the FULL
        # batch, so the stale sampling_logprobs are never used as current-policy logprobs.
        trainer, backend, _logger = make_trainer(cfg=self._cfg())
        await trainer.step()
        await trainer.step()
        assert backend.apply_calls[-1]["batch_size"] == 8
        assert backend.log.count("compute_logprobs") == 2

    async def test_no_buffer_when_replay_is_disabled(self) -> None:
        trainer, _backend, _logger = make_trainer()
        assert trainer.replay is None


class TestEval:
    async def test_eval_is_greedy_and_single_sample(self) -> None:
        trainer, backend, _logger = make_trainer()
        metrics = await trainer.evaluate()
        num_samples, params = backend.sample_calls[-1]
        assert num_samples == 1
        assert params.temperature == 0.0
        assert 0.0 <= metrics["eval/pass_rate"] <= 1.0
        assert metrics["eval/num_tasks"] == float(len(trainer.eval_tasks))

    async def test_eval_without_a_held_out_split_is_empty(self) -> None:
        trainer, _backend, _logger = make_trainer()
        trainer.eval_tasks = []
        assert await trainer.evaluate() == {}

    async def test_eval_runs_on_its_period(self) -> None:
        cfg = make_cfg(steps=4, eval_every=2)
        trainer, backend, logger = make_trainer(cfg=cfg)
        await trainer.run()
        # Two evals, each one extra sample() call on top of the four training ones.
        assert backend.log.count("sample") == 6
        assert any("eval/pass_rate" in metrics for _step, metrics in logger.steps)

    def test_train_and_eval_tasks_are_disjoint(self) -> None:
        trainer, _backend, _logger = make_trainer()
        train_ids = {t.task_id for t in trainer.train_tasks}
        eval_ids = {t.task_id for t in trainer.eval_tasks}
        assert train_ids and eval_ids
        assert not (train_ids & eval_ids)

    def test_the_split_is_stable_across_trainers(self) -> None:
        a, _b1, _l1 = make_trainer()
        b, _b2, _l2 = make_trainer()
        assert [t.task_id for t in a.eval_tasks] == [t.task_id for t in b.eval_tasks]


class TestCheckpoints:
    async def test_checkpointing_returns_the_saved_path(self) -> None:
        client = FakeTrainingClient(path="tinker://weights/step-2")
        trainer, _backend, _logger = make_trainer(backend=FakeBackend(training_client=client))
        path = await trainer.save_checkpoint(2)
        assert path == "tinker://weights/step-2"
        assert client.saved and "000002" in client.saved[0]

    async def test_a_backend_that_cannot_checkpoint_is_not_fatal(self) -> None:
        trainer, _backend, _logger = make_trainer()
        assert await trainer.save_checkpoint(1) is None

    async def test_a_failing_save_is_not_fatal(self) -> None:
        class Exploding(FakeTrainingClient):
            async def save_weights_for_sampler_async(
                self, name: str, ttl_seconds: int | None = None
            ) -> Any:
                raise RuntimeError("checkpoint service unavailable")

        trainer, _backend, _logger = make_trainer(backend=FakeBackend(training_client=Exploding()))
        assert await trainer.save_checkpoint(3) is None

    async def test_checkpoints_run_on_their_period(self) -> None:
        client = FakeTrainingClient()
        cfg = make_cfg(steps=4, checkpoint_every=2)
        trainer, _backend, _logger = make_trainer(
            cfg=cfg, backend=FakeBackend(training_client=client)
        )
        await trainer.run()
        assert len(client.saved) == 2


class TestRun:
    async def test_run_executes_every_step(self) -> None:
        cfg = make_cfg(steps=3)
        trainer, backend, logger = make_trainer(cfg=cfg)
        await trainer.run()
        assert trainer.step_index == 3
        assert backend.log.count("optim_step") == 3
        assert len(logger.steps) == 3

    async def test_log_every_throttles_the_logger(self) -> None:
        cfg = make_cfg(steps=4, log_every=2)
        trainer, _backend, logger = make_trainer(cfg=cfg)
        await trainer.run()
        assert [step for step, _metrics in logger.steps] == [0, 2]

    async def test_an_eval_is_logged_even_on_a_throttled_step(self) -> None:
        # log_every=4 would skip step 1, which is exactly where eval_every=2 lands.
        cfg = make_cfg(steps=2, log_every=4, eval_every=2)
        trainer, _backend, logger = make_trainer(cfg=cfg)
        await trainer.run()
        assert any("eval/pass_rate" in metrics for _step, metrics in logger.steps)

    async def test_the_budget_is_printed_up_front(self) -> None:
        trainer, _backend, logger = make_trainer()
        await trainer.run()
        titles = [getattr(o, "title", "") for o in logger.objects]
        assert any("estimated cost" in str(t) for t in titles)

    async def test_the_spend_is_reported_even_when_the_run_dies(self) -> None:
        # A run that falls over at step 2 of 1000 has still spent money, and that number
        # must not vanish with the traceback.
        cfg = make_cfg(steps=5)
        backend = FakeBackend(fail_at_step=2)
        trainer, _backend, logger = make_trainer(cfg=cfg, backend=backend)
        with pytest.raises(RuntimeError, match="fell over"):
            await trainer.run()
        titles = [str(getattr(o, "title", "")) for o in logger.objects]
        assert any("actual spend" in t for t in titles)
        assert logger.closed == 1

    async def test_the_final_report_compares_against_the_estimate(self) -> None:
        trainer, _backend, logger = make_trainer()
        await trainer.run()
        assert any("the estimate predicted" in text for text in logger.texts)

    def test_spend_and_budget_are_priced_with_the_model(self) -> None:
        trainer, _backend, _logger = make_trainer()
        assert trainer.budget().model == MODEL_CFG.name
        assert trainer.spend().model == MODEL_CFG.name


class TestLoggers:
    """The metrics sinks the loop writes through."""

    def test_none_is_a_real_no_op(self) -> None:
        sink = make_logger("none")
        sink.log(0, {"loss": 1.0})
        sink.log_text("hello")
        sink.log_object(object())
        sink.close()

    def test_rich_prints_the_six_columns_that_matter(self) -> None:
        console = Console(file=io.StringIO(), width=160, no_color=True)
        sink = RichLogger(console=console)
        sink.log(
            3,
            {
                "loss": 0.125,
                "log_reward_mean": -1.5,
                "log_reward_max": -0.25,
                "pass_rate": 0.5,
                "log_z_mean": 2.0,
                "tokens_total": 12345.0,
                "usd_total": 0.4212,
            },
        )
        stream = console.file
        assert isinstance(stream, io.StringIO)
        output = stream.getvalue()
        for expected in ("loss", "logR mean", "pass", "logZ", "tokens", "0.1250", "50.00%"):
            assert expected in output

    def test_rich_prints_the_header_once(self) -> None:
        console = Console(file=io.StringIO(), width=160, no_color=True)
        sink = RichLogger(console=console)
        for step in range(3):
            sink.log(step, {"loss": float(step)})
        stream = console.file
        assert isinstance(stream, io.StringIO)
        assert stream.getvalue().count("logR mean") == 1

    def test_rich_renders_missing_metrics_as_a_dash(self) -> None:
        console = Console(file=io.StringIO(), width=160, no_color=True)
        RichLogger(console=console).log(0, {"loss": 1.0})
        stream = console.file
        assert isinstance(stream, io.StringIO)
        assert "-" in stream.getvalue()

    def test_a_vargrad_run_reports_its_in_batch_log_z(self) -> None:
        # The two objectives name the column differently (log_z_mean vs
        # log_z_estimate_mean); the table has to accept either.
        console = Console(file=io.StringIO(), width=160, no_color=True)
        RichLogger(console=console).log(0, {"log_z_estimate_mean": -3.5})
        stream = console.file
        assert isinstance(stream, io.StringIO)
        assert "-3.500" in stream.getvalue()

    def test_unknown_kind_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown logger"):
            make_logger("tensorboard")

    def test_wandb_without_the_extra_says_what_to_install(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "wandb", None)
        with pytest.raises(ImportError, match="uv sync --extra wandb"):
            make_logger("wandb")

    def test_make_logger_returns_the_configured_kinds(self) -> None:
        assert isinstance(make_logger("rich"), RichLogger)
        assert "none" in LOGGER_KINDS


class TestComposition:
    """The parts of :func:`flowcode.train.train` that do not need an API key.

    Instantiating from ``_target_`` is the seam where a config edit and the code drift
    apart, and it costs nothing to check offline: ``CodeExecEnv`` defers task loading out
    of ``__init__`` and the objectives are plain torch.
    """

    def compose(self, *overrides: str) -> RootConfig:
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra

        from flowcode.cli import CONFIG_PATH

        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=CONFIG_PATH, version_base="1.3"):
            cfg = compose(config_name="config", overrides=list(overrides))
        return cast(RootConfig, cfg)

    @pytest.mark.parametrize("name", ["tb", "subtb", "db", "vargrad"])
    def test_every_objective_instantiates_and_satisfies_the_protocol(self, name: str) -> None:
        objective = _instantiate_objective(self.compose(f"objective={name}"))
        assert objective.name == name
        objective.register_tasks(["a", "b"])
        assert isinstance(list(objective.parameters()), list)

    @pytest.mark.parametrize("name", ["fixtures", "mbpp", "humaneval"])
    def test_every_env_instantiates_without_loading_tasks(self, name: str) -> None:
        # Loading is deferred out of __init__ on purpose; mbpp/humaneval would need the
        # network here otherwise.
        env = _instantiate_env(self.compose(f"env={name}"))
        assert env.name == name

    def test_the_fixtures_env_yields_tasks(self) -> None:
        env = _instantiate_env(self.compose("env=fixtures"))
        assert len(env.tasks()) > 0

    def test_a_non_objective_target_is_rejected(self) -> None:
        cfg = self.compose()
        cfg.objective = {"_target_": "flowcode.types.TokenUsage"}
        with pytest.raises(TypeError, match="Objective protocol"):
            _instantiate_objective(cfg)

    def test_a_non_environment_target_is_rejected(self) -> None:
        cfg = self.compose()
        cfg.env = {"_target_": "flowcode.types.TokenUsage"}
        with pytest.raises(TypeError, match="Environment protocol"):
            _instantiate_env(cfg)

    def test_the_tokenizer_comes_off_the_training_client(self) -> None:
        backend = FakeBackend(training_client=FakeTrainingClient())
        assert isinstance(_tokenizer_of(backend), FakeTokenizer)

    def test_a_backend_without_a_tokenizer_says_so(self) -> None:
        with pytest.raises(RuntimeError, match="get_tokenizer"):
            _tokenizer_of(FakeBackend())


@pytest.mark.tinker
class TestAgainstTheLiveApi:
    """Costs real money; deselected by default (see pyproject's addopts)."""

    async def test_a_smoke_run_completes(self) -> None:
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra

        from flowcode.cli import CONFIG_PATH
        from flowcode.train import train

        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=CONFIG_PATH, version_base="1.3"):
            cfg = compose(config_name="config", overrides=["train=smoke", "env=fixtures"])
        await train(cast(RootConfig, cfg))
