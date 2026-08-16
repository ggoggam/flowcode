"""The padding/alignment contract and the gradient sign, offline.

These are the two things in flowcode that are both invisible and fatal. An off-by-one in
the padding trains the model on the wrong tokens; a flipped sign trains it to *minimise*
the reward. Neither produces an exception — both produce a run that burns money and
converges to nonsense. So both are asserted exactly, against a fake client that returns
canned :class:`~tinker.types.TensorData` and never touches the network.

The sign, restated because it is the whole module: Tinker's backend computes
``L = sum(-target_logprobs * weights)``, so ``dL/dlogprobs = -weights``. To make the
server deposit ``dC/dtheta`` for a client-side loss ``C``, send ``weights = -dC/dlogprobs``.
That is the rule stated at ``training_client.py:475`` in the SDK, and
``TestGradientSign`` is what stops someone "simplifying" the minus sign away.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from tinker import types as tinker_types
from tinker.types import (
    Datum,
    ForwardBackwardOutput,
    ModelInput,
    OptimStepResponse,
    SampledSequence,
    SampleResponse,
)
from tinker.types import SamplingParams as TinkerSamplingParams

from flowcode.alignment import observation_length
from flowcode.config import ModelConfig
from flowcode.tinker_backend import (
    SamplingClientLike,
    TinkerBackend,
    TrainingClientLike,
    build_datum,
    build_model_input,
    from_tinker_response,
    from_tinker_sequence,
    to_tinker_sampling_params,
)
from flowcode.types import SamplingParams, Segment, TokenUsage, Trajectory

PROMPT = [101, 102, 103, 104]  # P = 4, so ob_len = 3
COMPLETION = [201, 202, 203, 204, 205]  # N = 5


def make_trajectory(
    task_id: str = "t0",
    prompt: list[int] | None = None,
    completion: list[int] | None = None,
    log_reward: float = -2.0,
) -> Trajectory:
    prompt = PROMPT if prompt is None else prompt
    completion = COMPLETION if completion is None else completion
    return Trajectory(
        task_id=task_id,
        prompt_tokens=list(prompt),
        completion_tokens=list(completion),
        sampling_logprobs=[-0.5] * len(completion),
        log_reward=log_reward,
        segments=[Segment(0, len(completion))],
    )


MODEL_CFG = ModelConfig(name="Qwen/Qwen3-8B", renderer="qwen3")


# ---------------------------------------------------------------------------- fakes


class FakeFuture[T]:
    """Stands in for ``tinker.APIFuture``: submitted already, awaited separately."""

    def __init__(self, value: T, log: list[str], label: str) -> None:
        self._value = value
        self._log = log
        self._label = label

    async def result_async(self, timeout: float | None = None) -> T:
        self._log.append(f"await:{self._label}")
        return self._value


def canned_logprobs(length: int, offset: float = 0.0) -> tinker_types.TensorData:
    """A predictable per-position ramp: position i scores ``-(i + 1) / 10 + offset``."""
    values = [-(i + 1) / 10.0 + offset for i in range(length)]
    return tinker_types.TensorData(data=values, dtype="float32", shape=[length])


class FakeTrainingClient:
    """Records every submitted request and never touches the network."""

    def __init__(self) -> None:
        self.log: list[str] = []
        self.forward_data: list[list[Datum]] = []
        self.backward_data: list[list[Datum]] = []
        self.optim_params: list[tinker_types.AdamParams] = []
        self.sampler_saves = 0
        self.sampling_client = FakeSamplingClient()

    def _output(self, data: list[Datum]) -> ForwardBackwardOutput:
        return ForwardBackwardOutput(
            loss_fn_output_type="ArrayRecord",
            loss_fn_outputs=[{"logprobs": canned_logprobs(d.model_input.length)} for d in data],
            metrics={"loss:sum": 1.25},
        )

    async def forward_async(
        self,
        data: list[Datum],
        loss_fn: tinker_types.LossFnType,
        loss_fn_config: dict[str, float] | None = None,
    ) -> FakeFuture[ForwardBackwardOutput]:
        assert loss_fn == "cross_entropy"
        self.forward_data.append(data)
        self.log.append("submit:forward")
        return FakeFuture(self._output(data), self.log, "forward")

    async def forward_backward_async(
        self,
        data: list[Datum],
        loss_fn: tinker_types.LossFnType,
        loss_fn_config: dict[str, float] | None = None,
    ) -> FakeFuture[ForwardBackwardOutput]:
        assert loss_fn == "cross_entropy"
        self.backward_data.append(data)
        self.log.append("submit:backward")
        return FakeFuture(self._output(data), self.log, "backward")

    async def optim_step_async(
        self, adam_params: tinker_types.AdamParams
    ) -> FakeFuture[OptimStepResponse]:
        self.optim_params.append(adam_params)
        self.log.append("submit:optim")
        return FakeFuture(OptimStepResponse(), self.log, "optim")

    async def save_weights_and_get_sampling_client_async(self) -> SamplingClientLike:
        self.sampler_saves += 1
        self.sampling_client = FakeSamplingClient()
        return self.sampling_client


class FakeSamplingClient:
    """Returns two four-token completions per request, with a fixed cache-hit count."""

    def __init__(self, cache_hit_tokens: int = 7, completion_length: int = 4) -> None:
        self.cache_hit_tokens = cache_hit_tokens
        self.completion_length = completion_length
        self.calls: list[tuple[list[int], int]] = []

    async def sample_async(
        self,
        prompt: ModelInput,
        num_samples: int,
        sampling_params: TinkerSamplingParams,
    ) -> SampleResponse:
        self.calls.append((prompt.to_ints(), num_samples))
        sequences = [
            SampledSequence(
                stop_reason="stop",
                tokens_np=np.arange(self.completion_length, dtype=np.int32),
                logprobs_np=np.full(self.completion_length, -0.25, dtype=np.float32),
            )
            for _ in range(num_samples)
        ]
        return SampleResponse(sequences=sequences, prompt_cache_hit_tokens=self.cache_hit_tokens)


def make_backend() -> tuple[TinkerBackend, FakeTrainingClient]:
    training: TrainingClientLike = FakeTrainingClient()
    assert isinstance(training, FakeTrainingClient)
    sampling: SamplingClientLike = training.sampling_client
    return TinkerBackend(training, sampling, MODEL_CFG), training


def backward(tensor: torch.Tensor) -> None:
    """``Tensor.backward`` is unannotated in the shipped torch stubs; isolate the ignore."""
    tensor.backward()  # type: ignore[no-untyped-call]


def weights_of(datum: Datum) -> list[float]:
    return [float(x) for x in datum.loss_fn_inputs["weights"].tolist()]


def targets_of(datum: Datum) -> list[int]:
    return [int(x) for x in datum.loss_fn_inputs["target_tokens"].tolist()]


# --------------------------------------------------------------- datum construction
#
# The alignment arithmetic itself is backend-agnostic and lives in tests/test_alignment.py.
# What is asserted here is only how a Datum is assembled out of it.


class TestDatumConstruction:
    def test_model_input_is_prompt_plus_completion_minus_last(self) -> None:
        assert build_model_input(make_trajectory()).to_ints() == PROMPT + COMPLETION[:-1]

    def test_three_lengths_agree(self) -> None:
        traj = make_trajectory()
        datum = build_datum(traj, [1.0] * len(COMPLETION))
        assert datum.model_input.length == len(targets_of(datum)) == len(weights_of(datum))

    def test_oracle_pass_uses_all_zero_weights(self) -> None:
        # Mirrors _get_custom_loss_forward_data: zero weights make the loss identically 0
        # while the forward still returns per-position logprobs.
        datum = build_datum(make_trajectory(), weights=None)
        assert weights_of(datum) == [0.0] * datum.model_input.length

    def test_weights_are_left_padded_with_zeros(self) -> None:
        datum = build_datum(make_trajectory(), [1.0, 2.0, 3.0, 4.0, 5.0])
        assert weights_of(datum) == [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    def test_single_token_completion(self) -> None:
        traj = make_trajectory(prompt=[7, 8, 9], completion=[42])
        datum = build_datum(traj, [1.0])
        assert datum.model_input.to_ints() == [7, 8, 9]
        assert targets_of(datum) == [0, 0, 42]
        assert weights_of(datum) == [0.0, 0.0, 1.0]

    def test_two_token_prompt(self) -> None:
        traj = make_trajectory(prompt=[7, 8], completion=[42, 43])
        datum = build_datum(traj, [1.0, 2.0])
        assert datum.model_input.to_ints() == [7, 8, 42]
        assert targets_of(datum) == [0, 42, 43]
        assert weights_of(datum) == [0.0, 1.0, 2.0]

    def test_rejects_wrong_length_weights(self) -> None:
        with pytest.raises(ValueError, match="do not pre-pad"):
            build_datum(make_trajectory(), [1.0, 2.0])


# ------------------------------------------------------------------- compute_logprobs


class TestComputeLogprobs:
    async def test_returns_completion_only_shape(self) -> None:
        backend, _ = make_backend()
        (logprobs,) = await backend.compute_logprobs([make_trajectory()])
        assert logprobs.shape == (len(COMPLETION),)

    async def test_slices_off_the_prompt_padding(self) -> None:
        backend, _training = make_backend()
        traj = make_trajectory()
        (logprobs,) = await backend.compute_logprobs([traj])
        full = canned_logprobs(build_model_input(traj).length).tolist()
        assert logprobs.tolist() == pytest.approx(full[observation_length(traj) :])
        assert logprobs.tolist()[0] == pytest.approx(-0.4)  # position ob_len == 3

    async def test_result_is_a_differentiable_leaf(self) -> None:
        backend, _ = make_backend()
        (logprobs,) = await backend.compute_logprobs([make_trajectory()])
        assert logprobs.requires_grad
        assert logprobs.is_leaf
        backward(logprobs.sum())
        assert logprobs.grad is not None

    async def test_uses_forward_not_forward_backward(self) -> None:
        backend, training = make_backend()
        await backend.compute_logprobs([make_trajectory()])
        assert training.log == ["submit:forward", "await:forward"]
        assert training.backward_data == []

    async def test_sends_zero_weights(self) -> None:
        backend, training = make_backend()
        await backend.compute_logprobs([make_trajectory()])
        (datum,) = training.forward_data[0]
        assert weights_of(datum) == [0.0] * datum.model_input.length

    async def test_batches_preserve_order(self) -> None:
        backend, _training = make_backend()
        trajs = [
            make_trajectory("a", completion=[1, 2, 3]),
            make_trajectory("b", completion=[1, 2, 3, 4, 5, 6]),
        ]
        results = await backend.compute_logprobs(trajs)
        assert [r.shape[0] for r in results] == [3, 6]

    async def test_empty_batch_rejected(self) -> None:
        backend, _ = make_backend()
        with pytest.raises(ValueError, match="at least one trajectory"):
            await backend.compute_logprobs([])


# --------------------------------------------------------------------- THE SIGN


class TestGradientSign:
    """``weights = -dC/dlogprobs``. Sign included, padding included, exactly."""

    async def test_identity_loss_sends_negative_ones(self) -> None:
        # C = sum(logprobs)  =>  dC/dlogprobs = 1  =>  weights = -1 per completion token.
        backend, training = make_backend()
        traj = make_trajectory()
        (logprobs,) = await backend.compute_logprobs([traj])
        backward(logprobs.sum())
        assert logprobs.grad is not None
        await backend.apply_gradient([traj], [logprobs.grad])

        (datum,) = training.backward_data[0]
        assert weights_of(datum) == [0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0]

    async def test_negated_loss_flips_the_sign(self) -> None:
        # C = -sum(logprobs)  =>  weights = +1. If someone drops the minus in
        # apply_gradient, this test and the one above swap and both fail.
        backend, training = make_backend()
        traj = make_trajectory()
        (logprobs,) = await backend.compute_logprobs([traj])
        backward(-logprobs.sum())
        assert logprobs.grad is not None
        await backend.apply_gradient([traj], [logprobs.grad])

        (datum,) = training.backward_data[0]
        assert weights_of(datum) == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    async def test_weights_are_exactly_minus_the_gradient(self) -> None:
        backend, training = make_backend()
        traj = make_trajectory()
        grad = torch.tensor([0.5, -1.5, 2.0, 0.0, -0.25])
        await backend.apply_gradient([traj], [grad])

        (datum,) = training.backward_data[0]
        assert weights_of(datum) == [0.0, 0.0, 0.0, -0.5, 1.5, -2.0, -0.0, 0.25]

    async def test_trajectory_balance_loss_end_to_end(self) -> None:
        """A real objective's worth of algebra, checked against hand arithmetic.

        C = (log Z + sum log P_F - log R)^2, so dC/dlogprobs_i = 2 * residual for every i,
        and the weight on every completion token is -2 * residual.
        """
        backend, training = make_backend()
        traj = make_trajectory(log_reward=-2.0)
        (logprobs,) = await backend.compute_logprobs([traj])
        log_z = torch.tensor(0.75, requires_grad=True)

        residual = log_z + logprobs.sum() - traj.log_reward
        loss = residual.pow(2)
        backward(loss)

        assert logprobs.grad is not None
        await backend.apply_gradient([traj], [logprobs.grad])

        expected_residual = 0.75 + sum(logprobs.tolist()) - (-2.0)
        expected_weight = -2.0 * expected_residual
        (datum,) = training.backward_data[0]
        weights = weights_of(datum)
        assert weights[:3] == [0.0, 0.0, 0.0]
        assert weights[3:] == pytest.approx([expected_weight] * 5, abs=1e-5)
        # The client-side log Z parameter got its gradient too, from the same backward.
        assert log_z.grad is not None
        assert float(log_z.grad) == pytest.approx(2 * expected_residual, abs=1e-5)

    async def test_per_token_gradients_are_not_smeared(self) -> None:
        # A per-token loss must produce per-token weights, not a broadcast constant.
        backend, training = make_backend()
        traj = make_trajectory()
        (logprobs,) = await backend.compute_logprobs([traj])
        coeffs = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        backward((logprobs * coeffs).sum())
        assert logprobs.grad is not None
        await backend.apply_gradient([traj], [logprobs.grad])

        (datum,) = training.backward_data[0]
        assert weights_of(datum)[3:] == pytest.approx([-1.0, -2.0, -3.0, -4.0, -5.0])

    async def test_target_tokens_survive_the_push(self) -> None:
        backend, training = make_backend()
        traj = make_trajectory()
        await backend.apply_gradient([traj], [torch.ones(len(COMPLETION))])
        (datum,) = training.backward_data[0]
        assert targets_of(datum) == [0, 0, 0, *COMPLETION]
        assert datum.model_input.to_ints() == PROMPT + COMPLETION[:-1]


class TestGradScale:
    async def test_default_does_not_normalize(self) -> None:
        # Tinker sums and accumulates; nothing here divides silently.
        backend, training = make_backend()
        traj = make_trajectory()
        metrics = await backend.apply_gradient([traj], [torch.ones(len(COMPLETION))])
        assert metrics["grad_scale"] == 1.0
        assert weights_of(training.backward_data[0][0])[3:] == [-1.0] * 5

    async def test_scale_is_applied_and_reported(self) -> None:
        backend, training = make_backend()
        traj = make_trajectory()
        metrics = await backend.apply_gradient(
            [traj], [torch.ones(len(COMPLETION))], grad_scale=0.25
        )
        assert metrics["grad_scale"] == 0.25
        assert weights_of(training.backward_data[0][0])[3:] == [-0.25] * 5

    async def test_denominator_inputs_are_reported(self) -> None:
        backend, _ = make_backend()
        trajs = [make_trajectory("a", completion=[1, 2, 3]), make_trajectory("b")]
        metrics = await backend.apply_gradient(trajs, [torch.ones(3), torch.ones(5)])
        assert metrics["num_trajectories"] == 2.0
        assert metrics["num_completion_tokens"] == 8.0

    async def test_server_metrics_are_merged_in(self) -> None:
        backend, _ = make_backend()
        metrics = await backend.apply_gradient([make_trajectory()], [torch.ones(5)])
        assert metrics["loss:sum"] == 1.25


class TestApplyGradientValidation:
    async def test_empty_batch_rejected(self) -> None:
        backend, _ = make_backend()
        with pytest.raises(ValueError, match="at least one trajectory"):
            await backend.apply_gradient([], [])

    async def test_length_mismatch_rejected(self) -> None:
        backend, _ = make_backend()
        with pytest.raises(ValueError, match="one-to-one"):
            await backend.apply_gradient([make_trajectory()], [])

    async def test_wrong_gradient_shape_rejected(self) -> None:
        backend, _ = make_backend()
        with pytest.raises(ValueError, match="unpadded"):
            await backend.apply_gradient([make_trajectory()], [torch.ones(8)])

    async def test_padded_gradient_is_rejected_not_silently_accepted(self) -> None:
        # Handing back a full-length (already padded) gradient is the likely mistake.
        backend, _ = make_backend()
        traj = make_trajectory()
        padded = torch.ones(build_model_input(traj).length)
        with pytest.raises(ValueError, match="unpadded"):
            await backend.apply_gradient([traj], [padded])

    async def test_nan_gradient_rejected(self) -> None:
        backend, _ = make_backend()
        grad = torch.tensor([1.0, float("nan"), 1.0, 1.0, 1.0])
        with pytest.raises(ValueError, match="reward_floor"):
            await backend.apply_gradient([make_trajectory()], [grad])

    async def test_inf_gradient_rejected(self) -> None:
        backend, _ = make_backend()
        grad = torch.tensor([1.0, float("inf"), 1.0, 1.0, 1.0])
        with pytest.raises(ValueError, match="reward_floor"):
            await backend.apply_gradient([make_trajectory()], [grad])


# ------------------------------------------------------------ submission ordering


class TestSubmissionOrdering:
    async def test_deferred_push_and_optim_share_a_clock_cycle(self) -> None:
        # Both requests must be SUBMITTED before either is AWAITED, otherwise Tinker runs
        # them on separate server clock cycles and the step costs an extra round trip.
        backend, training = make_backend()
        await backend.apply_gradient([make_trajectory()], [torch.ones(5)], defer=True)
        await backend.optim_step(lr=1e-5)

        assert training.log == [
            "submit:backward",
            "submit:optim",
            "await:backward",
            "await:optim",
        ]

    async def test_undeferred_push_awaits_immediately(self) -> None:
        backend, training = make_backend()
        await backend.apply_gradient([make_trajectory()], [torch.ones(5)])
        assert training.log == ["submit:backward", "await:backward"]

    async def test_deferred_metrics_omit_the_server_loss(self) -> None:
        backend, _ = make_backend()
        metrics = await backend.apply_gradient([make_trajectory()], [torch.ones(5)], defer=True)
        assert "loss:sum" not in metrics
        assert metrics["num_trajectories"] == 1.0

    async def test_optim_step_drains_every_deferred_push(self) -> None:
        backend, training = make_backend()
        for _ in range(3):
            await backend.apply_gradient([make_trajectory()], [torch.ones(5)], defer=True)
        metrics = await backend.optim_step(lr=1e-5)
        assert metrics["num_pending_backward"] == 3.0
        assert training.log.count("await:backward") == 3
        # Drained, not re-drained on the next step.
        assert (await backend.optim_step(lr=1e-5))["num_pending_backward"] == 0.0

    async def test_adam_params_are_passed_through(self) -> None:
        backend, training = make_backend()
        await backend.optim_step(lr=3e-4, beta1=0.8, weight_decay=0.01, grad_clip_norm=1.0)
        params = training.optim_params[0]
        assert params.learning_rate == pytest.approx(3e-4)
        assert params.beta1 == pytest.approx(0.8)
        assert params.weight_decay == pytest.approx(0.01)
        assert params.grad_clip_norm == pytest.approx(1.0)

    async def test_tinker_adam_defaults_are_preserved(self) -> None:
        # Tinker's eps is 1e-12 and weight_decay 0.0, unlike torch's AdamW.
        backend, training = make_backend()
        await backend.optim_step(lr=1e-5)
        params = training.optim_params[0]
        assert params.eps == pytest.approx(1e-12)
        assert params.weight_decay == pytest.approx(0.0)

    async def test_non_positive_lr_rejected(self) -> None:
        backend, _ = make_backend()
        with pytest.raises(ValueError, match="positive learning rate"):
            await backend.optim_step(lr=0.0)


# -------------------------------------------------------------------------- sampling


class TestSampling:
    async def test_returns_one_response_per_prompt_in_order(self) -> None:
        backend, training = make_backend()
        responses = await backend.sample([[1, 2, 3], [4, 5]], 2, SamplingParams(max_tokens=8))
        assert len(responses) == 2
        assert [p for p, _ in training.sampling_client.calls] == [[1, 2, 3], [4, 5]]

    async def test_logprobs_come_back_without_a_request_flag(self) -> None:
        backend, _ = make_backend()
        (response,) = await backend.sample([[1, 2, 3]], 2, SamplingParams(max_tokens=8))
        for seq in response.sequences:
            assert seq.logprobs
            assert len(seq.logprobs) == len(seq.tokens)

    def test_missing_logprobs_are_an_error_not_a_zero_fill(self) -> None:
        # Substituting zeros would tell the objective the behaviour policy assigned
        # probability 1 to every token it emitted. The adapter is the boundary where a
        # malformed API response has to be caught, because past it the neutral type
        # cannot represent the problem.
        malformed = SampledSequence(
            stop_reason="stop", tokens_np=np.arange(3, dtype=np.int32), logprobs_np=None
        )
        with pytest.raises(ValueError, match="logprobs"):
            from_tinker_sequence(malformed)

    def test_an_empty_sequence_without_logprobs_is_tolerated(self) -> None:
        # Nothing was sampled, so there is nothing to misrepresent; rollout drops it.
        adapted = from_tinker_sequence(
            SampledSequence(stop_reason="stop", tokens_np=None, logprobs_np=None)
        )
        assert adapted.tokens == []
        assert adapted.logprobs == []

    def test_adapter_carries_stop_reason_and_cache_hits(self) -> None:
        response = SampleResponse(
            sequences=[
                SampledSequence(
                    stop_reason="length",
                    tokens_np=np.arange(2, dtype=np.int32),
                    logprobs_np=np.full(2, -0.5, dtype=np.float32),
                )
            ],
            prompt_cache_hit_tokens=11,
        )
        adapted = from_tinker_response(response)
        assert adapted.prompt_cache_hit_tokens == 11
        assert adapted.sequences[0].stop_reason == "length"
        assert adapted.sequences[0].tokens == [0, 1]
        assert adapted.sequences[0].logprobs == pytest.approx([-0.5, -0.5])

    def test_empty_stop_list_reaches_the_sdk_as_none(self) -> None:
        # The SDK distinguishes "no stop conditions" (None) from an empty list; the
        # neutral type spells the former as an empty sequence.
        assert to_tinker_sampling_params(SamplingParams(max_tokens=8)).stop is None

    async def test_empty_prompts_rejected(self) -> None:
        backend, _ = make_backend()
        with pytest.raises(ValueError, match="at least one prompt"):
            await backend.sample([], 1, SamplingParams(max_tokens=8))

    async def test_non_positive_num_samples_rejected(self) -> None:
        backend, _ = make_backend()
        with pytest.raises(ValueError, match="num_samples must be positive"):
            await backend.sample([[1, 2]], 0, SamplingParams(max_tokens=8))

    async def test_sync_sampler_swaps_in_fresh_weights(self) -> None:
        backend, training = make_backend()
        before = training.sampling_client
        await backend.sync_sampler()
        assert training.sampler_saves == 1
        assert training.sampling_client is not before
        await backend.sample([[1, 2]], 1, SamplingParams(max_tokens=8))
        assert training.sampling_client.calls  # the NEW client received the request


# ----------------------------------------------------------------------- token usage


class TestTokenUsage:
    async def test_starts_empty(self) -> None:
        backend, _ = make_backend()
        assert backend.token_usage() == TokenUsage()

    async def test_forward_counts_model_input_length(self) -> None:
        backend, _ = make_backend()
        traj = make_trajectory()
        await backend.compute_logprobs([traj])
        usage = backend.token_usage()
        assert usage.train_tokens == build_model_input(traj).length
        assert usage.num_forward_passes == 1
        assert usage.num_backward_passes == 0

    async def test_two_passes_cost_twice_the_tokens(self) -> None:
        # This is exactly the passes=2 the cost estimator prices off-policy runs at.
        backend, _ = make_backend()
        traj = make_trajectory()
        (logprobs,) = await backend.compute_logprobs([traj])
        backward(logprobs.sum())
        assert logprobs.grad is not None
        await backend.apply_gradient([traj], [logprobs.grad])
        usage = backend.token_usage()
        assert usage.train_tokens == 2 * build_model_input(traj).length
        assert usage.num_forward_passes == usage.num_backward_passes == 1

    async def test_sampling_counts_prompts_per_sample_plus_generation(self) -> None:
        backend, _ = make_backend()
        await backend.sample([[1, 2, 3]], 2, SamplingParams(max_tokens=8))
        usage = backend.token_usage()
        assert usage.sample_tokens == 2 * 3 + 2 * 4
        assert usage.prompt_cache_hit_tokens == 7
        assert usage.billable_sample_tokens == 14 - 7

    async def test_optim_steps_are_counted(self) -> None:
        backend, _ = make_backend()
        await backend.optim_step(lr=1e-5)
        await backend.optim_step(lr=1e-5)
        assert backend.token_usage().num_optim_steps == 2

    async def test_snapshot_does_not_mutate_afterwards(self) -> None:
        backend, _ = make_backend()
        snapshot = backend.token_usage()
        await backend.compute_logprobs([make_trajectory()])
        assert snapshot.train_tokens == 0
        assert backend.token_usage().train_tokens > 0


class TestCreateValidation:
    async def test_rejects_a_lora_with_no_modules(self) -> None:
        cfg = ModelConfig(
            name="Qwen/Qwen3-8B",
            renderer="qwen3",
            train_mlp=False,
            train_attn=False,
            train_unembed=False,
        )
        with pytest.raises(ValueError, match="no LoRA parameters"):
            await TinkerBackend.create(cfg, api_key="tk-not-used")


# ------------------------------------------------------------------------ live API


@pytest.mark.tinker
async def test_live_round_trip_matches_the_alignment_contract() -> None:
    """Costs real money; deselected by default via ``addopts`` in pyproject.toml.

    Run with ``uv run pytest -m tinker``. Proves against the real service what the fakes
    above assume: that ``forward`` returns exactly ``model_input.length`` logprobs, so the
    completion-only slice has the length we claim.
    """
    from flowcode.config import get_api_key, get_project_id

    backend = await TinkerBackend.create(
        MODEL_CFG, api_key=get_api_key(), project_id=get_project_id()
    )
    traj = make_trajectory()
    (logprobs,) = await backend.compute_logprobs([traj])
    assert logprobs.shape == (traj.num_completion_tokens,)
    assert torch.isfinite(logprobs).all()
    assert (logprobs <= 0).all()

    backward(logprobs.sum())
    assert logprobs.grad is not None
    await backend.apply_gradient([traj], [logprobs.grad], defer=True)
    await backend.optim_step(lr=1e-6)
    assert backend.token_usage().num_optim_steps == 1


def test_fakes_satisfy_the_real_protocols() -> None:
    """Static-only guard; ty is what actually enforces the structural match."""
    training: TrainingClientLike = FakeTrainingClient()
    sampling: SamplingClientLike = FakeSamplingClient()
    backend = TinkerBackend(training, sampling, MODEL_CFG)
    assert isinstance(backend.model_cfg, ModelConfig)
    unused: Any = None
    assert unused is None
