"""The bridge between client-side GFlowNet objectives and Tinker's fixed loss menu.

Tinker exposes five server-side losses (``cross_entropy``, ``importance_sampling``,
``ppo``, ``cispo``, ``dro``). None of them is trajectory balance, and there is no way to
ship a new one. What makes GFlowNets trainable on Tinker anyway is a single algebraic
fact about the backend's cross-entropy::

    L = sum(-target_logprobs * weights)

``weights`` is an arbitrary per-token float array — nothing constrains it to be a mask, or
positive, or to sum to one. So for *any* differentiable client-side scalar
``C(logprobs)``::

    dL/dlogprobs = -weights

and therefore setting ``weights = -dC/dlogprobs`` makes the server's backward pass deposit
exactly ``dC/dtheta`` into the gradient accumulator. The objective can be trajectory
balance, sub-trajectory balance, detailed balance or anything else; the server never
learns what it computed.

This is not a trick we invented — it is precisely what the SDK's own
``forward_backward_custom_async`` does, and its comment at
``tinker/lib/public_interfaces/training_client.py:475`` states the rule verbatim. We
reimplement it rather than call it because ``forward_backward_custom`` is decorated
``@sync_only``, computes ``loss.backward()`` inside the SDK on one datum-list at a time,
and gives the objective no place to keep client-side ``log Z`` / ``log F`` parameters in
the same autograd graph. Splitting it into :meth:`TinkerBackend.compute_logprobs` and
:meth:`TinkerBackend.apply_gradient` puts the whole backward pass in our process, where
the flow parameters live.

The alignment contract
----------------------
Which output position carries which completion token's logprob is not a Tinker question —
every backend faces it — so it lives in :mod:`flowcode.alignment` and is merely *used*
here. :func:`build_datum` asserts the three lengths agree on every single call rather than
trusting them.

Normalization
-------------
Tinker's losses ``.sum()``, and gradients *accumulate* across every ``forward_backward``
until an ``optim_step`` clears them. Nothing in this module divides by anything. If you
push gradients for eight groups and then step, you have stepped on the sum over all eight,
not the mean. Pass ``grad_scale`` to :meth:`apply_gradient` if you want a denominator, and
read it back out of the returned metrics — the number is reported, never implied.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, TypeVar

import numpy as np
import tinker
import torch
from tinker import types as tinker_types

from flowcode.alignment import (
    build_input_tokens,
    build_target_tokens,
    pad_completion_values,
    slice_completion_values,
)
from flowcode.config import ModelConfig
from flowcode.types import (
    SampledSequence,
    SampleResponse,
    SamplingParams,
    TokenUsage,
    Trajectory,
    TrajectoryBatch,
)

if TYPE_CHECKING:
    # Tinker's own wire types, aliased because this module now speaks two vocabularies:
    # the SDK's on the client side and flowcode's neutral ones on the rollout side.
    from tinker.types import (
        Datum,
        ForwardBackwardOutput,
        ModelInput,
        OptimStepResponse,
    )
    from tinker.types import SampledSequence as TinkerSampledSequence
    from tinker.types import SampleResponse as TinkerSampleResponse
    from tinker.types import SamplingParams as TinkerSamplingParams

__all__ = [
    "LOSS_FN",
    "SamplingClientLike",
    "TinkerBackend",
    "TokenUsage",
    "TrainingClientLike",
    "build_datum",
    "build_model_input",
    "from_tinker_response",
    "from_tinker_sequence",
    "to_tinker_sampling_params",
]

logger = logging.getLogger(__name__)

LOSS_FN: tinker_types.LossFnType = "cross_entropy"
"""The only backend loss this module uses. See the module docstring for why that is enough."""

# --------------------------------------------------------------------------------------
# Structural types for the two Tinker clients.
#
# Declared as Protocols rather than as `tinker.TrainingClient` / `tinker.SamplingClient`
# so the unit tests can inject fakes that still type-check. The real SDK
# classes satisfy these structurally; `TinkerBackend.create` assigns real ones and ty
# verifies the match there.
# --------------------------------------------------------------------------------------

_T_co = TypeVar("_T_co", covariant=True)


class _FutureLike(Protocol[_T_co]):
    """The half of ``tinker.APIFuture`` we use: await the submitted request's result."""

    async def result_async(self, timeout: float | None = None) -> _T_co: ...


class TrainingClientLike(Protocol):
    """The subset of ``tinker.TrainingClient`` that :class:`TinkerBackend` depends on.

    Note the double-await shape on the first three methods: they are ``async def`` that
    *return a future*. The first await submits the request and hands back an ack; the
    second waits for the server to finish it. Collapsing them into one await is the
    classic way to accidentally serialise a pipeline.
    """

    async def forward_async(
        self,
        data: list[Datum],
        loss_fn: tinker_types.LossFnType,
        loss_fn_config: dict[str, float] | None = None,
    ) -> _FutureLike[ForwardBackwardOutput]: ...

    async def forward_backward_async(
        self,
        data: list[Datum],
        loss_fn: tinker_types.LossFnType,
        loss_fn_config: dict[str, float] | None = None,
    ) -> _FutureLike[ForwardBackwardOutput]: ...

    async def optim_step_async(
        self, adam_params: tinker_types.AdamParams
    ) -> _FutureLike[OptimStepResponse]: ...

    async def save_weights_and_get_sampling_client_async(self) -> SamplingClientLike: ...


class SamplingClientLike(Protocol):
    """The subset of ``tinker.SamplingClient`` that :class:`TinkerBackend` depends on.

    Unlike the training client's methods, ``sample_async`` is a *single* await: it returns
    the :class:`~tinker.types.SampleResponse` itself, not a future.
    """

    async def sample_async(
        self,
        prompt: ModelInput,
        num_samples: int,
        sampling_params: TinkerSamplingParams,
    ) -> TinkerSampleResponse: ...


# --------------------------------------------------------------------------------------
# The alignment contract, as free functions so the tests can pin it without a client.
# --------------------------------------------------------------------------------------


def build_model_input(trajectory: Trajectory) -> ModelInput:
    """Wrap :func:`~flowcode.alignment.build_input_tokens` in a Tinker :class:`ModelInput`.

    Args:
        trajectory: The trajectory to encode.

    Returns:
        A ``ModelInput`` of length ``ob_len + len(completion_tokens)``.
    """
    return tinker_types.ModelInput.from_ints(build_input_tokens(trajectory))


def build_datum(trajectory: Trajectory, weights: Sequence[float] | None = None) -> Datum:
    """Assemble one :class:`~tinker.types.Datum` obeying the alignment contract.

    Args:
        trajectory: The trajectory to encode.
        weights: Per-completion-token weights, or ``None`` for the all-zero array used by
            the oracle ``forward()`` pass. All-zero weights make the loss identically 0
            while still returning per-position logprobs, which mirrors what the SDK's
            ``_get_custom_loss_forward_data`` does.

    Returns:
        A ``Datum`` whose ``model_input``, ``target_tokens`` and ``weights`` all have the
        same length.

    Raises:
        ValueError: If ``weights`` has the wrong length.
        AssertionError: If the three lengths disagree — the contract this whole module
            rests on, checked every single time rather than trusted.
    """
    model_input = build_model_input(trajectory)
    target_tokens = build_target_tokens(trajectory)
    if weights is None:
        padded_weights = [0.0] * len(target_tokens)
    else:
        padded_weights = pad_completion_values(trajectory, weights)

    assert model_input.length == len(target_tokens) == len(padded_weights), (
        f"Trajectory {trajectory.task_id!r}: alignment contract violated — "
        f"model_input.length={model_input.length}, "
        f"len(target_tokens)={len(target_tokens)}, len(weights)={len(padded_weights)}"
    )

    return tinker_types.Datum(
        model_input=model_input,
        loss_fn_inputs={
            "target_tokens": tinker_types.TensorData(
                data=target_tokens, dtype="int64", shape=[len(target_tokens)]
            ),
            "weights": tinker_types.TensorData(
                data=padded_weights, dtype="float32", shape=[len(padded_weights)]
            ),
        },
    )


def _extract_logprobs(output: ForwardBackwardOutput, index: int) -> torch.Tensor:
    """Pull datum ``index``'s per-position logprobs out of a forward result."""
    outputs = output.loss_fn_outputs
    if index >= len(outputs):
        raise ValueError(
            f"Tinker returned {len(outputs)} loss_fn_outputs but datum {index} was "
            "requested; the server dropped or reordered data."
        )
    entry = outputs[index]
    if "logprobs" not in entry:
        raise ValueError(
            f"Tinker's forward() output for datum {index} has no 'logprobs' key (got "
            f"{sorted(entry)}). cross_entropy is expected to return per-position logprobs."
        )
    tensor_data = entry["logprobs"]
    values = torch.as_tensor(tensor_data.to_numpy(), dtype=torch.float32)
    return values.reshape(-1)


def to_tinker_sampling_params(params: SamplingParams) -> tinker_types.SamplingParams:
    """Adapt flowcode's neutral sampling params to the SDK's.

    Args:
        params: The neutral params a rollout built.

    Returns:
        The SDK equivalent. The fields correspond one-for-one, with one wrinkle: the SDK
        spells "no stop conditions" as ``None`` and distinguishes it from an empty list,
        so an empty neutral ``stop`` is passed through as ``None`` rather than ``[]``.
    """
    return tinker_types.SamplingParams(
        max_tokens=params.max_tokens,
        temperature=params.temperature,
        top_p=params.top_p,
        stop=list(params.stop) or None,
        seed=params.seed,
    )


def from_tinker_sequence(sequence: TinkerSampledSequence) -> SampledSequence:
    """Adapt one SDK sampled sequence to the neutral type.

    Raises:
        ValueError: If the sequence carries tokens but no logprobs. They arrive
            unconditionally on the real API, so their absence means the response is not
            what this code thinks it is — and silently substituting zeros would hand the
            objective a behaviour policy that assigns probability 1 to everything.
    """
    tokens_np = sequence.tokens_np
    logprobs_np = sequence.logprobs_np
    tokens = [int(t) for t in (np.asarray(tokens_np).reshape(-1) if tokens_np is not None else [])]
    if logprobs_np is None:
        if not tokens:
            return SampledSequence(tokens=[], logprobs=[], stop_reason="empty")
        raise ValueError(
            "the sampler returned tokens without logprobs; SampledSequence.logprobs_np is "
            "populated unconditionally by the Tinker API, so this response is malformed"
        )
    logprobs = [float(v) for v in np.asarray(logprobs_np).reshape(-1)]
    if len(logprobs) != len(tokens):
        # Defensive: keep the pair aligned rather than letting SampledSequence reject it.
        keep = min(len(tokens), len(logprobs))
        logger.warning(
            "sampler returned %d tokens but %d logprobs; truncating both to %d",
            len(tokens),
            len(logprobs),
            keep,
        )
        tokens, logprobs = tokens[:keep], logprobs[:keep]
    return SampledSequence(
        tokens=tokens,
        logprobs=logprobs,
        stop_reason=str(getattr(sequence, "stop_reason", "unknown")),
    )


def from_tinker_response(response: TinkerSampleResponse) -> SampleResponse:
    """Adapt one SDK sample response, and every sequence in it, to the neutral types."""
    return SampleResponse(
        sequences=[from_tinker_sequence(s) for s in response.sequences],
        prompt_cache_hit_tokens=int(response.prompt_cache_hit_tokens),
    )


class TinkerBackend:
    """Owns the Tinker training and sampling clients and the gradient bridge.

    Construct with :meth:`create` in production. The plain constructor takes the two
    clients directly, which is how the offline tests drive it.

    Attributes:
        model_cfg: The model group this backend was built from.
    """

    def __init__(
        self,
        training_client: TrainingClientLike,
        sampling_client: SamplingClientLike,
        model_cfg: ModelConfig,
    ) -> None:
        """Wrap already-created clients.

        Args:
            training_client: A ``tinker.TrainingClient`` (or anything satisfying
                :class:`TrainingClientLike`).
            sampling_client: A ``tinker.SamplingClient`` (or anything satisfying
                :class:`SamplingClientLike`).
            model_cfg: The model config these clients were built from. Kept so
                :mod:`flowcode.cost` can price :meth:`token_usage` without being told the
                model name a second time.
        """
        self._training_client = training_client
        self._sampling_client = sampling_client
        self.model_cfg = model_cfg
        self._usage = TokenUsage()
        self._pending_backward: list[_FutureLike[ForwardBackwardOutput]] = []

    @classmethod
    async def create(
        cls,
        model_cfg: ModelConfig,
        api_key: str,
        project_id: str | None = None,
    ) -> TinkerBackend:
        """Create the LoRA training client and its first sampling client.

        Args:
            model_cfg: Base model, LoRA rank and which module families get adapters.
            api_key: Tinker API key, from :func:`flowcode.config.get_api_key`.
            project_id: Optional Tinker project id to group runs under.

        Returns:
            A ready backend. The sampler already holds a snapshot of the (freshly
            initialised) LoRA weights, so :meth:`sample` works immediately.

        Raises:
            ValueError: If no module family is selected for LoRA — Tinker asserts on this
                server-side, and a bare ``assert`` from inside the SDK is a much worse
                error message than this one.
        """
        if not (model_cfg.train_mlp or model_cfg.train_attn or model_cfg.train_unembed):
            raise ValueError(
                "At least one of model.train_mlp / model.train_attn / model.train_unembed "
                "must be true; with all three false there are no LoRA parameters to train."
            )

        service_client = tinker.ServiceClient(api_key=api_key, project_id=project_id)
        training_client = await service_client.create_lora_training_client_async(
            base_model=model_cfg.name,
            rank=model_cfg.lora_rank,
            train_mlp=model_cfg.train_mlp,
            train_attn=model_cfg.train_attn,
            train_unembed=model_cfg.train_unembed,
        )
        # One await, not two: unlike forward_backward_async this returns the client.
        sampling_client = await training_client.save_weights_and_get_sampling_client_async()
        logger.info(
            "TinkerBackend ready: base_model=%s lora_rank=%d", model_cfg.name, model_cfg.lora_rank
        )
        return cls(training_client, sampling_client, model_cfg)

    # ---------------------------------------------------------------- sampling

    async def sample(
        self,
        prompts: Sequence[Sequence[int]],
        num_samples: int,
        sampling_params: SamplingParams,
    ) -> list[SampleResponse]:
        """Sample ``num_samples`` completions for each prompt, concurrently.

        Args:
            prompts: One token-id list per prompt. Order is preserved in the result.
            num_samples: Completions per prompt — the GFlowNet group size.
            sampling_params: Neutral :class:`~flowcode.types.SamplingParams`, adapted to
                the SDK's own type here so that callers never import ``tinker``.

        Returns:
            One :class:`~flowcode.types.SampleResponse` per prompt, in the same order.
            Sampled-token logprobs arrive unconditionally from the API; there is no
            request flag to set.

        Raises:
            ValueError: If ``prompts`` is empty or ``num_samples`` is not positive.
        """
        if not prompts:
            raise ValueError("sample() needs at least one prompt")
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")

        model_inputs = [tinker_types.ModelInput.from_ints(list(p)) for p in prompts]
        params = to_tinker_sampling_params(sampling_params)
        # Deliberately not wrapped in asyncio.wait_for: the Tinker docs warn that
        # cancelling a request mid-flight leaves the session's clock cycle in a state the
        # client cannot recover from. Let it take as long as it takes.
        responses = await asyncio.gather(
            *(
                self._sampling_client.sample_async(
                    prompt=model_input,
                    num_samples=num_samples,
                    sampling_params=params,
                )
                for model_input in model_inputs
            )
        )

        adapted = [from_tinker_response(r) for r in responses]

        gross_tokens = 0
        cache_hits = 0
        for model_input, response in zip(model_inputs, adapted, strict=True):
            gross_tokens += num_samples * model_input.length
            gross_tokens += sum(len(seq.tokens) for seq in response.sequences)
            cache_hits += response.prompt_cache_hit_tokens
        self._usage += TokenUsage(sample_tokens=gross_tokens, prompt_cache_hit_tokens=cache_hits)

        return adapted

    # ------------------------------------------------------------ the oracle pass

    async def compute_logprobs(self, trajectories: TrajectoryBatch) -> list[torch.Tensor]:
        """Score each trajectory's completion tokens under the *current* policy weights.

        This is the forward half of the bridge: an all-zero-weight ``forward()`` whose
        loss is identically zero but whose per-position logprobs come back anyway. It
        mirrors the SDK's ``_get_custom_loss_forward_data``.

        The returned tensors are autograd *leaves* with ``requires_grad=True``. The
        objective builds its loss out of them, calls ``.backward()``, and hands
        ``tensor.grad`` back to :meth:`apply_gradient`. There is no autograd connection to
        the model — the server holds those weights — so this leaf-and-grad handoff is the
        entire chain rule across the network boundary.

        Args:
            trajectories: The batch to score. Must be non-empty.

        Returns:
            One tensor per trajectory, shape ``(len(completion_tokens),)``, prompt padding
            already sliced off, ``requires_grad=True``.

        Raises:
            ValueError: If the batch is empty, or the server returns an unexpected shape.
        """
        if not trajectories:
            raise ValueError("compute_logprobs() needs at least one trajectory")

        data = [build_datum(t, weights=None) for t in trajectories]
        # Double await: the first submits, the second waits for the result.
        future = await self._training_client.forward_async(data, LOSS_FN)
        output = await future.result_async()

        self._usage += TokenUsage(
            train_tokens=sum(d.model_input.length for d in data),
            num_forward_passes=1,
        )

        logprobs: list[torch.Tensor] = []
        for i, trajectory in enumerate(trajectories):
            full = _extract_logprobs(output, i)
            completion_only = slice_completion_values(trajectory, full)
            logprobs.append(completion_only.detach().clone().requires_grad_(True))
        return logprobs

    # ------------------------------------------------------------- the gradient push

    async def apply_gradient(
        self,
        trajectories: TrajectoryBatch,
        grads: Sequence[torch.Tensor],
        *,
        grad_scale: float = 1.0,
        defer: bool = False,
    ) -> dict[str, float]:
        """Push ``dC/dlogprobs`` to the server as ``weights = -grad``.

        See the module docstring for why the sign is what it is. In one line: the backend
        computes ``L = sum(-logprobs * weights)``, so ``weights = -dC/dlogprobs`` gives
        ``dL/dlogprobs = dC/dlogprobs`` and the server's backward pass lands exactly the
        gradient the client-side objective asked for.

        Args:
            trajectories: The same batch, in the same order, that produced ``grads``.
            grads: One gradient tensor per trajectory, each shaped
                ``(len(completion_tokens),)`` — completion-only, exactly as
                :meth:`compute_logprobs` returned. Re-padding is done here.
            grad_scale: Multiplier applied to every weight before sending. ``1.0`` (the
                default) means no normalization at all: gradients accumulate as a raw sum
                over everything pushed since the last :meth:`optim_step`. Pass
                ``1 / total_completion_tokens`` for a token mean, ``1 / len(batch)`` for a
                per-trajectory mean. Whatever you pass is echoed back in the metrics.
            defer: If true, submit the request but do not wait for it. :meth:`optim_step`
                then submits the optimiser step *before* draining it, so both land on one
                server clock cycle instead of two. The returned metrics omit the
                server-side loss in this mode, because it has not arrived yet.

        Returns:
            Metrics: ``num_trajectories``, ``num_completion_tokens``, ``train_tokens``,
            ``grad_scale``, ``weight_abs_sum``, plus every float in the server's
            ``ForwardBackwardOutput.metrics`` (unless ``defer``).

        Raises:
            ValueError: If the batch is empty, the two sequences disagree in length, or a
                gradient has the wrong shape or contains non-finite values.
        """
        if not trajectories:
            raise ValueError("apply_gradient() needs at least one trajectory")
        if len(grads) != len(trajectories):
            raise ValueError(
                f"apply_gradient() got {len(grads)} gradients for {len(trajectories)} "
                "trajectories; they must correspond one-to-one and in the same order as "
                "the compute_logprobs() call that produced them"
            )

        data: list[Datum] = []
        weight_abs_sum = 0.0
        total_completion_tokens = 0
        for trajectory, grad in zip(trajectories, grads, strict=True):
            n = trajectory.num_completion_tokens
            if grad.ndim != 1 or grad.shape[0] != n:
                raise ValueError(
                    f"Trajectory {trajectory.task_id!r}: gradient has shape "
                    f"{tuple(grad.shape)} but the completion has {n} tokens. Pass the "
                    "gradient of the tensor compute_logprobs() returned, unpadded."
                )
            if not torch.isfinite(grad).all():
                raise ValueError(
                    f"Trajectory {trajectory.task_id!r}: gradient contains NaN or inf. "
                    "This is nearly always a log R of -inf reaching the objective; check "
                    "env.reward_floor."
                )
            # THE SIGN. L_server = sum(-logprobs * weights) => dL/dlogprobs = -weights,
            # so weights = -dC/dlogprobs makes the server reproduce dC/dlogprobs.
            weights = (-grad.detach().to(torch.float32) * grad_scale).tolist()
            weight_abs_sum += float(sum(abs(w) for w in weights))
            total_completion_tokens += n
            data.append(build_datum(trajectory, weights))

        future = await self._training_client.forward_backward_async(data, LOSS_FN)
        train_tokens = sum(d.model_input.length for d in data)
        self._usage += TokenUsage(train_tokens=train_tokens, num_backward_passes=1)

        metrics: dict[str, float] = {
            "num_trajectories": float(len(data)),
            "num_completion_tokens": float(total_completion_tokens),
            "train_tokens": float(train_tokens),
            "grad_scale": grad_scale,
            "weight_abs_sum": weight_abs_sum,
        }

        if defer:
            self._pending_backward.append(future)
            return metrics

        output = await future.result_async()
        metrics.update({k: float(v) for k, v in output.metrics.items()})
        return metrics

    async def optim_step(
        self,
        lr: float,
        *,
        beta1: float = 0.9,
        beta2: float = 0.95,
        eps: float = 1e-12,
        weight_decay: float = 0.0,
        grad_clip_norm: float = 0.0,
    ) -> dict[str, float]:
        """Apply Adam to the accumulated gradients and clear the accumulator.

        Submission happens *before* any deferred :meth:`apply_gradient` future is drained,
        which is the whole point: Tinker batches whatever has been submitted into one
        server clock cycle, so waiting for the backward pass first would cost an extra
        round trip per step.

        Args:
            lr: Learning rate for this step. Passed per-call rather than stored so the
                caller owns the schedule.
            beta1: Adam first-moment decay.
            beta2: Adam second-moment decay.
            eps: Adam denominator epsilon. Tinker's default is 1e-12, not torch's 1e-8.
            weight_decay: Decoupled weight decay. Tinker defaults to 0.0, unlike AdamW.
            grad_clip_norm: Global grad-norm clip; ``0.0`` disables.

        Returns:
            Metrics: ``lr`` and ``num_pending_backward`` (how many deferred pushes this
            step absorbed).

        Raises:
            ValueError: If ``lr`` is not positive.
        """
        if lr <= 0:
            raise ValueError(f"optim_step() needs a positive learning rate, got {lr}")

        adam_params = tinker_types.AdamParams(
            learning_rate=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
        )
        # Submit first...
        optim_future = await self._training_client.optim_step_async(adam_params)
        # ...then drain, so the fwd/bwd and the step share a clock cycle.
        pending = self._pending_backward
        self._pending_backward = []
        for backward_future in pending:
            await backward_future.result_async()
        await optim_future.result_async()

        self._usage += TokenUsage(num_optim_steps=1)
        return {"lr": lr, "num_pending_backward": float(len(pending))}

    async def sync_sampler(self) -> None:
        """Point the sampler at the current policy weights.

        Until this runs, :meth:`sample` draws from whatever snapshot was last saved, so
        the sampler's logprobs are the behaviour policy's and not the current policy's.
        That is legal for GFlowNets — the objectives are off-policy-consistent — but it is
        exactly what ``train.on_policy_only`` promises is *not* happening, so that mode
        must call this before every rollout.
        """
        self._sampling_client = (
            await self._training_client.save_weights_and_get_sampling_client_async()
        )

    def token_usage(self) -> TokenUsage:
        """Snapshot of everything this backend has actually spent.

        Returns:
            An immutable :class:`~flowcode.types.TokenUsage`. Feed it to
            :func:`flowcode.cost.from_usage` for dollars.
        """
        return self._usage

    def __repr__(self) -> str:
        u = self._usage
        return (
            f"TinkerBackend(model={self.model_cfg.name!r}, train_tokens={u.train_tokens}, "
            f"sample_tokens={u.sample_tokens})"
        )
