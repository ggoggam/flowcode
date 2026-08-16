"""The objective interface and the arithmetic every GFlowNet loss shares.

An *objective* turns (a) per-token logprobs under the **current** policy and (b) a batch of
scored :class:`~flowcode.types.Trajectory` into a scalar loss plus a metrics dict. Nothing in
this package imports ``tinker``: the losses are plain torch autograd over tensors somebody
else produced, which is what makes them unit-testable offline at zero API cost.

Reduction, and why it matters
-----------------------------
Tinker's ``forward_backward`` **sums** the per-datum losses it is handed; it does not divide
by the batch size. Normalisation is therefore *our* responsibility, and every objective here
reduces with a **mean over trajectories** so that the gradient magnitude — and hence the
usable learning rate — does not drift when ``group_size`` or ``groups_per_step`` changes.
:func:`reduce_per_trajectory` is the single place that happens.

Off-policy correction
---------------------
Every objective accepts ``off_policy_correction`` and it defaults to **False** on purpose.
GFlowNet objectives are off-policy-consistent by construction: the balance conditions they
enforce are properties of the *policy and flow functions*, not expectations under the
sampling distribution, so a trajectory drawn from a stale policy (or from a replay buffer, or
from a deliberately tempered explorer) is still a valid constraint to regress on. That is the
central practical advantage over PPO here, which needs ratio clipping to survive the same
staleness. The importance weight is provided for the rare case where you want the *residual
weighting* to match the on-policy distribution, and it is a variance trade, not a correctness
fix — see :func:`importance_weights`.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from typing import Literal, Protocol, runtime_checkable

import torch

from flowcode.objectives.flows import FlowStates
from flowcode.types import Segment, Trajectory, token_level_segments

__all__ = [
    "DEFAULT_IMPORTANCE_LOG_CLIP",
    "DEFAULT_LOG_REWARD_CEILING",
    "DEFAULT_LOG_REWARD_FLOOR",
    "BaseObjective",
    "Granularity",
    "Objective",
    "boundary_cumulative_logprobs",
    "build_flow_states",
    "importance_weights",
    "reduce_per_trajectory",
    "resolve_segments",
    "segment_logprob_sums",
    "temper_log_reward",
    "tempered_log_rewards",
    "validate_logprobs",
]

Granularity = Literal["token", "turn"]
"""Where GFlowNet state boundaries live inside a completion.

``token`` puts a boundary after every completion token — the standard LLM-GFlowNet setting,
and the finest possible credit assignment. ``turn`` uses the boundaries the environment
already marked in :attr:`Trajectory.segments`, which for the agentic framing is one state per
assistant turn / tool call. Both run through the same code path; it is a config flag, not a
second implementation.
"""

DEFAULT_LOG_REWARD_FLOOR = -20.0
"""Lower clamp on ``log R`` before tempering. A reward of exactly zero is ``-inf`` in log
space and would poison every residual it touches; the floor turns "impossible" into "very
unlikely" (e^-20 ~ 2e-9) which is what the balance conditions can actually represent."""

DEFAULT_LOG_REWARD_CEILING = 20.0
"""Upper clamp, purely defensive against a misconfigured reward returning ``+inf``."""

DEFAULT_IMPORTANCE_LOG_CLIP = 5.0
"""Clamp on the *log* importance ratio. e^±5 is already a 148x reweighting; beyond that the
estimator is all variance and no signal."""


@runtime_checkable
class Objective(Protocol):
    """What the training loop needs from a loss.

    Implementations are plain objects, not ``nn.Module``s: the policy lives on Tinker and only
    the auxiliary flow/partition estimators are local torch parameters, so there is no single
    module tree to hang everything off.
    """

    name: str

    def loss(
        self, logprobs: list[torch.Tensor], batch: Sequence[Trajectory]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the loss for one batch.

        Args:
            logprobs: ``logprobs[i]`` is the current policy's per-token logprobs for
                ``batch[i]``, shape ``(len(batch[i].completion_tokens),)``, requiring grad.
                These come from the Tinker bridge, not from the sampler — the sampler's own
                logprobs live in :attr:`Trajectory.sampling_logprobs` and are stale whenever
                replay or a delayed sampler sync is in play.
            batch: The scored trajectories, aligned index-for-index with ``logprobs``.

        Returns:
            ``(loss, metrics)``: a scalar tensor to call ``.backward()`` on, and a flat dict
            of python floats for logging.
        """
        ...

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        """The client-side parameters this objective owns (log Z / log F), possibly none.

        These get their own optimiser at ``train.flow_lr``, typically 100-1000x the policy LR.
        """
        ...

    def register_tasks(self, task_ids: Sequence[str]) -> None:
        """Declare the task set so per-task tables can be sized once, before the optimiser.

        Hydra instantiates objectives before the environment has necessarily loaded its task
        set, so this cannot be a constructor argument. Calling it again with new ids is
        allowed and grows the tables; doing so *after* an optimiser has been built over
        :meth:`parameters` invalidates that optimiser's state (see
        :class:`~flowcode.objectives.flows.ConditionalLogZ`).
        """
        ...


def validate_logprobs(logprobs: Sequence[torch.Tensor], batch: Sequence[Trajectory]) -> None:
    """Fail loudly on a misaligned batch rather than silently training on garbage.

    Args:
        logprobs: Per-trajectory per-token logprob tensors.
        batch: The trajectories they are supposed to describe.

    Raises:
        ValueError: If the batch is empty, the two sequences disagree in length, or any
            tensor's shape does not match its trajectory's completion length.
    """
    if not batch:
        raise ValueError("empty batch: an objective needs at least one trajectory")
    if len(logprobs) != len(batch):
        raise ValueError(
            f"got {len(logprobs)} logprob tensors for {len(batch)} trajectories; they must "
            "align index-for-index"
        )
    for i, (lp, traj) in enumerate(zip(logprobs, batch, strict=True)):
        if lp.ndim != 1:
            raise ValueError(
                f"logprobs[{i}] has shape {tuple(lp.shape)}; expected 1-D (num_completion_tokens,)"
            )
        if lp.shape[0] != traj.num_completion_tokens:
            raise ValueError(
                f"logprobs[{i}] has length {lp.shape[0]} but trajectory {traj.task_id!r} has "
                f"{traj.num_completion_tokens} completion tokens"
            )


def temper_log_reward(
    log_reward: float,
    beta: float,
    floor: float = DEFAULT_LOG_REWARD_FLOOR,
    ceiling: float = DEFAULT_LOG_REWARD_CEILING,
) -> float:
    """Clamp then temper a single ``log R``.

    The GFlowNet target is ``p(x) ∝ R(x)^β``, so tempering is a multiplication in log space.
    ``β > 1`` sharpens toward the argmax (and gives up the mode coverage that is the whole
    point); ``β < 1`` flattens.

    Args:
        log_reward: Raw ``log R(x)``. ``nan`` is treated as the floor.
        beta: Reward temperature ``β``.
        floor: Lower clamp applied before tempering.
        ceiling: Upper clamp applied before tempering.

    Returns:
        ``β * clamp(log_reward, floor, ceiling)``.
    """
    if math.isnan(log_reward):
        return beta * floor
    return beta * min(max(log_reward, floor), ceiling)


def tempered_log_rewards(
    batch: Sequence[Trajectory],
    beta: float,
    floor: float = DEFAULT_LOG_REWARD_FLOOR,
    ceiling: float = DEFAULT_LOG_REWARD_CEILING,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Vectorised :func:`temper_log_reward` over a batch.

    Args:
        batch: Trajectories to read ``log_reward`` from.
        beta: Reward temperature.
        floor: Lower clamp on ``log R``.
        ceiling: Upper clamp on ``log R``.
        device: Target device; defaults to CPU.
        dtype: Target dtype.

    Returns:
        Shape ``(len(batch),)``, no grad (rewards are constants).
    """
    values = [temper_log_reward(t.log_reward, beta, floor, ceiling) for t in batch]
    return torch.tensor(values, device=device, dtype=dtype)


def resolve_segments(traj: Trajectory, granularity: Granularity) -> list[Segment]:
    """Pick the state boundaries for one trajectory under the configured granularity.

    ``turn`` trusts the environment's own segmentation. ``token`` overrides it with one
    boundary per completion token, but *keeps* any ``partial_log_reward`` the environment
    attached, re-anchored to the token index it was scored at — losing those would throw away
    the only intermediate supervision the flow estimator ever gets.

    Args:
        traj: The trajectory whose boundaries are wanted.
        granularity: ``"token"`` or ``"turn"``.

    Returns:
        Segments tiling ``[0, num_completion_tokens)``.

    Raises:
        ValueError: If ``granularity`` is not one of the two literals.
    """
    if granularity == "turn":
        return list(traj.segments)
    if granularity != "token":
        raise ValueError(f"granularity must be 'token' or 'turn', got {granularity!r}")

    partials = {
        seg.end: seg.partial_log_reward
        for seg in traj.segments
        if seg.partial_log_reward is not None
    }
    if not partials:
        return token_level_segments(traj.num_completion_tokens)
    return [
        Segment(start=s.start, end=s.end, partial_log_reward=partials.get(s.end))
        for s in token_level_segments(traj.num_completion_tokens)
    ]


def segment_logprob_sums(logprobs: torch.Tensor, segments: Sequence[Segment]) -> torch.Tensor:
    """Sum ``log P_F`` within each segment, differentiably.

    Args:
        logprobs: Shape ``(T,)`` per-token logprobs for one trajectory.
        segments: Segments tiling ``[0, T)``.

    Returns:
        Shape ``(M,)`` where ``M = len(segments)``.
    """
    boundaries = boundary_cumulative_logprobs(logprobs, segments)
    return boundaries[1:] - boundaries[:-1]


def boundary_cumulative_logprobs(
    logprobs: torch.Tensor, segments: Sequence[Segment]
) -> torch.Tensor:
    """Cumulative ``Σ log P_F`` at every state boundary, including the source state.

    This is the cumulative sum that makes the sub-trajectory losses vectorisable: the balance
    residual of *any* pair of states ``(i, j)`` is a difference of two entries of this vector
    (plus the corresponding flows), so no per-pair summation is ever needed.

    Args:
        logprobs: Shape ``(T,)`` per-token logprobs for one trajectory.
        segments: ``M`` segments tiling ``[0, T)``.

    Returns:
        Shape ``(M + 1,)``: entry ``k`` is ``Σ_{t < boundary_k} log P_F(t)``, so entry 0 is
        exactly 0 (the source state ``s_0``) and entry ``M`` is the full trajectory logprob.
    """
    cumulative = torch.cat([logprobs.new_zeros(1), torch.cumsum(logprobs, dim=0)])
    ends = torch.tensor([0, *[s.end for s in segments]], device=logprobs.device, dtype=torch.long)
    return cumulative[ends]


def importance_weights(
    logprobs: Sequence[torch.Tensor],
    batch: Sequence[Trajectory],
    enabled: bool,
    log_clip: float = DEFAULT_IMPORTANCE_LOG_CLIP,
) -> torch.Tensor | None:
    """Per-trajectory importance weights ``P_current / P_sampling``, or ``None`` when off.

    The weights are **detached** — they reweight which residuals matter, they are not part of
    the function being differentiated — clipped in log space, and normalised to mean 1 so the
    effective learning rate does not move with the batch's average staleness.

    Correctness note: the GFlowNet losses do not need this. A balance residual is zero at the
    optimum for *every* trajectory regardless of how it was sampled, so an off-policy batch
    biases nothing. Turning this on trades a little bias in *emphasis* for higher variance;
    it exists for ablations and for the case where you want the residual weighting to match
    the on-policy distribution exactly.

    Args:
        logprobs: Current-policy per-token logprobs.
        batch: Trajectories carrying ``sampling_logprobs`` from the behaviour policy.
        enabled: When ``False``, returns ``None`` (callers treat that as uniform weights).
        log_clip: Symmetric clamp on the log ratio.

    Returns:
        Shape ``(len(batch),)`` detached weights with mean 1, or ``None``.
    """
    if not enabled:
        return None
    device = logprobs[0].device
    current = torch.stack([lp.detach().sum() for lp in logprobs])
    behaviour = torch.tensor(
        [float(sum(t.sampling_logprobs)) for t in batch], device=device, dtype=current.dtype
    )
    log_ratio = torch.clamp(current - behaviour, min=-log_clip, max=log_clip)
    weights = torch.exp(log_ratio)
    return weights / weights.mean().clamp_min(1e-12)


def reduce_per_trajectory(
    per_trajectory: torch.Tensor, weights: torch.Tensor | None = None
) -> torch.Tensor:
    """Reduce per-trajectory losses to the scalar the training loop backprops.

    **Mean**, not sum. Tinker's backend sums whatever per-datum losses it is given, so if we
    also summed here the gradient scale would be quadratic in batch size and every learning
    rate would be a function of ``group_size``.

    Args:
        per_trajectory: Shape ``(B,)`` losses.
        weights: Optional shape ``(B,)`` non-negative weights (already mean-1 normalised).

    Returns:
        A scalar tensor.
    """
    if weights is None:
        return per_trajectory.mean()
    return (per_trajectory * weights).sum() / weights.sum().clamp_min(1e-12)


def build_flow_states(
    logprobs: Sequence[torch.Tensor],
    batch: Sequence[Trajectory],
    segments_per_traj: Sequence[Sequence[Segment]],
    cumulative_per_traj: Sequence[torch.Tensor],
    beta: float,
    floor: float = DEFAULT_LOG_REWARD_FLOOR,
    ceiling: float = DEFAULT_LOG_REWARD_CEILING,
) -> FlowStates:
    """Pack every *intermediate* state in the batch into one :class:`FlowStates`.

    "Intermediate" excludes both endpoints: ``s_0``'s flow is ``log Z(x)`` (a free per-task
    parameter) and ``s_M``'s flow is pinned to the true ``β log R(x)``. Everything between
    them is what the flow estimator has to predict, and they are batched across the whole
    minibatch so the estimator runs as a single forward pass.

    Args:
        logprobs: Current-policy per-token logprobs (used only for device/dtype).
        batch: The trajectories.
        segments_per_traj: Resolved segments, one list per trajectory.
        cumulative_per_traj: ``(M+1,)`` boundary cumulative logprobs per trajectory.
        beta: Reward temperature, applied to partial log rewards too.
        floor: Lower clamp on partial log rewards.
        ceiling: Upper clamp on partial log rewards.

    Returns:
        A :class:`FlowStates` whose ``k``-th row is one intermediate state. ``traj_slices``
        records which rows belong to which trajectory.
    """
    device = logprobs[0].device
    task_ids: list[str] = []
    traj_index: list[int] = []
    end_index: list[int] = []
    position: list[float] = []
    cum_logprob: list[float] = []
    partial: list[float] = []
    has_partial: list[float] = []
    slices: list[tuple[int, int]] = []

    cursor = 0
    for b, (traj, segments) in enumerate(zip(batch, segments_per_traj, strict=True)):
        cumulative = cumulative_per_traj[b].detach()
        n_tokens = float(traj.num_completion_tokens)
        start = cursor
        # states s_1 .. s_{M-1}: segment k-1 ends at boundary k
        for k in range(1, len(segments)):
            seg = segments[k - 1]
            task_ids.append(traj.task_id)
            traj_index.append(b)
            end_index.append(seg.end)
            position.append(seg.end / n_tokens)
            cum_logprob.append(float(cumulative[k].item()))
            if seg.partial_log_reward is None:
                partial.append(0.0)
                has_partial.append(0.0)
            else:
                partial.append(temper_log_reward(seg.partial_log_reward, beta, floor, ceiling))
                has_partial.append(1.0)
            cursor += 1
        slices.append((start, cursor))

    def _f(values: list[float]) -> torch.Tensor:
        return torch.tensor(values, device=device, dtype=torch.float32)

    def _i(values: list[int]) -> torch.Tensor:
        return torch.tensor(values, device=device, dtype=torch.long)

    return FlowStates(
        task_ids=task_ids,
        traj_index=_i(traj_index),
        end_index=_i(end_index),
        position=_f(position),
        cum_logprob=_f(cum_logprob),
        partial_log_reward=_f(partial),
        has_partial=_f(has_partial),
        completions=[t.completion_tokens for t in batch],
        traj_slices=slices,
    )


class BaseObjective(ABC):
    """Shared constructor, knobs and defaults for the four losses.

    Subclasses implement :meth:`loss`; everything structural (name, reward tempering, the
    off-policy switch, an empty parameter set) lives here so the four objective modules stay
    about their maths.
    """

    def __init__(
        self,
        name: str,
        reward_temperature: float = 1.0,
        off_policy_correction: bool = False,
        log_reward_floor: float = DEFAULT_LOG_REWARD_FLOOR,
        log_reward_ceiling: float = DEFAULT_LOG_REWARD_CEILING,
        importance_log_clip: float = DEFAULT_IMPORTANCE_LOG_CLIP,
    ) -> None:
        """Initialise the shared knobs.

        Args:
            name: Objective name, echoed into metrics and checkpoints.
            reward_temperature: ``β`` in the target ``p(x) ∝ R(x)^β``. Must be positive.
            off_policy_correction: Enable importance reweighting. Off by default; see the
                module docstring for why that is the right default.
            log_reward_floor: Lower clamp on ``log R``.
            log_reward_ceiling: Upper clamp on ``log R``.
            importance_log_clip: Symmetric clamp on the log importance ratio.

        Raises:
            ValueError: If ``reward_temperature`` is not positive or the clamps are inverted.
        """
        if reward_temperature <= 0.0:
            raise ValueError(f"reward_temperature must be > 0, got {reward_temperature}")
        if log_reward_floor >= log_reward_ceiling:
            raise ValueError(
                f"log_reward_floor ({log_reward_floor}) must be below log_reward_ceiling "
                f"({log_reward_ceiling})"
            )
        self.name = name
        self.reward_temperature = reward_temperature
        self.off_policy_correction = off_policy_correction
        self.log_reward_floor = log_reward_floor
        self.log_reward_ceiling = log_reward_ceiling
        self.importance_log_clip = importance_log_clip

    @abstractmethod
    def loss(
        self, logprobs: list[torch.Tensor], batch: Sequence[Trajectory]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """See :meth:`Objective.loss`."""
        raise NotImplementedError

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        """No client-side parameters by default (VarGrad genuinely has none)."""
        return iter(())

    def register_tasks(self, task_ids: Sequence[str]) -> None:
        """No-op by default; objectives with per-task tables override this."""
        return None

    def _tempered_log_rewards(
        self, batch: Sequence[Trajectory], device: torch.device
    ) -> torch.Tensor:
        return tempered_log_rewards(
            batch,
            self.reward_temperature,
            self.log_reward_floor,
            self.log_reward_ceiling,
            device=device,
        )

    def _importance_weights(
        self, logprobs: Sequence[torch.Tensor], batch: Sequence[Trajectory]
    ) -> torch.Tensor | None:
        return importance_weights(
            logprobs, batch, self.off_policy_correction, self.importance_log_clip
        )

    def _base_metrics(
        self, loss: torch.Tensor, log_rewards: torch.Tensor, weights: torch.Tensor | None
    ) -> dict[str, float]:
        metrics = {
            "loss": float(loss.detach().item()),
            "log_reward_mean": float(log_rewards.mean().item()),
            "log_reward_max": float(log_rewards.max().item()),
        }
        if weights is not None:
            metrics["importance_weight_max"] = float(weights.max().item())
            metrics["importance_weight_min"] = float(weights.min().item())
        return metrics

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self.name!r}, "
            f"reward_temperature={self.reward_temperature}, "
            f"off_policy_correction={self.off_policy_correction})"
        )
