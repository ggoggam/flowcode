"""Detailed Balance.

Reference:
    Yoshua Bengio, Tristan Deleu, Edward J. Hu, Salem Lahlou, Mo Tiwari, Emmanuel Bengio.
    *GFlowNet foundations.* JMLR 2023. https://arxiv.org/abs/2111.09266 — DB is the original
    flow-matching-style constraint; TB and SubTB were later proposed as remedies for its
    credit-assignment behaviour.

One constraint per *edge* of the trajectory::

    L = Σ_i ( log F(s_i) + log P_F(s_i → s_{i+1}) - log F(s_{i+1}) )²

with ``log F(s_0) = log Z(x)`` and ``log F(s_M) = β log R(x)`` pinned exactly as in SubTB.
Since generation is a tree (each token prefix has one parent), ``P_B ≡ 1`` and the backward
term drops out.

This is the ``λ → 0`` limit of :class:`~flowcode.objectives.subtb.SubTB`: the densest possible
credit assignment, and therefore the most sensitive to a badly fit flow estimator. Read the
:mod:`flowcode.objectives.flows` docstring before reaching for it — our ``log F`` cannot see
the prefix, only its position and cumulative logprob, and DB is the objective that leans on
that approximation hardest. Every adjacent residual is a direct claim about a specific state's
flow, so where TB would absorb the error into a single terminal constraint, DB propagates it
into every token's gradient. Prefer ``objective=vargrad`` unless you are deliberately
studying the trade-off.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from itertools import chain
from typing import Literal

import torch
from torch import nn

from flowcode.objectives.base import (
    DEFAULT_IMPORTANCE_LOG_CLIP,
    DEFAULT_LOG_REWARD_CEILING,
    DEFAULT_LOG_REWARD_FLOOR,
    BaseObjective,
    Granularity,
    boundary_cumulative_logprobs,
    build_flow_states,
    reduce_per_trajectory,
    resolve_segments,
    validate_logprobs,
)
from flowcode.objectives.flows import ConditionalLogZ, LogFlow, LogFlowEstimator, LogZ
from flowcode.types import Segment, Trajectory

__all__ = ["DetailedBalance"]

EdgeReduction = Literal["sum", "mean"]
"""How the per-edge residuals collapse into a per-trajectory loss.

``"sum"`` is the objective as written in the papers. ``"mean"`` divides by the edge count,
which makes the gradient scale length-invariant — worth switching to if you run
``granularity="token"`` over completions whose lengths vary by an order of magnitude, because
under ``"sum"`` a 512-token completion contributes 512x the gradient of a 1-token one purely
because of its length.
"""


class DetailedBalance(BaseObjective):
    """The DB loss over adjacent state boundaries."""

    def __init__(
        self,
        name: str = "db",
        granularity: Granularity = "token",
        reward_temperature: float = 1.0,
        off_policy_correction: bool = False,
        flow: LogFlow | None = None,
        log_reward_floor: float = DEFAULT_LOG_REWARD_FLOOR,
        log_reward_ceiling: float = DEFAULT_LOG_REWARD_CEILING,
        importance_log_clip: float = DEFAULT_IMPORTANCE_LOG_CLIP,
        edge_reduction: EdgeReduction = "sum",
        log_z: LogZ | None = None,
    ) -> None:
        """Build the objective.

        Args:
            name: Objective name for logging.
            granularity: ``"token"`` for a boundary per completion token, ``"turn"`` to use
                the environment's own segmentation.
            reward_temperature: ``β`` in ``p(x) ∝ R(x)^β``.
            off_policy_correction: Importance-reweight residuals. Off by default.
            flow: The ``log F`` estimator for intermediate states, instantiated by Hydra from
                the ``flow:`` block. Defaults to a fresh
                :class:`~flowcode.objectives.flows.LogFlowEstimator`.
            log_reward_floor: Lower clamp on ``log R`` before tempering.
            log_reward_ceiling: Upper clamp on ``log R`` before tempering.
            importance_log_clip: Symmetric clamp on the log importance ratio.
            edge_reduction: See :data:`EdgeReduction`.
            log_z: Estimator for ``log F(s_0) = log Z(x)``; an implementation detail of DB,
                injectable for tests.

        Raises:
            ValueError: If ``granularity`` or ``edge_reduction`` is not a recognised literal.
        """
        super().__init__(
            name=name,
            reward_temperature=reward_temperature,
            off_policy_correction=off_policy_correction,
            log_reward_floor=log_reward_floor,
            log_reward_ceiling=log_reward_ceiling,
            importance_log_clip=importance_log_clip,
        )
        if granularity not in ("token", "turn"):
            raise ValueError(f"granularity must be 'token' or 'turn', got {granularity!r}")
        if edge_reduction not in ("sum", "mean"):
            raise ValueError(f"edge_reduction must be 'sum' or 'mean', got {edge_reduction!r}")
        self.granularity: Granularity = granularity
        self.edge_reduction: EdgeReduction = edge_reduction
        self.flow: LogFlow = LogFlowEstimator() if flow is None else flow
        self.log_z: LogZ = ConditionalLogZ() if log_z is None else log_z

    def loss(
        self, logprobs: list[torch.Tensor], batch: Sequence[Trajectory]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the detailed-balance loss.

        Args:
            logprobs: Current-policy per-token logprobs, one 1-D tensor per trajectory.
            batch: The scored trajectories.

        Returns:
            ``(loss, metrics)``.

        Raises:
            ValueError: If ``logprobs`` and ``batch`` are misaligned.
        """
        validate_logprobs(logprobs, batch)
        device = logprobs[0].device

        segments_per_traj = [resolve_segments(t, self.granularity) for t in batch]
        cumulative_per_traj = [
            boundary_cumulative_logprobs(lp, segs)
            for lp, segs in zip(logprobs, segments_per_traj, strict=True)
        ]
        log_rewards = self._tempered_log_rewards(batch, device)
        log_z = self.log_z.log_z([t.task_id for t in batch]).to(device)

        states = build_flow_states(
            logprobs,
            batch,
            segments_per_traj,
            cumulative_per_traj,
            self.reward_temperature,
            self.log_reward_floor,
            self.log_reward_ceiling,
        )
        interior_flows = self.flow.log_flow(states).to(device)

        per_trajectory = []
        residual_abs: list[torch.Tensor] = []
        edge_counts: list[int] = []
        for b in range(len(batch)):
            start, stop = states.traj_slices[b]
            log_flows = torch.cat(
                [log_z[b : b + 1], interior_flows[start:stop], log_rewards[b : b + 1]]
            )
            # Same identity SubTB uses, restricted to adjacent states:
            #   F_i + (C_{i+1} - C_i) - F_{i+1}  =  potential_i - potential_{i+1}
            potential = log_flows - cumulative_per_traj[b]
            residuals = potential[:-1] - potential[1:]
            squared = residuals.square()
            per_trajectory.append(squared.sum() if self.edge_reduction == "sum" else squared.mean())
            residual_abs.append(residuals.detach().abs().mean())
            edge_counts.append(int(residuals.shape[0]))

        stacked = torch.stack(per_trajectory)
        weights = self._importance_weights(logprobs, batch)
        loss = reduce_per_trajectory(stacked, weights)

        metrics = self._base_metrics(loss, log_rewards, weights)
        metrics.update(
            {
                "log_z_mean": float(log_z.detach().mean().item()),
                "residual_abs_mean": float(torch.stack(residual_abs).mean().item()),
                "edges_per_trajectory": float(sum(edge_counts)) / float(len(batch)),
                "interior_flow_mean": (
                    float(interior_flows.detach().mean().item()) if len(states) else 0.0
                ),
            }
        )
        return loss, metrics

    def parameters(self) -> Iterator[nn.Parameter]:
        """Flow-estimator parameters plus the source-state ``log Z`` table."""
        return chain(self.flow.parameters(), self.log_z.parameters())

    def register_tasks(self, task_ids: Sequence[str]) -> None:
        """Size both per-task tables for the full task set."""
        self.flow.register_tasks(task_ids)
        self.log_z.register_tasks(task_ids)

    def segments_for(self, traj: Trajectory) -> list[Segment]:
        """The boundaries this objective would use for ``traj`` — exposed for tests/debugging."""
        return resolve_segments(traj, self.granularity)
