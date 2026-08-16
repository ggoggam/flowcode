"""Trajectory Balance.

Reference:
    Nikolay Malkin, Moksh Jain, Emmanuel Bengio, Chen Sun, Yoshua Bengio.
    *Trajectory balance: Improved credit assignment in GFlowNets.* NeurIPS 2022.
    https://arxiv.org/abs/2201.13259

The constraint is that the total forward flow of a trajectory equals its terminal reward::

    Z(x) · Π_t P_F(s_{t+1} | s_t)  =  R(x)^β

which in log space, squared, is the loss::

    L = ( log Z(x) + Σ_t log P_F(t) - β log R(x) )²

Because generation is left-to-right over token prefixes, the GFlowNet DAG is a tree: each
state has exactly one parent, so ``P_B ≡ 1`` and the backward term that appears in the general
TB loss vanishes. That is why nothing here models a backward policy.

TB imposes exactly one constraint per trajectory, which is both its strength (no flow
estimator to misfit, so it is unbiased regardless of how good our client-side approximations
are) and its weakness (all credit is assigned at the terminal state; long completions get a
very sparse learning signal). Its one fiddly piece is ``log Z``: it has to travel dozens of
nats before the residual becomes informative, which is why it gets its own optimiser at
``train.flow_lr``. :class:`~flowcode.objectives.vargrad.VarGrad` removes it entirely and is
the recommended default.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import torch
from torch import nn

from flowcode.objectives.base import (
    DEFAULT_IMPORTANCE_LOG_CLIP,
    DEFAULT_LOG_REWARD_CEILING,
    DEFAULT_LOG_REWARD_FLOOR,
    BaseObjective,
    reduce_per_trajectory,
    validate_logprobs,
)
from flowcode.objectives.flows import ConditionalLogZ, LogZ
from flowcode.types import Trajectory

__all__ = ["TrajectoryBalance"]


class TrajectoryBalance(BaseObjective):
    """The TB loss with a learned, prompt-conditional ``log Z``."""

    def __init__(
        self,
        name: str = "tb",
        reward_temperature: float = 1.0,
        off_policy_correction: bool = False,
        log_z: LogZ | None = None,
        log_reward_floor: float = DEFAULT_LOG_REWARD_FLOOR,
        log_reward_ceiling: float = DEFAULT_LOG_REWARD_CEILING,
        importance_log_clip: float = DEFAULT_IMPORTANCE_LOG_CLIP,
    ) -> None:
        """Build the objective.

        Args:
            name: Objective name for logging.
            reward_temperature: ``β`` in ``p(x) ∝ R(x)^β``.
            off_policy_correction: Importance-reweight residuals to the on-policy
                distribution. Off by default — TB is off-policy-consistent.
            log_z: The partition-function estimator, instantiated by Hydra from the
                ``log_z:`` block of ``conf/objective/tb.yaml``. Defaults to a fresh
                :class:`~flowcode.objectives.flows.ConditionalLogZ` when omitted.
            log_reward_floor: Lower clamp on ``log R`` before tempering.
            log_reward_ceiling: Upper clamp on ``log R`` before tempering.
            importance_log_clip: Symmetric clamp on the log importance ratio.
        """
        super().__init__(
            name=name,
            reward_temperature=reward_temperature,
            off_policy_correction=off_policy_correction,
            log_reward_floor=log_reward_floor,
            log_reward_ceiling=log_reward_ceiling,
            importance_log_clip=importance_log_clip,
        )
        self.log_z: LogZ = ConditionalLogZ() if log_z is None else log_z

    def loss(
        self, logprobs: list[torch.Tensor], batch: Sequence[Trajectory]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the trajectory-balance loss.

        Args:
            logprobs: Current-policy per-token logprobs, one 1-D tensor per trajectory.
            batch: The scored trajectories.

        Returns:
            ``(loss, metrics)``. Metrics include the mean residual (signed — a persistent
            positive value means ``log Z`` is still climbing) and its absolute value, which is
            the number to watch for convergence.

        Raises:
            ValueError: If ``logprobs`` and ``batch`` are misaligned.
        """
        validate_logprobs(logprobs, batch)
        device = logprobs[0].device

        forward_logprob = torch.stack([lp.sum() for lp in logprobs])
        log_rewards = self._tempered_log_rewards(batch, device)
        log_z = self.log_z.log_z([t.task_id for t in batch]).to(device)

        residual = log_z + forward_logprob - log_rewards
        weights = self._importance_weights(logprobs, batch)
        loss = reduce_per_trajectory(residual.square(), weights)

        metrics = self._base_metrics(loss, log_rewards, weights)
        metrics.update(
            {
                "log_z_mean": float(log_z.detach().mean().item()),
                "residual_mean": float(residual.detach().mean().item()),
                "residual_abs_mean": float(residual.detach().abs().mean().item()),
                "forward_logprob_mean": float(forward_logprob.detach().mean().item()),
            }
        )
        return loss, metrics

    def parameters(self) -> Iterator[nn.Parameter]:
        """The ``log Z`` parameters — the only client-side state TB owns."""
        return self.log_z.parameters()

    def register_tasks(self, task_ids: Sequence[str]) -> None:
        """Size the ``log Z`` table for the full task set."""
        self.log_z.register_tasks(task_ids)
