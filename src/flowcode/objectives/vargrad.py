"""VarGrad — trajectory balance with the partition function estimated in-batch.

References:
    Lorenz Richter, Ayman Boustati, Nikolas Nüsken, Francisco J. R. Ruiz, Ömer Deniz Akyildiz.
    *VarGrad: A low-variance gradient estimator for variational inference.* NeurIPS 2020.
    https://arxiv.org/abs/2010.10436

    Nikolay Malkin, Salem Lahlou, Tristan Deleu, Xu Ji, Edward Hu, Katie Everett, Dinghuai
    Zhang, Yoshua Bengio. *GFlowNets and variational inference.* ICLR 2023.
    https://arxiv.org/abs/2210.00580 — establishes that the TB objective with the
    log-partition eliminated is exactly the VarGrad estimator.

The idea in one line: TB's loss ``(log Z + Σ log P_F - β log R)²`` is a quadratic in
``log Z``, so for a group of ``G`` trajectories sharing a prompt the minimising ``log Z`` is
available in closed form — it is the group mean of ``A_i := β log R_i - Σ_t log P_F(i, t)``.
Substituting it back leaves::

    L = (1/G) Σ_i ( A_i - mean_g(A) )²

i.e. the **within-group variance** of ``A``. No learned partition function, no second
optimiser, no ``flow_lr`` to tune — which removes the single most fiddly hyperparameter in TB
training. This is why ``objective=vargrad`` is the shipped default.

Two consequences worth knowing:

* It needs ``group_size > 1``. A group of one has zero variance and contributes no gradient;
  the loss handles that gracefully (zero, not NaN) and reports it as a metric, but a run where
  ``singleton_group_fraction`` is high is a run that is not learning.
* The population variance (``ddof=0``) is used deliberately: it is *exactly* TB's loss
  evaluated at the optimal in-batch ``log Z``, which makes the two objectives directly
  comparable and is the identity the unit tests assert.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import torch

from flowcode.objectives.base import (
    DEFAULT_IMPORTANCE_LOG_CLIP,
    DEFAULT_LOG_REWARD_CEILING,
    DEFAULT_LOG_REWARD_FLOOR,
    BaseObjective,
    validate_logprobs,
)
from flowcode.types import Trajectory

__all__ = ["VarGrad"]


class VarGrad(BaseObjective):
    """The logZ-free trajectory-balance loss: within-group variance of ``β log R - Σ log P_F``."""

    def __init__(
        self,
        name: str = "vargrad",
        reward_temperature: float = 1.0,
        off_policy_correction: bool = False,
        log_reward_floor: float = DEFAULT_LOG_REWARD_FLOOR,
        log_reward_ceiling: float = DEFAULT_LOG_REWARD_CEILING,
        importance_log_clip: float = DEFAULT_IMPORTANCE_LOG_CLIP,
    ) -> None:
        """Build the objective.

        Args:
            name: Objective name for logging.
            reward_temperature: ``β`` in ``p(x) ∝ R(x)^β``.
            off_policy_correction: Importance-reweight residuals to the on-policy
                distribution. Off by default — VarGrad is off-policy-consistent.
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

    def loss(
        self, logprobs: list[torch.Tensor], batch: Sequence[Trajectory]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the VarGrad loss, grouping trajectories by ``task_id``.

        Args:
            logprobs: Current-policy per-token logprobs, one 1-D tensor per trajectory.
            batch: The scored trajectories. Trajectories sharing a ``task_id`` form a group
                and share the in-batch ``log Z`` estimate.

        Returns:
            ``(loss, metrics)``. ``metrics["singleton_group_fraction"]`` is the share of
            trajectories that landed in a group of one and therefore contributed nothing —
            watch it, because it is silent otherwise.

        Raises:
            ValueError: If ``logprobs`` and ``batch`` are misaligned.
        """
        validate_logprobs(logprobs, batch)
        device = logprobs[0].device

        forward_logprob = torch.stack([lp.sum() for lp in logprobs])
        log_rewards = self._tempered_log_rewards(batch, device)
        # A_i is TB's residual with log Z removed; its group mean *is* the optimal log Z.
        scores = log_rewards - forward_logprob

        groups: dict[str, list[int]] = defaultdict(list)
        for i, traj in enumerate(batch):
            groups[traj.task_id].append(i)

        weights = self._importance_weights(logprobs, batch)
        squared = scores.new_zeros(len(batch))
        singletons = 0
        log_z_estimates: list[float] = []

        for indices in groups.values():
            if len(indices) < 2:
                singletons += len(indices)
                continue
            index = torch.tensor(indices, device=device, dtype=torch.long)
            member_scores = scores[index]
            if weights is None:
                estimate = member_scores.mean()
            else:
                member_weights = weights[index]
                estimate = (member_scores * member_weights).sum() / member_weights.sum().clamp_min(
                    1e-12
                )
            log_z_estimates.append(float(estimate.detach().item()))
            squared = squared.index_copy(0, index, (member_scores - estimate).square())

        # Mean over trajectories (not over groups) so the gradient scale is independent of how
        # the batch happens to be partitioned. Singleton rows are exact zeros in `squared`.
        if weights is None:
            loss = squared.mean()
        else:
            loss = (squared * weights).sum() / weights.sum().clamp_min(1e-12)

        if singletons == len(batch):
            # Every group had one member: the loss is a genuine, differentiable zero rather
            # than a constant with no grad_fn, so `.backward()` still works and writes zeros.
            loss = loss + 0.0 * forward_logprob.sum()

        metrics = self._base_metrics(loss, log_rewards, weights)
        metrics.update(
            {
                "num_groups": float(len(groups)),
                "singleton_group_fraction": float(singletons) / float(len(batch)),
                "log_z_estimate_mean": (
                    sum(log_z_estimates) / len(log_z_estimates) if log_z_estimates else 0.0
                ),
                "forward_logprob_mean": float(forward_logprob.detach().mean().item()),
                "score_std": float(scores.detach().std(unbiased=False).item())
                if len(batch) > 1
                else 0.0,
            }
        )
        return loss, metrics
