"""Sub-Trajectory Balance, SubTB(λ).

Reference:
    Kanika Madan, Jarrid Rector-Brooks, Maksym Korablyov, Emmanuel Bengio, Moksh Jain,
    Andrei Nica, Tom Bosc, Yoshua Bengio, Nikolay Malkin.
    *Learning GFlowNets from partial episodes for improved convergence and stability.*
    ICML 2023. https://arxiv.org/abs/2209.12782

Where TB imposes one balance constraint per trajectory, SubTB imposes one per *pair* of state
boundaries, weighted by the length of the sub-trajectory between them::

    L = Σ_{i<j} λ^(j-i) ( log F(s_i) + Σ_{t ∈ [i,j)} log P_F(t) - log F(s_j) )²
        ------------------------------------------------------------------------
                                Σ_{i<j} λ^(j-i)

with the two endpoints pinned rather than predicted: ``log F(s_0) = log Z(x)`` (a free
per-task parameter) and ``log F(s_M) = β log R(x)`` (the true terminal reward, a constant).
``λ = 1`` weights every sub-trajectory equally; ``λ → 0`` concentrates on adjacent pairs and
recovers Detailed Balance, which ships separately as
:class:`~flowcode.objectives.db.DetailedBalance`.

Granularity
-----------
One implementation serves both settings. ``granularity="token"`` puts a boundary after every
completion token — the standard LLM-GFlowNet decomposition, ``O(T²)`` pairs. ``"turn"`` uses
the boundaries the environment marked in :attr:`Trajectory.segments`, one state per assistant
turn or tool call, which matches the agentic framing and keeps the pair count tiny. The only
difference is which segment list :func:`~flowcode.objectives.base.resolve_segments` returns;
all the arithmetic below is shared.

Vectorisation
-------------
The naive form is a double loop over ``O(T²)`` pairs, each summing ``O(T)`` logprobs — cubic,
and unusable at ``T = 512``. The trick is that the residual factorises. Writing ``C_k`` for
the cumulative ``Σ log P_F`` at boundary ``k`` (one ``cumsum``) and::

    G_k := log F(s_k) - C_k

the residual of *any* pair collapses to a difference of two entries::

    log F(s_i) + (C_j - C_i) - log F(s_j)  =  G_i - G_j

So the whole pair set is one gather on a length-``M+1`` vector: no double loop, no per-pair
summation, and the cost of a pair is a single subtraction. Pairs are then selected by index
(``triu_indices`` when they all fit, a seeded sample otherwise), which also means the
subsampled path never materialises the ``(M+1)²`` matrix.

Subsampling
-----------
When the pair count exceeds ``max_pairs``, pairs are drawn uniformly at random with a seeded
generator, by sampling flat triangular indices and inverting the triangular number in closed
form — again avoiding any ``O(T²)`` allocation. The full-trajectory pair ``(0, M)``, which is
exactly the TB constraint and the only one anchored to the true reward at both ends, is always
included so the estimator never loses its anchor.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from itertools import chain

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

__all__ = ["SubTB", "triangular_pair_indices"]


def triangular_pair_indices(
    num_states: int,
    max_pairs: int,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """All ``(i, j)`` pairs with ``0 <= i < j <= M``, subsampled when there are too many.

    Args:
        num_states: ``M + 1``, the number of state boundaries including ``s_0`` and ``s_M``.
        max_pairs: Cap on how many pairs to return. Must be at least 1.
        generator: Seeded RNG used only on the subsampling path.
        device: Device for the returned index tensors.

    Returns:
        ``(i, j)`` long tensors of equal length. When the total pair count
        ``M(M+1)/2 <= max_pairs`` this is the exact upper triangle; otherwise it is
        ``max_pairs - 1`` uniformly sampled pairs (with replacement) plus the forced
        full-trajectory pair ``(0, M)``.

    Raises:
        ValueError: If ``num_states < 2`` or ``max_pairs < 1``.
    """
    if num_states < 2:
        raise ValueError(f"need at least two state boundaries, got {num_states}")
    if max_pairs < 1:
        raise ValueError(f"max_pairs must be >= 1, got {max_pairs}")

    last = num_states - 1
    total = num_states * last // 2
    if total <= max_pairs:
        pairs = torch.triu_indices(num_states, num_states, offset=1, device=device)
        return pairs[0], pairs[1]

    # Sample flat indices into the enumeration "for j in 1..M: for i in 0..j-1", whose offset
    # for a given j is the triangular number j(j-1)/2. Inverting that offset recovers j.
    flat = torch.randint(
        0, total, (max_pairs - 1,), generator=generator, dtype=torch.long, device=device
    )
    as_float = flat.to(torch.float64)
    j = torch.floor((1.0 + torch.sqrt(1.0 + 8.0 * as_float)) / 2.0).to(torch.long)
    # float64 sqrt is exact enough for any T we will ever see, but a single Newton-style
    # correction in each direction makes the inversion bulletproof rather than merely likely.
    for _ in range(2):
        j = j.clamp(1, last)
        base = j * (j - 1) // 2
        j = torch.where(flat < base, j - 1, j)
        j = j.clamp(1, last)
        base = j * (j - 1) // 2
        j = torch.where(flat >= base + j, j + 1, j)
    j = j.clamp(1, last)
    i = (flat - j * (j - 1) // 2).clamp(0, last - 1)

    anchor_i = torch.zeros(1, dtype=torch.long, device=device)
    anchor_j = torch.full((1,), last, dtype=torch.long, device=device)
    return torch.cat([i, anchor_i]), torch.cat([j, anchor_j])


class SubTB(BaseObjective):
    """SubTB(λ) over configurable state boundaries, vectorised over sub-trajectory pairs."""

    def __init__(
        self,
        name: str = "subtb",
        lambda_: float = 0.9,
        granularity: Granularity = "token",
        max_pairs: int = 4096,
        reward_temperature: float = 1.0,
        off_policy_correction: bool = False,
        flow: LogFlow | None = None,
        log_reward_floor: float = DEFAULT_LOG_REWARD_FLOOR,
        log_reward_ceiling: float = DEFAULT_LOG_REWARD_CEILING,
        importance_log_clip: float = DEFAULT_IMPORTANCE_LOG_CLIP,
        seed: int = 0,
        log_z: LogZ | None = None,
    ) -> None:
        """Build the objective.

        Args:
            name: Objective name for logging.
            lambda_: Sub-trajectory length discount ``λ ∈ (0, 1]``. Trailing underscore
                because ``lambda`` is a keyword; the Hydra config uses the same spelling.
            granularity: ``"token"`` for a boundary per completion token, ``"turn"`` to use
                the environment's own segmentation.
            max_pairs: Cap on sub-trajectory pairs per trajectory before subsampling kicks in.
            reward_temperature: ``β`` in ``p(x) ∝ R(x)^β``.
            off_policy_correction: Importance-reweight residuals. Off by default.
            flow: The ``log F`` estimator for intermediate states, instantiated by Hydra from
                the ``flow:`` block. Defaults to a fresh
                :class:`~flowcode.objectives.flows.LogFlowEstimator`.
            log_reward_floor: Lower clamp on ``log R`` before tempering.
            log_reward_ceiling: Upper clamp on ``log R`` before tempering.
            importance_log_clip: Symmetric clamp on the log importance ratio.
            seed: Seeds the pair-subsampling RNG, so a run is reproducible.
            log_z: Estimator for ``log F(s_0) = log Z(x)``. Not part of the shipped config —
                the source-state flow is an implementation detail of SubTB — but injectable
                for tests and for sharing one table with a TB baseline.

        Raises:
            ValueError: If ``lambda_`` is outside ``(0, 1]`` or ``max_pairs`` is below 1.
        """
        super().__init__(
            name=name,
            reward_temperature=reward_temperature,
            off_policy_correction=off_policy_correction,
            log_reward_floor=log_reward_floor,
            log_reward_ceiling=log_reward_ceiling,
            importance_log_clip=importance_log_clip,
        )
        if not 0.0 < lambda_ <= 1.0:
            raise ValueError(
                f"lambda_ must lie in (0, 1], got {lambda_}. λ=0 would zero every weight; "
                "use flowcode.objectives.db.DetailedBalance for the adjacent-pair case."
            )
        if max_pairs < 1:
            raise ValueError(f"max_pairs must be >= 1, got {max_pairs}")
        if granularity not in ("token", "turn"):
            raise ValueError(f"granularity must be 'token' or 'turn', got {granularity!r}")
        self.lambda_ = lambda_
        self.granularity: Granularity = granularity
        self.max_pairs = max_pairs
        self.seed = seed
        self.flow: LogFlow = LogFlowEstimator() if flow is None else flow
        self.log_z: LogZ = ConditionalLogZ() if log_z is None else log_z
        self._generator = torch.Generator().manual_seed(seed)

    def loss(
        self, logprobs: list[torch.Tensor], batch: Sequence[Trajectory]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the SubTB(λ) loss.

        Args:
            logprobs: Current-policy per-token logprobs, one 1-D tensor per trajectory.
            batch: The scored trajectories.

        Returns:
            ``(loss, metrics)``. ``metrics["pairs_per_trajectory"]`` reports the realised pair
            count, which is where you see subsampling engage.

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

        # One MLP forward for every intermediate state in the whole minibatch.
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
        pair_counts: list[int] = []
        residual_abs: list[torch.Tensor] = []
        for b in range(len(batch)):
            start, stop = states.traj_slices[b]
            log_flows = torch.cat(
                [
                    log_z[b : b + 1],
                    interior_flows[start:stop],
                    log_rewards[b : b + 1],
                ]
            )
            traj_loss, num_pairs, residuals = self._trajectory_loss(
                log_flows, cumulative_per_traj[b], device
            )
            per_trajectory.append(traj_loss)
            pair_counts.append(num_pairs)
            residual_abs.append(residuals.detach().abs().mean())

        stacked = torch.stack(per_trajectory)
        weights = self._importance_weights(logprobs, batch)
        loss = reduce_per_trajectory(stacked, weights)

        metrics = self._base_metrics(loss, log_rewards, weights)
        metrics.update(
            {
                "log_z_mean": float(log_z.detach().mean().item()),
                "residual_abs_mean": float(torch.stack(residual_abs).mean().item()),
                "pairs_per_trajectory": float(sum(pair_counts)) / float(len(batch)),
                "states_per_trajectory": float(len(states) + 2 * len(batch)) / float(len(batch)),
                "interior_flow_mean": (
                    float(interior_flows.detach().mean().item()) if len(states) else 0.0
                ),
                "subsampled": float(
                    any(
                        len(segs) * (len(segs) + 1) // 2 > self.max_pairs
                        for segs in segments_per_traj
                    )
                ),
            }
        )
        return loss, metrics

    def _trajectory_loss(
        self, log_flows: torch.Tensor, cumulative: torch.Tensor, device: torch.device
    ) -> tuple[torch.Tensor, int, torch.Tensor]:
        """λ-weighted mean squared sub-trajectory residual for one trajectory.

        Args:
            log_flows: ``(M+1,)`` flows at every boundary, endpoints already pinned.
            cumulative: ``(M+1,)`` cumulative forward logprobs at every boundary.
            device: Device for the index tensors.

        Returns:
            ``(loss, num_pairs, residuals)``.
        """
        # The whole vectorisation, in one line: every pair residual is a difference of these.
        potential = log_flows - cumulative
        num_states = potential.shape[0]
        i, j = triangular_pair_indices(num_states, self.max_pairs, self._generator, device)
        residuals = potential[i] - potential[j]
        gap = (j - i).to(potential.dtype)
        if self.lambda_ == 1.0:
            weights = torch.ones_like(gap)
        else:
            weights = torch.exp(gap * math.log(self.lambda_))
        loss = (weights * residuals.square()).sum() / weights.sum().clamp_min(1e-12)
        return loss, int(i.shape[0]), residuals

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
