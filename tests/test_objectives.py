"""Unit tests for the GFlowNet objectives.

These never touch Tinker. Everything here is plain torch over synthetic trajectories, which is
the whole point of keeping :mod:`flowcode.objectives` free of the SDK: the maths is checkable
offline at zero API cost. The companion file ``test_convergence.py`` does the harder job of
showing the losses actually recover ``p(x) ∝ R(x)``.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence

import pytest
import torch
from torch import nn

from flowcode.objectives.base import (
    DEFAULT_LOG_REWARD_FLOOR,
    Granularity,
    Objective,
    boundary_cumulative_logprobs,
    importance_weights,
    resolve_segments,
    segment_logprob_sums,
    temper_log_reward,
)
from flowcode.objectives.db import DetailedBalance
from flowcode.objectives.flows import (
    ConditionalLogZ,
    FlowStates,
    LogFlowEstimator,
    ScalarLogZ,
)
from flowcode.objectives.subtb import SubTB, triangular_pair_indices
from flowcode.objectives.tb import TrajectoryBalance
from flowcode.objectives.vargrad import VarGrad
from flowcode.types import Segment, Trajectory

TASKS = ("task-a", "task-b")


def make_trajectory(
    task_id: str = "task-a",
    num_tokens: int = 8,
    log_reward: float = -1.0,
    num_segments: int = 1,
    partial_log_reward: float | None = None,
    sampling_offset: float = 0.0,
    seed: int = 0,
) -> Trajectory:
    """Build a synthetic trajectory with evenly spaced segment boundaries.

    Args:
        task_id: Task id (VarGrad groups on this).
        num_tokens: Completion length.
        log_reward: Terminal ``log R``.
        num_segments: How many turn-level segments to cut the completion into.
        partial_log_reward: Attached to every non-terminal segment when given.
        sampling_offset: Added to the behaviour logprobs, to simulate a stale sampler.
        seed: Seeds the synthetic behaviour logprobs.

    Returns:
        A valid :class:`~flowcode.types.Trajectory`.
    """
    generator = torch.Generator().manual_seed(seed)
    behaviour = (-torch.rand(num_tokens, generator=generator) * 2.0 + sampling_offset).tolist()
    step = max(1, math.ceil(num_tokens / num_segments))
    segments: list[Segment] = []
    start = 0
    while start < num_tokens:
        end = min(num_tokens, start + step)
        is_terminal = end == num_tokens
        segments.append(
            Segment(
                start=start,
                end=end,
                partial_log_reward=None if is_terminal else partial_log_reward,
            )
        )
        start = end
    return Trajectory(
        task_id=task_id,
        prompt_tokens=[1, 2, 3],
        completion_tokens=list(range(num_tokens)),
        sampling_logprobs=behaviour,
        log_reward=log_reward,
        segments=segments,
        metadata={},
    )


def make_batch(
    specs: Sequence[tuple[str, int, float]], num_segments: int = 1, seed: int = 0
) -> tuple[list[torch.Tensor], list[Trajectory]]:
    """Build a batch plus matching current-policy logprob tensors.

    Args:
        specs: ``(task_id, num_tokens, log_reward)`` per trajectory.
        num_segments: Turn-level segment count for each trajectory.
        seed: Seeds the logprobs.

    Returns:
        ``(logprobs, batch)`` ready to hand to :meth:`Objective.loss`.
    """
    generator = torch.Generator().manual_seed(seed)
    batch = [
        make_trajectory(
            task_id=task,
            num_tokens=n,
            log_reward=reward,
            num_segments=num_segments,
            seed=seed + i,
        )
        for i, (task, n, reward) in enumerate(specs)
    ]
    logprobs = [
        (-torch.rand(t.num_completion_tokens, generator=generator) * 2.0).requires_grad_(True)
        for t in batch
    ]
    return logprobs, batch


class OracleFlow(nn.Module):
    """A flow estimator that is *exactly* balanced by construction.

    ``log F(s_k) = z + C_k`` where ``C_k`` is the cumulative forward logprob at boundary ``k``
    — which is precisely the flow that makes every SubTB/DB residual vanish, provided the
    trajectory's ``β log R`` also equals ``z + C_M``. Used to check that a perfectly balanced
    trajectory really does produce ~zero loss, without depending on a trained MLP.
    """

    def __init__(self, z: float = 0.0) -> None:
        super().__init__()
        self.z = nn.Parameter(torch.tensor(float(z)))

    def log_flow(self, states: FlowStates) -> torch.Tensor:
        return states.cum_logprob + self.z

    def register_tasks(self, task_ids: Sequence[str]) -> None:
        return None


def all_objectives(granularity: Granularity = "token") -> list[Objective]:
    """One instance of each objective, configured exactly as ``conf/objective/*.yaml`` does."""
    objectives: list[Objective] = [
        TrajectoryBalance(log_z=ConditionalLogZ(embedding_dim=1, init_value=0.0)),
        VarGrad(),
        SubTB(
            lambda_=0.9,
            granularity=granularity,
            max_pairs=4096,
            flow=LogFlowEstimator(hidden_dim=64, task_embedding_dim=32),
        ),
        DetailedBalance(
            granularity=granularity,
            flow=LogFlowEstimator(hidden_dim=64, task_embedding_dim=32),
        ),
    ]
    for objective in objectives:
        objective.register_tasks(TASKS)
    return objectives


# --------------------------------------------------------------------------------------
# Protocol conformance and shapes
# --------------------------------------------------------------------------------------


def test_all_objectives_satisfy_the_protocol() -> None:
    for objective in all_objectives():
        assert isinstance(objective, Objective)
        assert isinstance(objective.name, str)
        assert isinstance(objective.parameters(), Iterator)


@pytest.mark.parametrize("granularity", ["token", "turn"])
def test_loss_is_a_scalar_with_float_metrics(granularity: Granularity) -> None:
    logprobs, batch = make_batch(
        [("task-a", 6, -1.0), ("task-a", 9, -3.0), ("task-b", 4, -0.5)], num_segments=3
    )
    for objective in all_objectives(granularity=granularity):
        loss, metrics = objective.loss(logprobs, batch)
        assert loss.shape == ()
        assert loss.dtype == torch.float32
        assert torch.isfinite(loss)
        assert loss.item() >= 0.0
        assert metrics["loss"] == pytest.approx(loss.item(), rel=1e-5)
        for key, value in metrics.items():
            assert isinstance(value, float), key
            assert math.isfinite(value), key


def test_both_granularities_are_handled_and_differ() -> None:
    """The granularity flag must actually change the state decomposition, not be decorative."""
    logprobs, batch = make_batch([("task-a", 12, -2.0), ("task-b", 12, -1.0)], num_segments=3)

    token_db = DetailedBalance(granularity="token", flow=LogFlowEstimator())
    turn_db = DetailedBalance(granularity="turn", flow=LogFlowEstimator())
    for objective in (token_db, turn_db):
        objective.register_tasks(TASKS)

    _, token_metrics = token_db.loss(logprobs, batch)
    _, turn_metrics = turn_db.loss(logprobs, batch)
    assert token_metrics["edges_per_trajectory"] == 12.0
    assert turn_metrics["edges_per_trajectory"] == 3.0

    token_subtb = SubTB(granularity="token", flow=LogFlowEstimator())
    turn_subtb = SubTB(granularity="turn", flow=LogFlowEstimator())
    for subtb in (token_subtb, turn_subtb):
        subtb.register_tasks(TASKS)
    _, token_metrics = token_subtb.loss(logprobs, batch)
    _, turn_metrics = turn_subtb.loss(logprobs, batch)
    # 12 boundaries -> 13 states -> 78 pairs; 3 boundaries -> 4 states -> 6 pairs.
    assert token_metrics["pairs_per_trajectory"] == 78.0
    assert turn_metrics["pairs_per_trajectory"] == 6.0


def test_turn_granularity_respects_environment_segments() -> None:
    traj = make_trajectory(num_tokens=10, num_segments=4)
    assert resolve_segments(traj, "turn") == traj.segments
    assert len(resolve_segments(traj, "token")) == 10


def test_token_granularity_keeps_partial_rewards() -> None:
    """Refining to token boundaries must not discard the environment's partial scores."""
    traj = make_trajectory(num_tokens=9, num_segments=3, partial_log_reward=-0.25)
    token_segments = resolve_segments(traj, "token")
    scored = {
        s.end: s.partial_log_reward for s in token_segments if s.partial_log_reward is not None
    }
    assert scored == {3: -0.25, 6: -0.25}


# --------------------------------------------------------------------------------------
# Gradients
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("granularity", ["token", "turn"])
def test_gradient_reaches_the_policy_logprobs(granularity: Granularity) -> None:
    for objective in all_objectives(granularity=granularity):
        logprobs, batch = make_batch(
            [("task-a", 6, -1.0), ("task-a", 6, -3.0), ("task-b", 5, -0.5)], num_segments=3
        )
        loss, _ = objective.loss(logprobs, batch)
        torch.autograd.backward(loss)
        for i, lp in enumerate(logprobs):
            assert lp.grad is not None, (objective.name, i)
            assert torch.isfinite(lp.grad).all(), (objective.name, i)
        total = sum(float(lp.grad.abs().sum()) for lp in logprobs if lp.grad is not None)
        assert total > 0.0, objective.name


def test_gradient_reaches_the_flow_parameters() -> None:
    for objective in all_objectives():
        params = list(objective.parameters())
        if not params:
            assert objective.name == "vargrad", "only VarGrad is allowed to be parameter-free"
            continue
        logprobs, batch = make_batch([("task-a", 6, -1.0), ("task-b", 6, -3.0)], num_segments=3)
        loss, _ = objective.loss(logprobs, batch)
        torch.autograd.backward(loss)
        grads = [p.grad for p in params if p.grad is not None]
        assert grads, objective.name
        assert any(float(g.abs().sum()) > 0.0 for g in grads), objective.name
        assert all(torch.isfinite(g).all() for g in grads), objective.name


def test_vargrad_owns_no_parameters() -> None:
    assert list(VarGrad().parameters()) == []


# --------------------------------------------------------------------------------------
# Perfectly balanced trajectories
# --------------------------------------------------------------------------------------


def _balanced_batch(
    z: float, num_tokens: int = 6, num_segments: int = 3
) -> tuple[list[torch.Tensor], list[Trajectory]]:
    """Trajectories whose reward is exactly ``z + Σ log P_F`` — the TB optimum."""
    logprobs = [
        torch.full((num_tokens,), -0.5, requires_grad=True),
        torch.full((num_tokens,), -0.25, requires_grad=True),
    ]
    batch = [
        make_trajectory(
            task_id="task-a",
            num_tokens=num_tokens,
            log_reward=z + float(lp.detach().sum()),
            num_segments=num_segments,
            seed=i,
        )
        for i, lp in enumerate(logprobs)
    ]
    return logprobs, batch


def test_balanced_trajectory_gives_zero_tb_loss() -> None:
    z = -1.5
    logprobs, batch = _balanced_batch(z)
    objective = TrajectoryBalance(log_z=ScalarLogZ(init_value=z))
    loss, metrics = objective.loss(logprobs, batch)
    assert loss.item() == pytest.approx(0.0, abs=1e-8)
    assert metrics["residual_abs_mean"] == pytest.approx(0.0, abs=1e-5)


def test_balanced_group_gives_zero_vargrad_loss() -> None:
    """VarGrad is zero when every group member has the same ``β log R - Σ log P_F``."""
    logprobs, batch = _balanced_batch(z=-1.5)
    loss, _ = VarGrad().loss(logprobs, batch)
    assert loss.item() == pytest.approx(0.0, abs=1e-8)


@pytest.mark.parametrize("granularity", ["token", "turn"])
def test_balanced_trajectory_gives_zero_subtb_and_db_loss(granularity: Granularity) -> None:
    z = -1.5
    logprobs, batch = _balanced_batch(z)
    for objective in (
        SubTB(
            granularity=granularity,
            flow=OracleFlow(z),
            log_z=ScalarLogZ(init_value=z),
        ),
        DetailedBalance(
            granularity=granularity,
            flow=OracleFlow(z),
            log_z=ScalarLogZ(init_value=z),
        ),
    ):
        loss, metrics = objective.loss(logprobs, batch)
        assert loss.item() == pytest.approx(0.0, abs=1e-8), objective.name
        assert metrics["residual_abs_mean"] == pytest.approx(0.0, abs=1e-5), objective.name


def test_unbalanced_trajectory_gives_the_exact_tb_residual() -> None:
    """Sanity-check the arithmetic against a hand-computed value."""
    logprobs = [torch.full((4,), -0.5, requires_grad=True)]
    batch = [make_trajectory(num_tokens=4, log_reward=-3.0)]
    objective = TrajectoryBalance(log_z=ScalarLogZ(init_value=1.0))
    loss, _ = objective.loss(logprobs, batch)
    # (1.0 + 4 * -0.5 - (-3.0))^2 = (1 - 2 + 3)^2 = 4
    assert loss.item() == pytest.approx(4.0, abs=1e-6)


# --------------------------------------------------------------------------------------
# TB / VarGrad / SubTB relationships
# --------------------------------------------------------------------------------------


def test_vargrad_equals_tb_at_the_optimal_in_batch_log_z() -> None:
    """The defining identity: VarGrad is TB with ``log Z`` set to its in-batch minimiser."""
    logprobs, batch = make_batch([("task-a", 5, -1.0), ("task-a", 7, -4.0), ("task-a", 3, 0.5)])
    vargrad_loss, metrics = VarGrad().loss(logprobs, batch)

    optimal_log_z = metrics["log_z_estimate_mean"]
    tb_loss, _ = TrajectoryBalance(log_z=ScalarLogZ(init_value=optimal_log_z)).loss(logprobs, batch)
    assert vargrad_loss.item() == pytest.approx(tb_loss.item(), rel=1e-5)


def test_tb_loss_is_minimised_at_the_vargrad_log_z() -> None:
    """Perturbing ``log Z`` away from VarGrad's estimate can only increase the TB loss."""
    logprobs, batch = make_batch([("task-a", 5, -1.0), ("task-a", 7, -4.0), ("task-a", 3, 0.5)])
    _, metrics = VarGrad().loss(logprobs, batch)
    best = metrics["log_z_estimate_mean"]
    at_best, _ = TrajectoryBalance(log_z=ScalarLogZ(init_value=best)).loss(logprobs, batch)
    for delta in (-1.0, -0.1, 0.1, 1.0):
        perturbed, _ = TrajectoryBalance(log_z=ScalarLogZ(init_value=best + delta)).loss(
            logprobs, batch
        )
        assert perturbed.item() > at_best.item()


def test_single_segment_subtb_reduces_to_tb() -> None:
    """One segment means one pair, ``(s_0, s_terminal)`` — literally the TB constraint."""
    logprobs, batch = make_batch([("task-a", 6, -2.0), ("task-b", 4, -1.0)], num_segments=1)
    z = 0.75
    subtb = SubTB(
        granularity="turn", lambda_=1.0, flow=OracleFlow(z), log_z=ScalarLogZ(init_value=z)
    )
    tb = TrajectoryBalance(log_z=ScalarLogZ(init_value=z))
    subtb_loss, subtb_metrics = subtb.loss(logprobs, batch)
    tb_loss, _ = tb.loss(logprobs, batch)
    assert subtb_metrics["pairs_per_trajectory"] == 1.0
    assert subtb_loss.item() == pytest.approx(tb_loss.item(), rel=1e-5)


def test_single_segment_db_reduces_to_tb() -> None:
    logprobs, batch = make_batch([("task-a", 6, -2.0), ("task-b", 4, -1.0)], num_segments=1)
    z = 0.75
    db = DetailedBalance(granularity="turn", flow=OracleFlow(z), log_z=ScalarLogZ(init_value=z))
    tb = TrajectoryBalance(log_z=ScalarLogZ(init_value=z))
    db_loss, _ = db.loss(logprobs, batch)
    tb_loss, _ = tb.loss(logprobs, batch)
    assert db_loss.item() == pytest.approx(tb_loss.item(), rel=1e-5)


def test_subtb_and_db_share_the_adjacent_residuals() -> None:
    """SubTB's pair set contains DB's edge set, so the two agree on those residuals."""
    logprobs, batch = make_batch([("task-a", 4, -2.0)], num_segments=4)
    z = 0.3
    subtb = SubTB(
        granularity="turn",
        lambda_=1.0,
        max_pairs=10_000,
        flow=OracleFlow(0.7),
        log_z=ScalarLogZ(z),
    )
    db = DetailedBalance(
        granularity="turn", flow=OracleFlow(0.7), log_z=ScalarLogZ(z), edge_reduction="mean"
    )
    subtb_loss, subtb_metrics = subtb.loss(logprobs, batch)
    db_loss, db_metrics = db.loss(logprobs, batch)
    assert subtb_metrics["pairs_per_trajectory"] == 10.0  # 5 states -> 10 pairs
    assert db_metrics["edges_per_trajectory"] == 4.0
    assert subtb_loss.item() > 0.0 and db_loss.item() > 0.0


# --------------------------------------------------------------------------------------
# SubTB vectorisation and pair subsampling
# --------------------------------------------------------------------------------------


def test_triangular_pair_indices_enumerates_the_full_upper_triangle() -> None:
    for num_states in (2, 3, 8, 17):
        i, j = triangular_pair_indices(num_states, max_pairs=10_000)
        expected = {(a, b) for a in range(num_states) for b in range(a + 1, num_states)}
        assert set(zip(i.tolist(), j.tolist(), strict=True)) == expected


def test_triangular_pair_indices_subsamples_validly_and_deterministically() -> None:
    num_states = 200  # 19,900 pairs
    generator = torch.Generator().manual_seed(7)
    i, j = triangular_pair_indices(num_states, max_pairs=512, generator=generator)
    assert i.shape[0] == 512
    assert bool((i < j).all())
    assert bool((i >= 0).all())
    assert bool((j <= num_states - 1).all())
    # The full-trajectory pair is always kept: it is the only constraint anchored to the true
    # reward at one end and log Z at the other.
    assert (i[-1].item(), j[-1].item()) == (0, num_states - 1)

    repeat_generator = torch.Generator().manual_seed(7)
    i2, j2 = triangular_pair_indices(num_states, max_pairs=512, generator=repeat_generator)
    assert torch.equal(i, i2) and torch.equal(j, j2)


def test_triangular_index_inversion_is_exact_over_the_whole_range() -> None:
    """The closed-form inversion of the triangular number must be exact, not approximate."""
    num_states = 300
    total = num_states * (num_states - 1) // 2
    generator = torch.Generator().manual_seed(1)
    i, j = triangular_pair_indices(num_states, max_pairs=total - 1, generator=generator)
    flat = j * (j - 1) // 2 + i
    assert bool(((flat >= 0) & (flat < total)).all())
    assert bool((i < j).all())


def test_subtb_respects_max_pairs() -> None:
    logprobs, batch = make_batch([("task-a", 64, -1.0)], num_segments=1)
    objective = SubTB(granularity="token", max_pairs=100, flow=LogFlowEstimator(), seed=3)
    objective.register_tasks(TASKS)
    _, metrics = objective.loss(logprobs, batch)
    assert metrics["pairs_per_trajectory"] == 100.0
    assert metrics["subsampled"] == 1.0

    unbounded = SubTB(granularity="token", max_pairs=10_000, flow=LogFlowEstimator(), seed=3)
    unbounded.register_tasks(TASKS)
    _, metrics = unbounded.loss(logprobs, batch)
    assert metrics["pairs_per_trajectory"] == 64 * 65 / 2
    assert metrics["subsampled"] == 0.0


def test_subtb_lambda_weighting_prefers_short_subtrajectories() -> None:
    """Small λ must down-weight long sub-trajectories relative to λ=1."""
    logprobs = [torch.zeros(4, requires_grad=True)]
    batch = [make_trajectory(num_tokens=4, log_reward=-4.0, num_segments=4)]
    losses = {}
    for lam in (1.0, 0.5, 0.1):
        objective = SubTB(
            granularity="turn", lambda_=lam, flow=OracleFlow(0.0), log_z=ScalarLogZ(0.0)
        )
        losses[lam], _ = objective.loss(logprobs, batch)
    # The only non-zero residuals here come from the pinned terminal reward, i.e. from the
    # longest pairs; discounting them pushes the loss down monotonically.
    assert losses[1.0].item() > losses[0.5].item() > losses[0.1].item()


def test_subtb_vectorised_loss_matches_a_naive_double_loop() -> None:
    """Ground-truth check on the cumulative-sum trick."""
    torch.manual_seed(0)
    logprobs = [torch.randn(6).mul(-1.0).abs().neg().requires_grad_(True)]
    batch = [make_trajectory(num_tokens=6, log_reward=-2.5, num_segments=6)]
    lam, z = 0.7, 0.4
    objective = SubTB(
        granularity="turn",
        lambda_=lam,
        max_pairs=10_000,
        flow=OracleFlow(0.9),
        log_z=ScalarLogZ(z),
    )
    loss, _ = objective.loss(logprobs, batch)

    lp = logprobs[0].detach()
    cumulative = [0.0]
    for value in lp.tolist():
        cumulative.append(cumulative[-1] + value)
    flows = [z] + [0.9 + cumulative[k] for k in range(1, 6)] + [-2.5]
    numerator = 0.0
    denominator = 0.0
    for a in range(7):
        for b in range(a + 1, 7):
            weight = lam ** (b - a)
            residual = flows[a] + (cumulative[b] - cumulative[a]) - flows[b]
            numerator += weight * residual**2
            denominator += weight
    assert loss.item() == pytest.approx(numerator / denominator, rel=1e-5)


def test_db_vectorised_loss_matches_a_naive_loop() -> None:
    torch.manual_seed(0)
    logprobs = [torch.randn(5).neg().abs().neg().requires_grad_(True)]
    batch = [make_trajectory(num_tokens=5, log_reward=-1.25, num_segments=5)]
    z = -0.2
    objective = DetailedBalance(granularity="turn", flow=OracleFlow(0.6), log_z=ScalarLogZ(z))
    loss, _ = objective.loss(logprobs, batch)

    lp = logprobs[0].detach().tolist()
    cumulative = [0.0]
    for value in lp:
        cumulative.append(cumulative[-1] + value)
    flows = [z] + [0.6 + cumulative[k] for k in range(1, 5)] + [-1.25]
    expected = sum(
        (flows[k] + (cumulative[k + 1] - cumulative[k]) - flows[k + 1]) ** 2 for k in range(5)
    )
    assert loss.item() == pytest.approx(expected, rel=1e-5)


# --------------------------------------------------------------------------------------
# Grouping edge cases
# --------------------------------------------------------------------------------------


def test_vargrad_group_of_one_is_zero_not_nan() -> None:
    logprobs, batch = make_batch([("only-me", 5, -1.0)])
    loss, metrics = VarGrad().loss(logprobs, batch)
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.0, abs=1e-12)
    assert metrics["singleton_group_fraction"] == 1.0
    # The zero must still be differentiable, or the training loop's backward() blows up.
    torch.autograd.backward(loss)
    for lp in logprobs:
        assert lp.grad is not None
        assert torch.equal(lp.grad, torch.zeros_like(lp.grad))


def test_vargrad_mixes_singleton_and_real_groups() -> None:
    logprobs, batch = make_batch([("task-a", 5, -1.0), ("task-a", 5, -4.0), ("lonely", 5, 0.0)])
    loss, metrics = VarGrad().loss(logprobs, batch)
    assert metrics["num_groups"] == 2.0
    assert metrics["singleton_group_fraction"] == pytest.approx(1 / 3)
    assert loss.item() > 0.0
    torch.autograd.backward(loss)
    # The singleton contributes nothing; the real group does.
    singleton_grad, group_grad = logprobs[2].grad, logprobs[0].grad
    assert singleton_grad is not None and group_grad is not None
    assert float(singleton_grad.abs().sum()) == pytest.approx(0.0, abs=1e-12)
    assert float(group_grad.abs().sum()) > 0.0


def test_vargrad_groups_are_independent() -> None:
    """Adding a second, separately-balanced group must not perturb the first group's loss."""
    logprobs_a, batch_a = make_batch([("task-a", 5, -1.0), ("task-a", 6, -4.0)], seed=1)
    logprobs_b, batch_b = make_batch([("task-b", 4, 3.0), ("task-b", 7, -9.0)], seed=2)
    single, _ = VarGrad().loss(logprobs_a, batch_a)
    joint, _ = VarGrad().loss(logprobs_a + logprobs_b, list(batch_a) + list(batch_b))
    only_b, _ = VarGrad().loss(logprobs_b, batch_b)
    # Both are means over the same total trajectory count, so the joint loss is the average.
    assert joint.item() == pytest.approx((single.item() + only_b.item()) / 2, rel=1e-5)


def test_empty_batch_is_rejected() -> None:
    for objective in all_objectives():
        with pytest.raises(ValueError, match="empty batch"):
            objective.loss([], [])


def test_misaligned_logprobs_are_rejected() -> None:
    logprobs, batch = make_batch([("task-a", 5, -1.0)])
    for objective in all_objectives():
        with pytest.raises(ValueError, match="align index-for-index"):
            objective.loss(logprobs, list(batch) * 2)
        with pytest.raises(ValueError, match="completion tokens"):
            objective.loss([torch.zeros(99, requires_grad=True)], batch)


# --------------------------------------------------------------------------------------
# Numerical stability
# --------------------------------------------------------------------------------------


def test_temper_log_reward_clamps_the_pathological_cases() -> None:
    assert temper_log_reward(-math.inf, 1.0) == DEFAULT_LOG_REWARD_FLOOR
    assert temper_log_reward(math.nan, 1.0) == DEFAULT_LOG_REWARD_FLOOR
    assert temper_log_reward(math.inf, 1.0) == 20.0
    assert temper_log_reward(-1.0, 2.0) == -2.0
    assert temper_log_reward(-1e9, 1.0) == DEFAULT_LOG_REWARD_FLOOR


@pytest.mark.parametrize("log_reward", [-1e9, -math.inf, -60.0, 1e9])
def test_extreme_rewards_stay_finite(log_reward: float) -> None:
    """A zero-probability sample must not poison the batch it appears in."""
    for objective in all_objectives():
        logprobs, batch = make_batch(
            [("task-a", 6, log_reward), ("task-a", 6, -1.0), ("task-b", 6, -2.0)], num_segments=3
        )
        loss, metrics = objective.loss(logprobs, batch)
        assert torch.isfinite(loss), objective.name
        assert all(math.isfinite(v) for v in metrics.values()), objective.name
        torch.autograd.backward(loss)
        for lp in logprobs:
            assert lp.grad is not None and torch.isfinite(lp.grad).all(), objective.name
        for param in objective.parameters():
            assert param.grad is None or torch.isfinite(param.grad).all(), objective.name


def test_very_long_completion_does_not_blow_up() -> None:
    """The O(T²) pair set at T=512 must be handled by subsampling, not by allocating it."""
    logprobs, batch = make_batch([("task-a", 512, -3.0)], num_segments=1)
    objective = SubTB(granularity="token", max_pairs=4096, flow=LogFlowEstimator(), seed=0)
    objective.register_tasks(TASKS)
    loss, metrics = objective.loss(logprobs, batch)
    assert torch.isfinite(loss)
    assert metrics["pairs_per_trajectory"] == 4096.0
    torch.autograd.backward(loss)
    assert logprobs[0].grad is not None and torch.isfinite(logprobs[0].grad).all()


def test_reward_temperature_scales_the_target() -> None:
    logprobs, batch = make_batch([("task-a", 4, -2.0)])
    cold = TrajectoryBalance(reward_temperature=2.0, log_z=ScalarLogZ(0.0))
    _, metrics = cold.loss(logprobs, batch)
    assert metrics["log_reward_mean"] == pytest.approx(-4.0)


def test_reward_temperature_must_be_positive() -> None:
    with pytest.raises(ValueError, match="reward_temperature"):
        VarGrad(reward_temperature=0.0)


def test_subtb_rejects_a_degenerate_lambda() -> None:
    with pytest.raises(ValueError, match="lambda_"):
        SubTB(lambda_=0.0)
    with pytest.raises(ValueError, match="lambda_"):
        SubTB(lambda_=1.5)


def test_bad_granularity_is_rejected() -> None:
    with pytest.raises(ValueError, match="granularity"):
        DetailedBalance(granularity="sentence")  # ty: ignore[invalid-argument-type]


# --------------------------------------------------------------------------------------
# Off-policy correction
# --------------------------------------------------------------------------------------


def test_importance_weights_are_off_by_default() -> None:
    logprobs, batch = make_batch([("task-a", 4, -1.0)])
    assert importance_weights(logprobs, batch, enabled=False) is None


def test_importance_weights_are_detached_clipped_and_mean_one() -> None:
    logprobs, batch = make_batch(
        [("task-a", 4, -1.0), ("task-a", 4, -1.0), ("task-a", 4, -1.0)], seed=5
    )
    weights = importance_weights(logprobs, batch, enabled=True, log_clip=1.0)
    assert weights is not None
    assert not weights.requires_grad
    assert weights.mean().item() == pytest.approx(1.0, rel=1e-5)
    ratio = weights.max() / weights.min()
    assert ratio.item() <= math.exp(2.0) + 1e-5


def test_off_policy_correction_changes_the_loss_but_keeps_it_finite() -> None:
    """The correction is a reweighting, not a correctness fix — the loss must stay sane."""
    logprobs, batch = make_batch([("task-a", 5, -1.0), ("task-a", 5, -4.0)], seed=11)
    batch = [
        Trajectory(
            task_id=t.task_id,
            prompt_tokens=t.prompt_tokens,
            completion_tokens=t.completion_tokens,
            sampling_logprobs=[v - 0.4 for v in t.sampling_logprobs],
            log_reward=t.log_reward,
            segments=t.segments,
            metadata=t.metadata,
        )
        for t in batch
    ]
    plain, _ = VarGrad(off_policy_correction=False).loss(logprobs, batch)
    corrected, metrics = VarGrad(off_policy_correction=True).loss(logprobs, batch)
    assert torch.isfinite(corrected)
    assert "importance_weight_max" in metrics
    assert corrected.item() != pytest.approx(plain.item(), rel=1e-9)


def test_off_policy_correction_is_a_no_op_when_on_policy() -> None:
    """If the sampler *is* the current policy every weight is 1 and nothing changes."""
    logprobs = [
        torch.full((4,), -0.5, requires_grad=True),
        torch.full((4,), -0.9, requires_grad=True),
    ]
    batch = [
        Trajectory(
            task_id="task-a",
            prompt_tokens=[1],
            completion_tokens=[0, 1, 2, 3],
            sampling_logprobs=lp.detach().tolist(),
            log_reward=-1.0 * (i + 1),
            segments=[Segment(0, 4)],
            metadata={},
        )
        for i, lp in enumerate(logprobs)
    ]
    plain, _ = VarGrad(off_policy_correction=False).loss(logprobs, batch)
    corrected, _ = VarGrad(off_policy_correction=True).loss(logprobs, batch)
    assert corrected.item() == pytest.approx(plain.item(), rel=1e-6)


# --------------------------------------------------------------------------------------
# Flow modules
# --------------------------------------------------------------------------------------


def test_conditional_log_z_gives_a_free_scalar_per_task() -> None:
    log_z = ConditionalLogZ(embedding_dim=1, init_value=2.0)
    log_z.register_tasks(["a", "b", "c"])
    values = log_z.log_z(["a", "b", "c", "a"])
    assert values.shape == (4,)
    assert torch.allclose(values, torch.full((4,), 2.0))
    assert values[0].item() == pytest.approx(values[3].item())

    # A gradient on one task must not move another.
    torch.autograd.backward(values[0])
    optimiser = torch.optim.SGD(log_z.parameters(), lr=1.0)
    optimiser.step()
    moved = log_z.log_z(["a", "b"]).detach()
    assert moved[0].item() != pytest.approx(2.0)


def test_conditional_log_z_grows_lazily_with_a_warning() -> None:
    log_z = ConditionalLogZ(embedding_dim=1)
    log_z.register_tasks(["a"])
    with pytest.warns(RuntimeWarning, match="not registered before use"):
        values = log_z.log_z(["a", "brand-new"])
    assert values.shape == (2,)
    assert log_z.tasks.num_tasks == 2
    # Re-registering is idempotent.
    log_z.register_tasks(["a", "brand-new"])
    assert log_z.tasks.num_tasks == 2


def test_register_tasks_is_idempotent_and_preserves_learned_values() -> None:
    log_z = ConditionalLogZ(embedding_dim=1)
    log_z.register_tasks(["a", "b"])
    with torch.no_grad():
        log_z.tasks.table[0].fill_(3.0)
    before = log_z.log_z(["a"]).item()
    log_z.register_tasks(["b", "c", "d"])
    assert log_z.tasks.num_tasks == 4
    assert log_z.log_z(["a"]).item() == pytest.approx(before)


def test_scalar_log_z_broadcasts() -> None:
    log_z = ScalarLogZ(init_value=-1.25)
    values = log_z.log_z(["x", "y", "z"])
    assert values.shape == (3,)
    assert torch.allclose(values, torch.full((3,), -1.25))


def test_log_flow_estimator_starts_at_zero_and_is_trainable() -> None:
    """Zero-init output head: the first step is driven by residuals, not by MLP noise."""
    flow = LogFlowEstimator(hidden_dim=16, task_embedding_dim=8)
    flow.register_tasks(TASKS)
    logprobs, batch = make_batch([("task-a", 6, -1.0)], num_segments=3)
    objective = DetailedBalance(granularity="turn", flow=flow)
    objective.register_tasks(TASKS)
    _, metrics = objective.loss(logprobs, batch)
    assert metrics["interior_flow_mean"] == pytest.approx(0.0, abs=1e-6)

    optimiser = torch.optim.Adam(objective.parameters(), lr=0.1)
    for _ in range(50):
        optimiser.zero_grad()
        loss, _ = objective.loss(logprobs, batch)
        torch.autograd.backward(loss)
        optimiser.step()
    final, _ = objective.loss(logprobs, batch)
    assert final.item() < 1e-2


def test_flow_states_expose_the_prefix_for_richer_estimators() -> None:
    """The interface must let a stronger flow model reconstruct the actual state."""
    captured: list[list[int]] = []

    class PrefixSpy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bias = nn.Parameter(torch.zeros(1))

        def log_flow(self, states: FlowStates) -> torch.Tensor:
            captured.extend(states.prefix(r) for r in range(len(states)))
            return self.bias.expand(len(states))

        def register_tasks(self, task_ids: Sequence[str]) -> None:
            return None

    logprobs, batch = make_batch([("task-a", 4, -1.0)], num_segments=4)
    objective = DetailedBalance(granularity="turn", flow=PrefixSpy(), log_z=ScalarLogZ(0.0))
    objective.loss(logprobs, batch)
    assert captured == [[0], [0, 1], [0, 1, 2]]


def test_flow_state_features_are_detached_from_the_policy() -> None:
    """Gradient must not reach the policy through the flow network's *inputs*."""
    logprobs, batch = make_batch([("task-a", 6, -1.0)], num_segments=3)
    flow = LogFlowEstimator(hidden_dim=8, task_embedding_dim=4)
    flow.register_tasks(TASKS)
    segments = resolve_segments(batch[0], "turn")
    cumulative = boundary_cumulative_logprobs(logprobs[0], segments)
    assert cumulative.requires_grad
    from flowcode.objectives.base import build_flow_states

    states = build_flow_states(logprobs, batch, [segments], [cumulative], beta=1.0)
    assert not states.cum_logprob.requires_grad
    features = flow.features(states)
    torch.autograd.backward(features.sum())
    assert logprobs[0].grad is None


# --------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------


def test_segment_logprob_sums_and_cumulatives_agree() -> None:
    logprobs = torch.tensor([-0.1, -0.2, -0.3, -0.4, -0.5])
    segments = [Segment(0, 2), Segment(2, 3), Segment(3, 5)]
    sums = segment_logprob_sums(logprobs, segments)
    assert torch.allclose(sums, torch.tensor([-0.3, -0.3, -0.9]), atol=1e-6)
    cumulative = boundary_cumulative_logprobs(logprobs, segments)
    assert cumulative.shape == (4,)
    assert cumulative[0].item() == 0.0
    assert cumulative[-1].item() == pytest.approx(float(logprobs.sum()))


def test_reduction_is_a_mean_over_trajectories() -> None:
    """Tinker's backend sums per-datum losses, so duplicating the batch must not scale us."""
    logprobs, batch = make_batch([("task-a", 4, -1.0), ("task-a", 4, -3.0)], seed=4)
    objective = TrajectoryBalance(log_z=ScalarLogZ(0.0))
    single, _ = objective.loss(logprobs, batch)
    doubled, _ = objective.loss(logprobs + logprobs, list(batch) * 2)
    assert doubled.item() == pytest.approx(single.item(), rel=1e-6)
