"""The test that proves the objectives are correct, with no API anywhere in sight.

Every other check in this repo is a shape, a sign, or an algebraic identity. This one asks the
only question that matters: *if you train a policy with these losses, does it converge to the
GFlowNet target* ``p(x) ∝ R(x)^β``?

The setup is small enough to answer exactly. A tabular autoregressive policy generates
length-4 sequences from a 6-token vocabulary — 1296 terminal states — under a reward with
first-order and pairwise terms, so the target is genuinely non-factorised and a policy that
ignored its prefix could not represent it. Both the target distribution and the policy's own
distribution are computed by full enumeration, so the total-variation distance and KL are
exact, not sampled.

What "correct" means here, and why the mode-coverage assertion is the important one
---------------------------------------------------------------------------------
A reward-maximising RL objective (PPO, REINFORCE, best-of-n distillation) converges to a point
mass on ``argmax R``. That is the *wrong answer* for this project: we want a policy that
proposes diverse correct programs, not one that memorises a single solution. A GFlowNet
objective converges to the reward *distribution*, which means every mode keeps mass in
proportion to its reward. So the assertions are two-sided: TV and KL must be small (the
distribution is right), **and** the policy's entropy must stay near the target's with every
top mode still covered (it did not collapse). An implementation with a sign error or a
misplaced ``log Z`` typically passes neither; one that has silently become reward-maximising
passes the first family of checks on the modes and fails the entropy check hard.

Two deliberate design choices:

* **Sampling is off-policy.** Trajectories are drawn from an ε-uniform mixture, never from the
  policy itself, and ``off_policy_correction`` stays at its default ``False``. If the
  objectives were only valid on-policy this test would not converge. That is the property the
  whole project leans on for its replay buffer.
* **DB and SubTB get a tabular flow.** :class:`~flowcode.objectives.flows.LogFlowEstimator`
  cannot see the prefix (see that module's docstring on why Tinker forbids a flow head on the
  LM), so on this toy it provably cannot represent the true ``log F``. Giving those two
  objectives an exact-capacity flow isolates *the loss* — which is what is under test — from
  the estimator approximation we are forced into in production. The production estimator is
  separately smoke-tested at the bottom of this file.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import pytest
import torch
from torch import nn

from flowcode.objectives.base import Objective
from flowcode.objectives.db import DetailedBalance
from flowcode.objectives.flows import FlowStates, LogFlowEstimator, ScalarLogZ
from flowcode.objectives.subtb import SubTB
from flowcode.objectives.tb import TrajectoryBalance
from flowcode.objectives.vargrad import VarGrad
from flowcode.types import Trajectory, token_level_segments

# Every test in this file trains to convergence, which is ~11s locally but ~456s on the CI
# runner — 96% of the whole suite's wall clock. Deselected from the default `mise run test`
# (see addopts in pyproject.toml) and run by the nightly slow-tests workflow instead, so a
# regression here surfaces within a day rather than on the next PR. Run locally with
# `mise run test:slow`.
pytestmark = pytest.mark.slow

VOCAB = 6
LENGTH = 4
NUM_TERMINALS = VOCAB**LENGTH  # 1296
TASK_ID = "toy"


# --------------------------------------------------------------------------------------
# The toy world
# --------------------------------------------------------------------------------------


def _prefix_offsets() -> list[int]:
    """Start index of each prefix-length block in the flattened state table."""
    offsets = [0]
    for _ in range(LENGTH):
        offsets.append(offsets[-1] + VOCAB ** (len(offsets) - 1))
    return offsets


PREFIX_OFFSETS = _prefix_offsets()
NUM_PREFIX_STATES = PREFIX_OFFSETS[-1]  # 1 + 6 + 36 + 216 = 259


def prefix_index(prefix: Sequence[int]) -> int:
    """Flat index of a prefix among all prefixes of length ``0..LENGTH-1``.

    Args:
        prefix: Tokens generated so far; length must be below :data:`LENGTH`.

    Returns:
        A row index into the policy's logit table.
    """
    index = 0
    for token in prefix:
        index = index * VOCAB + token
    return PREFIX_OFFSETS[len(prefix)] + index


@dataclass(frozen=True)
class ToyWorld:
    """All 1296 terminal sequences, their exact log rewards, and the exact target.

    Args:
        sequences: ``(1296, 4)`` long — every terminal state, in a fixed order.
        prefix_ids: ``(1296, 4)`` long — the policy-table row used at each generation step.
        log_reward: ``(1296,)`` float — ``log R(x)``.
        log_target: ``(1296,)`` float — ``log p*(x) = log R(x) - log Z``.
        log_partition: The exact ``log Z``, i.e. ``logsumexp(log_reward)``.
    """

    sequences: torch.Tensor
    prefix_ids: torch.Tensor
    log_reward: torch.Tensor
    log_target: torch.Tensor
    log_partition: float

    @property
    def target(self) -> torch.Tensor:
        """``(1296,)`` exact target probabilities."""
        return self.log_target.exp()


def build_world(seed: int = 0, scale: float = 1.0) -> ToyWorld:
    """Enumerate the toy state space and build a non-factorised reward over it.

    The reward is a chain MRF, ``log R(x) = Σ_t a[t, x_t] + Σ_t b[x_t, x_{t+1}]``. The pairwise
    term is what makes this a real test: the target is not a product of per-position
    marginals, so a policy that only got the position marginals right would still show a large
    TV distance.

    Args:
        seed: Seeds the reward tables.
        scale: Multiplies the reward tables. Larger means a peakier, harder target.

    Returns:
        A fully enumerated :class:`ToyWorld`.
    """
    generator = torch.Generator().manual_seed(seed)
    unary = torch.randn(LENGTH, VOCAB, generator=generator) * scale
    pairwise = torch.randn(VOCAB, VOCAB, generator=generator) * scale

    sequences = torch.zeros(NUM_TERMINALS, LENGTH, dtype=torch.long)
    for i in range(NUM_TERMINALS):
        remaining = i
        for t in range(LENGTH - 1, -1, -1):
            sequences[i, t] = remaining % VOCAB
            remaining //= VOCAB

    prefix_ids = torch.zeros(NUM_TERMINALS, LENGTH, dtype=torch.long)
    for t in range(LENGTH):
        for i in range(NUM_TERMINALS):
            prefix_ids[i, t] = prefix_index(sequences[i, :t].tolist())

    log_reward = unary[torch.arange(LENGTH), sequences].sum(dim=1)
    log_reward = log_reward + pairwise[sequences[:, :-1], sequences[:, 1:]].sum(dim=1)

    log_partition = float(torch.logsumexp(log_reward, dim=0))
    return ToyWorld(
        sequences=sequences,
        prefix_ids=prefix_ids,
        log_reward=log_reward,
        log_target=log_reward - log_partition,
        log_partition=log_partition,
    )


class ToyPolicy(nn.Module):
    """A tabular autoregressive policy: one logit row per reachable prefix.

    Exact by construction — 259 rows x 6 logits can represent *any* distribution over the 1296
    sequences — so a failure to converge is a failure of the objective, never of capacity.
    """

    def __init__(self) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(NUM_PREFIX_STATES, VOCAB))

    def log_probs(self, prefix_rows: torch.Tensor) -> torch.Tensor:
        """Per-token log-softmax for a batch of prefix rows, shape ``(B, VOCAB)``."""
        return torch.log_softmax(self.logits[prefix_rows], dim=-1)

    def distribution(self, world: ToyWorld) -> torch.Tensor:
        """The exact ``(1296,)`` distribution over terminal states, by enumeration."""
        all_log_probs = torch.log_softmax(self.logits, dim=-1)
        log_p = torch.zeros(NUM_TERMINALS)
        for t in range(LENGTH):
            log_p = log_p + all_log_probs[world.prefix_ids[:, t], world.sequences[:, t]]
        return log_p.exp()


class TabularLogFlow(nn.Module):
    """An exact-capacity ``log F(s)``: one free parameter per intermediate prefix.

    This is the flow the papers assume and that Tinker denies us in production. Used here so
    that DB and SubTB are tested on their *own* maths rather than on the expressiveness of
    :class:`~flowcode.objectives.flows.LogFlowEstimator`. It reconstructs the state from
    :meth:`FlowStates.prefix`, which is exactly the escape hatch that interface exists for.
    """

    def __init__(self) -> None:
        super().__init__()
        self.table = nn.Parameter(torch.zeros(NUM_PREFIX_STATES))

    def log_flow(self, states: FlowStates) -> torch.Tensor:
        if len(states) == 0:
            return self.table.new_zeros(0)
        traj_index = states.traj_index.tolist()
        end_index = states.end_index.tolist()
        rows = torch.tensor(
            [
                prefix_index(states.completions[t][:e])
                for t, e in zip(traj_index, end_index, strict=True)
            ],
            dtype=torch.long,
        )
        return self.table[rows]

    def register_tasks(self, task_ids: Sequence[str]) -> None:
        return None


# --------------------------------------------------------------------------------------
# Rollouts and training
# --------------------------------------------------------------------------------------


def sample_batch(
    policy: ToyPolicy,
    world: ToyWorld,
    batch_size: int,
    epsilon: float,
    generator: torch.Generator,
) -> tuple[list[torch.Tensor], list[Trajectory]]:
    """Roll out a batch from an ε-uniform mixture of the policy.

    Sampling from a *different* distribution than the one being trained is the point: it
    exercises the off-policy consistency the replay buffer depends on, and it supplies the
    exploration that mode coverage needs. The returned ``sampling_logprobs`` are the behaviour
    policy's; the returned tensors are the *current policy's*, with grad, exactly mirroring
    what the Tinker bridge hands the objectives in production.

    Args:
        policy: The policy being trained.
        world: The enumerated toy world, for reward lookup.
        batch_size: Number of trajectories.
        epsilon: Mixture weight on the uniform distribution.
        generator: Seeded RNG.

    Returns:
        ``(logprobs, batch)``.
    """
    rows = torch.zeros(batch_size, dtype=torch.long)
    tokens = torch.zeros(batch_size, LENGTH, dtype=torch.long)
    policy_log_probs = []
    behaviour_log_probs = []

    for _ in range(LENGTH):
        log_probs = policy.log_probs(rows)
        with torch.no_grad():
            behaviour = (1.0 - epsilon) * log_probs.exp() + epsilon / VOCAB
        actions = torch.multinomial(behaviour, num_samples=1, generator=generator).squeeze(-1)
        policy_log_probs.append(log_probs.gather(1, actions[:, None]).squeeze(-1))
        behaviour_log_probs.append(behaviour.gather(1, actions[:, None]).squeeze(-1).log().detach())
        tokens[:, len(policy_log_probs) - 1] = actions
        rows = (
            rows * VOCAB
            + actions
            + (
                PREFIX_OFFSETS[len(policy_log_probs)]
                - PREFIX_OFFSETS[len(policy_log_probs) - 1] * VOCAB
            )
        )

    stacked = torch.stack(policy_log_probs, dim=1)
    behaviour_stacked = torch.stack(behaviour_log_probs, dim=1)

    flat = torch.zeros(batch_size, dtype=torch.long)
    for t in range(LENGTH):
        flat = flat * VOCAB + tokens[:, t]
    rewards = world.log_reward[flat]

    segments = token_level_segments(LENGTH)
    batch = [
        Trajectory(
            task_id=TASK_ID,
            prompt_tokens=[0],
            completion_tokens=tokens[i].tolist(),
            sampling_logprobs=behaviour_stacked[i].tolist(),
            log_reward=float(rewards[i]),
            segments=list(segments),
            metadata={},
        )
        for i in range(batch_size)
    ]
    return [stacked[i] for i in range(batch_size)], batch


@dataclass(frozen=True)
class ConvergenceResult:
    """What one training run achieved against the exact target.

    Args:
        total_variation: ``0.5 * Σ |p - p*|`` over all 1296 terminal states.
        kl: ``KL(p* || p)``, in nats — the mode-covering direction, so it explodes if the
            policy drops a mode the target cares about.
        entropy_ratio: ``H(p) / H(p*)``. A reward-maximising policy drives this to ~0.
        worst_mode_ratio: Over the target's 10 highest-probability states, the smallest
            ``p(x) / p*(x)``. Directly measures mode coverage.
        final_loss: Mean loss over the last 50 steps.
    """

    total_variation: float
    kl: float
    entropy_ratio: float
    worst_mode_ratio: float
    final_loss: float


def evaluate(policy: ToyPolicy, world: ToyWorld, final_loss: float) -> ConvergenceResult:
    """Compare the policy's exact distribution against the exact target."""
    with torch.no_grad():
        model = policy.distribution(world)
        target = world.target
        total_variation = float(0.5 * (model - target).abs().sum())
        kl = float((target * (target.clamp_min(1e-30).log() - model.clamp_min(1e-30).log())).sum())
        model_entropy = float(-(model * model.clamp_min(1e-30).log()).sum())
        target_entropy = float(-(target * target.clamp_min(1e-30).log()).sum())
        top = torch.topk(target, k=10).indices
        worst = float((model[top] / target[top]).min())
    return ConvergenceResult(
        total_variation=total_variation,
        kl=kl,
        entropy_ratio=model_entropy / target_entropy,
        worst_mode_ratio=worst,
        final_loss=final_loss,
    )


def train(
    objective: Objective,
    world: ToyWorld,
    steps: int = 700,
    batch_size: int = 64,
    policy_lr: float = 0.05,
    flow_lr: float = 0.2,
    epsilon: float = 0.15,
    seed: int = 0,
) -> ConvergenceResult:
    """Train a fresh :class:`ToyPolicy` with ``objective`` and score it exactly.

    Two optimisers, matching production: one for the policy (which on Tinker would be the
    LoRA) and one at a much higher learning rate for the client-side flow parameters.

    Args:
        objective: The loss under test.
        world: The enumerated toy world.
        steps: Gradient steps.
        batch_size: Trajectories per step. Also the VarGrad group size, since every trajectory
            shares one ``task_id``.
        policy_lr: Adam LR for the policy table.
        flow_lr: Adam LR for ``log Z`` / ``log F``.
        epsilon: Exploration mixture weight for the behaviour policy.
        seed: Seeds the policy init and the rollouts.

    Returns:
        The measured :class:`ConvergenceResult`.
    """
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed + 1)
    policy = ToyPolicy()
    objective.register_tasks([TASK_ID])

    policy_optimiser = torch.optim.Adam(policy.parameters(), lr=policy_lr)
    flow_params = list(objective.parameters())
    flow_optimiser = torch.optim.Adam(flow_params, lr=flow_lr) if flow_params else None

    recent: list[float] = []
    for _ in range(steps):
        logprobs, batch = sample_batch(policy, world, batch_size, epsilon, generator)
        loss, _ = objective.loss(logprobs, batch)

        policy_optimiser.zero_grad(set_to_none=True)
        if flow_optimiser is not None:
            flow_optimiser.zero_grad(set_to_none=True)
        torch.autograd.backward(loss)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        policy_optimiser.step()
        if flow_optimiser is not None:
            flow_optimiser.step()

        recent.append(float(loss.detach()))
        recent = recent[-50:]

    return evaluate(policy, world, sum(recent) / len(recent))


# --------------------------------------------------------------------------------------
# The tests
# --------------------------------------------------------------------------------------


STEPS = 600
BATCH_SIZE = 64
OBJECTIVE_NAMES = ("vargrad", "tb", "subtb", "db")


@pytest.fixture(scope="module")
def world() -> ToyWorld:
    return build_world(seed=0, scale=1.0)


def _objective(name: str) -> Objective:
    """One objective, configured the way the shipped Hydra config does (bar the flow)."""
    if name == "vargrad":
        return VarGrad(reward_temperature=1.0)
    if name == "tb":
        return TrajectoryBalance(reward_temperature=1.0, log_z=ScalarLogZ(init_value=0.0))
    if name == "subtb":
        return SubTB(
            lambda_=0.9,
            granularity="token",
            max_pairs=4096,
            flow=TabularLogFlow(),
            log_z=ScalarLogZ(init_value=0.0),
        )
    if name == "db":
        return DetailedBalance(
            granularity="token",
            flow=TabularLogFlow(),
            log_z=ScalarLogZ(init_value=0.0),
            edge_reduction="mean",
        )
    raise AssertionError(f"unknown objective {name!r}")


# Thresholds are ~3x the measured values (see the report in the module docstring of this
# section below), which leaves headroom for torch version drift without letting a genuinely
# broken objective through: a sign error or a misplaced log Z lands at TV > 0.5, and an
# untrained policy sits at TV ~ 0.62 on this world.
MAX_TOTAL_VARIATION = 0.06
MAX_KL = 0.02


@pytest.fixture(scope="module")
def trained(world: ToyWorld) -> dict[str, tuple[Objective, ConvergenceResult]]:
    """Train every objective exactly once and share the outcome across the tests.

    Training is the expensive part of this file, so it happens here rather than per-test; the
    whole module then stays inside a CI-friendly runtime budget.
    """
    outcome: dict[str, tuple[Objective, ConvergenceResult]] = {}
    for name in OBJECTIVE_NAMES:
        objective = _objective(name)
        result = train(objective, world, steps=STEPS, batch_size=BATCH_SIZE)
        outcome[name] = (objective, result)
        print(
            f"\n  {name:8s} TV={result.total_variation:.4f} KL={result.kl:.4f} "
            f"H/H*={result.entropy_ratio:.3f} worst_mode={result.worst_mode_ratio:.3f} "
            f"loss={result.final_loss:.4f}"
        )
    return outcome


@pytest.mark.parametrize("name", OBJECTIVE_NAMES)
def test_objective_converges_to_the_reward_distribution(
    name: str, trained: dict[str, tuple[Objective, ConvergenceResult]]
) -> None:
    """``p(x)`` must converge to ``R(x)/Z`` — measured exactly over all 1296 terminal states."""
    result = trained[name][1]
    assert result.total_variation < MAX_TOTAL_VARIATION, f"{name}: TV={result.total_variation}"
    assert result.kl < MAX_KL, f"{name}: KL={result.kl}"


@pytest.mark.parametrize("name", OBJECTIVE_NAMES)
def test_objective_covers_modes_instead_of_collapsing(
    name: str, trained: dict[str, tuple[Objective, ConvergenceResult]]
) -> None:
    """The property that separates a GFlowNet from reward-maximising RL.

    An argmax-seeking objective would put ~all mass on one sequence: entropy ratio near 0 and
    every other mode starved. A correct GFlowNet keeps each mode in proportion to its reward,
    so the entropy must land near the target's and no top mode may be dropped *or* inflated.
    """
    result = trained[name][1]
    assert 0.95 < result.entropy_ratio < 1.05, f"{name}: H/H* = {result.entropy_ratio}"
    assert result.worst_mode_ratio > 0.6, f"{name}: worst top-10 mode ratio"
    assert result.worst_mode_ratio < 1.6, f"{name}: worst top-10 mode ratio"


def test_a_reward_maximiser_would_fail_the_mode_coverage_assertion(world: ToyWorld) -> None:
    """Calibration: show the mode-coverage check is not vacuous.

    A policy collapsed onto ``argmax R`` — the fixed point of reward-maximising RL — is scored
    with the same function and must fail exactly the assertions the GFlowNet objectives pass.
    """
    policy = ToyPolicy()
    best = int(world.log_reward.argmax())
    with torch.no_grad():
        for t in range(LENGTH):
            row = int(world.prefix_ids[best, t])
            policy.logits[row, int(world.sequences[best, t])] = 30.0
    result = evaluate(policy, world, final_loss=0.0)
    assert result.entropy_ratio < 0.1
    assert result.total_variation > 0.5
    assert result.worst_mode_ratio < 0.6


def test_an_untrained_policy_is_far_from_the_target(world: ToyWorld) -> None:
    """The other calibration end: uniform is not accidentally close to the answer."""
    result = evaluate(ToyPolicy(), world, final_loss=0.0)
    assert result.total_variation > 0.4
    assert result.kl > 0.5


def test_tb_learns_the_true_log_partition_function(
    world: ToyWorld, trained: dict[str, tuple[Objective, ConvergenceResult]]
) -> None:
    """``log Z`` is not a free-floating nuisance parameter — it must land on the real value."""
    objective = trained["tb"][0]
    assert isinstance(objective, TrajectoryBalance)
    assert isinstance(objective.log_z, ScalarLogZ)
    learned = float(objective.log_z.value.detach())
    assert learned == pytest.approx(world.log_partition, abs=0.3), (
        f"learned log Z {learned:.3f} vs exact {world.log_partition:.3f}"
    )


def test_vargrad_recovers_the_same_log_partition_without_learning_one(world: ToyWorld) -> None:
    """VarGrad has no ``log Z`` parameter, yet its in-batch estimate must converge to log Z."""
    objective = VarGrad()
    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(1)
    policy = ToyPolicy()
    optimiser = torch.optim.Adam(policy.parameters(), lr=0.05)
    steps, window = 500, 50
    estimate = 0.0
    for step in range(steps):
        logprobs, batch = sample_batch(policy, world, BATCH_SIZE, 0.15, generator)
        loss, metrics = objective.loss(logprobs, batch)
        optimiser.zero_grad(set_to_none=True)
        torch.autograd.backward(loss)
        optimiser.step()
        if step >= steps - window:
            estimate += metrics["log_z_estimate_mean"] / window
    assert estimate == pytest.approx(world.log_partition, abs=0.4)


def test_reward_temperature_sharpens_the_learned_distribution(
    world: ToyWorld, trained: dict[str, tuple[Objective, ConvergenceResult]]
) -> None:
    """β > 1 must move the policy toward the argmax — the knob has to actually do something."""
    warm = trained["vargrad"][1]
    cold = train(VarGrad(reward_temperature=2.0), world, steps=400, batch_size=BATCH_SIZE)
    # Scored against the β=1 target, a β=2 policy is deliberately *wrong*, and sharper.
    assert cold.entropy_ratio < 0.85
    assert cold.total_variation > warm.total_variation


def test_production_flow_estimator_trains_without_diverging(world: ToyWorld) -> None:
    """Smoke test for the flow estimator we are actually stuck with on Tinker.

    :class:`~flowcode.objectives.flows.LogFlowEstimator` cannot see the prefix, so it provably
    cannot reach the exact target on this toy and no TV threshold is asserted. What must hold
    is that it stays finite, drives its own loss down, and still moves the policy toward the
    target rather than blowing up — the realistic claim for DB/SubTB in production, and the
    reason ``objective=vargrad`` is the default.
    """
    objective = SubTB(
        lambda_=0.9,
        granularity="token",
        max_pairs=4096,
        flow=LogFlowEstimator(hidden_dim=64, task_embedding_dim=32),
    )
    untrained = evaluate(ToyPolicy(), world, final_loss=0.0)
    result = train(objective, world, steps=300, batch_size=32, flow_lr=0.05)
    assert math.isfinite(result.final_loss)
    assert result.total_variation < untrained.total_variation
    for param in objective.parameters():
        assert torch.isfinite(param).all()
