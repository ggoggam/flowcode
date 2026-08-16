"""The replay buffer: bounded, deterministic, and soft about prioritization.

The two properties worth defending here are unglamorous. **Bounded**, because a 1000-step
run adds tens of thousands of trajectories and the buffer is the only thing holding them.
**Deterministic**, because replay is one of exactly two places a seeded run could stop
being reproducible.

The third is the interesting one: reward prioritization must stay *soft*. A buffer that
keeps drifting toward the best-scoring trajectories reintroduces the mode collapse the
GFlowNet objective exists to avoid, so the weight of the worst entry is asserted to be a
real fraction of the best rather than an epsilon.
"""

from __future__ import annotations

import math
import random
from collections import Counter

import pytest

from flowcode.replay import PRIORITIZE_KINDS, RECENCY_DECAY, REWARD_SHARPNESS, ReplayBuffer
from flowcode.types import Segment, Trajectory


def make_trajectory(task_id: str = "t0", log_reward: float = -1.0, length: int = 3) -> Trajectory:
    return Trajectory(
        task_id=task_id,
        prompt_tokens=[1, 2, 3],
        completion_tokens=list(range(10, 10 + length)),
        sampling_logprobs=[-0.5] * length,
        log_reward=log_reward,
        segments=[Segment(0, length)],
    )


def make_batch(rewards: list[float], task_prefix: str = "t") -> list[Trajectory]:
    return [
        make_trajectory(task_id=f"{task_prefix}{i}", log_reward=r) for i, r in enumerate(rewards)
    ]


class TestConstruction:
    @pytest.mark.parametrize("capacity", [0, -1])
    def test_capacity_must_be_positive(self, capacity: int) -> None:
        with pytest.raises(ValueError, match="capacity"):
            ReplayBuffer(capacity=capacity)

    def test_unknown_prioritize_names_the_legal_values(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            ReplayBuffer(capacity=4, prioritize="best")
        for kind in PRIORITIZE_KINDS:
            assert kind in str(excinfo.value)

    @pytest.mark.parametrize("kind", PRIORITIZE_KINDS)
    def test_every_configured_scheme_constructs(self, kind: str) -> None:
        assert len(ReplayBuffer(capacity=4, prioritize=kind)) == 0


class TestCapacityAndEviction:
    def test_len_tracks_contents(self) -> None:
        buffer = ReplayBuffer(capacity=10)
        buffer.add(make_batch([-1.0, -2.0]))
        assert len(buffer) == 2

    def test_capacity_is_never_exceeded(self) -> None:
        buffer = ReplayBuffer(capacity=3)
        for _ in range(20):
            buffer.add(make_batch([-1.0, -2.0]))
        assert len(buffer) == 3

    def test_eviction_is_fifo(self) -> None:
        # Oldest out first, regardless of reward: evicting the worst instead would turn the
        # buffer into a hall of fame and collapse the diversity replay is meant to preserve.
        buffer = ReplayBuffer(capacity=3, prioritize="reward")
        buffer.add([make_trajectory(task_id=f"t{i}", log_reward=-float(i)) for i in range(5)])
        assert buffer.task_ids() == ["t2", "t3", "t4"]

    def test_eviction_is_counted(self) -> None:
        buffer = ReplayBuffer(capacity=2)
        buffer.add(make_batch([-1.0, -2.0, -3.0, -4.0]))
        assert buffer.num_added == 4
        assert buffer.num_evicted == 2

    def test_clear_empties_but_keeps_counters(self) -> None:
        buffer = ReplayBuffer(capacity=4)
        buffer.add(make_batch([-1.0, -2.0]))
        buffer.clear()
        assert len(buffer) == 0
        assert buffer.num_added == 2


class TestSamplingContract:
    def test_empty_buffer_returns_nothing(self) -> None:
        assert ReplayBuffer(capacity=4).sample(3, random.Random(0)) == []

    @pytest.mark.parametrize("n", [0, -5])
    def test_non_positive_n_returns_nothing(self, n: int) -> None:
        buffer = ReplayBuffer(capacity=4)
        buffer.add(make_batch([-1.0, -2.0]))
        assert buffer.sample(n, random.Random(0)) == []

    def test_asking_for_more_than_exists_returns_everything(self) -> None:
        # Normal early in a run, and not an error.
        buffer = ReplayBuffer(capacity=10)
        buffer.add(make_batch([-1.0, -2.0]))
        assert len(buffer.sample(50, random.Random(0))) == 2

    def test_sampling_is_without_replacement(self) -> None:
        buffer = ReplayBuffer(capacity=10)
        buffer.add(make_batch([-1.0] * 6))
        drawn = buffer.sample(4, random.Random(1))
        assert len({id(t) for t in drawn}) == 4

    def test_sampling_does_not_mutate_the_buffer(self) -> None:
        buffer = ReplayBuffer(capacity=10)
        buffer.add(make_batch([-1.0] * 5))
        buffer.sample(3, random.Random(0))
        assert len(buffer) == 5

    def test_replayed_trajectories_carry_what_recomputation_needs(self) -> None:
        # The training loop must re-score replayed trajectories with the CURRENT policy via
        # compute_logprobs, which needs the prompt and completion tokens intact.
        buffer = ReplayBuffer(capacity=4)
        original = make_trajectory(task_id="keep-me", log_reward=-0.5)
        buffer.add([original])
        (drawn,) = buffer.sample(1, random.Random(0))
        assert drawn.prompt_tokens == original.prompt_tokens
        assert drawn.completion_tokens == original.completion_tokens
        assert drawn.sampling_logprobs == original.sampling_logprobs
        assert drawn.task_id == "keep-me"


class TestDeterminism:
    @pytest.mark.parametrize("kind", PRIORITIZE_KINDS)
    def test_same_seed_same_draw(self, kind: str) -> None:
        buffer = ReplayBuffer(capacity=50, prioritize=kind)
        buffer.add(make_batch([-float(i) for i in range(20)]))
        first = buffer.sample(6, random.Random(7))
        second = buffer.sample(6, random.Random(7))
        assert [t.task_id for t in first] == [t.task_id for t in second]

    def test_different_seeds_differ(self) -> None:
        buffer = ReplayBuffer(capacity=50)
        buffer.add(make_batch([-1.0] * 30))
        a = [t.task_id for t in buffer.sample(6, random.Random(1))]
        b = [t.task_id for t in buffer.sample(6, random.Random(2))]
        assert a != b

    def test_the_global_rng_is_never_touched(self) -> None:
        buffer = ReplayBuffer(capacity=10)
        buffer.add(make_batch([-1.0] * 8))
        random.seed(1234)
        state = random.getstate()
        buffer.sample(4, random.Random(0))
        assert random.getstate() == state


class TestPrioritization:
    def test_uniform_weights_are_flat(self) -> None:
        buffer = ReplayBuffer(capacity=10, prioritize="uniform")
        buffer.add(make_batch([-1.0, -8.0, -0.1]))
        assert buffer.weights() == [1.0, 1.0, 1.0]

    def test_reward_weights_are_monotone_in_log_reward(self) -> None:
        buffer = ReplayBuffer(capacity=10, prioritize="reward")
        buffer.add(make_batch([-5.0, -1.0, -3.0]))
        weights = buffer.weights()
        assert weights[1] > weights[2] > weights[0]

    def test_reward_weights_span_exactly_the_documented_range(self) -> None:
        # Rank-based, so the spread is a constant of the scheme rather than a function of
        # how far apart the rewards happen to be on this particular step.
        buffer = ReplayBuffer(capacity=10, prioritize="reward")
        buffer.add(make_batch([-1.0, -2.0, -3.0, -4.0]))
        weights = buffer.weights()
        assert max(weights) == pytest.approx(1.0)
        assert min(weights) == pytest.approx(math.exp(-REWARD_SHARPNESS))

    def test_reward_weighting_is_invariant_to_the_reward_scale(self) -> None:
        # The whole reason for ranks: log R is compressed early (everything on the floor)
        # and again late (everything passing), and a value-softmax would silently change
        # its selection pressure between those regimes.
        compressed = ReplayBuffer(capacity=10, prioritize="reward")
        compressed.add(make_batch([-0.01, -0.02, -0.03]))
        spread = ReplayBuffer(capacity=10, prioritize="reward")
        spread.add(make_batch([-1.0, -10.0, -20.0]))
        assert compressed.weights() == pytest.approx(spread.weights())

    def test_reward_prioritization_is_not_top_k(self) -> None:
        # Every trajectory keeps real mass. A hard top-k here is exactly the distribution
        # collapse GFlowNets are supposed to avoid.
        buffer = ReplayBuffer(capacity=100, prioritize="reward")
        buffer.add(make_batch([-float(i) for i in range(20)]))
        counts: Counter[str] = Counter()
        rng = random.Random(0)
        for _ in range(400):
            counts.update(t.task_id for t in buffer.sample(4, rng))
        assert len(counts) == 20, "some trajectory was never drawn"
        best, worst = counts["t0"], counts["t19"]
        assert best > worst, "high reward should be preferred"
        # Inclusion probability tracks the weight, so the worst entry lands near
        # exp(-REWARD_SHARPNESS) of the best. Half of that is a generous floor and still
        # nowhere near the zero a top-k scheme would produce.
        assert worst > 0.5 * math.exp(-REWARD_SHARPNESS) * best, "low reward must keep real mass"

    def test_reward_ties_are_broken_deterministically(self) -> None:
        a = ReplayBuffer(capacity=10, prioritize="reward")
        b = ReplayBuffer(capacity=10, prioritize="reward")
        a.add(make_batch([-1.0, -1.0, -1.0]))
        b.add(make_batch([-1.0, -1.0, -1.0]))
        assert a.weights() == b.weights()

    def test_recency_weights_favour_the_newest(self) -> None:
        buffer = ReplayBuffer(capacity=10, prioritize="recency")
        buffer.add(make_batch([-1.0] * 5))
        weights = buffer.weights()
        assert weights == sorted(weights), "weights must increase with recency"
        assert weights[-1] == pytest.approx(1.0)
        assert weights[0] == pytest.approx(math.exp(-RECENCY_DECAY * 4 / 5))

    def test_recency_prefers_recent_trajectories_in_practice(self) -> None:
        buffer = ReplayBuffer(capacity=100, prioritize="recency")
        buffer.add(make_batch([-1.0] * 20))
        counts: Counter[str] = Counter()
        rng = random.Random(3)
        for _ in range(400):
            counts.update(t.task_id for t in buffer.sample(4, rng))
        assert counts["t19"] > counts["t0"]

    def test_weights_of_an_empty_buffer(self) -> None:
        assert ReplayBuffer(capacity=4, prioritize="reward").weights() == []

    def test_single_entry_reward_weight_is_defined(self) -> None:
        buffer = ReplayBuffer(capacity=4, prioritize="reward")
        buffer.add([make_trajectory()])
        assert buffer.weights() == [pytest.approx(1.0)]


class TestStats:
    def test_stats_describe_the_buffer(self) -> None:
        buffer = ReplayBuffer(capacity=3, prioritize="reward")
        buffer.add(make_batch([-1.0, -3.0, -5.0, -7.0]))
        stats = buffer.stats()
        assert stats["size"] == 3.0
        assert stats["capacity"] == 3.0
        assert stats["num_added"] == 4.0
        assert stats["num_evicted"] == 1.0
        assert stats["distinct_tasks"] == 3.0
        assert stats["log_reward_mean"] == pytest.approx((-3.0 + -5.0 + -7.0) / 3)

    def test_stats_of_an_empty_buffer_are_zeros(self) -> None:
        stats = ReplayBuffer(capacity=5).stats()
        assert stats["size"] == 0.0
        assert stats["log_reward_mean"] == 0.0

    def test_repr_mentions_size_and_scheme(self) -> None:
        buffer = ReplayBuffer(capacity=5, prioritize="recency")
        assert "recency" in repr(buffer)
        assert "capacity=5" in repr(buffer)
