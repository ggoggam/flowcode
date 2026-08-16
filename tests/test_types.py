"""Trajectory / Segment validation and the segment helpers.

The point of the validation in :mod:`flowcode.types` is that a malformed trajectory
becomes a loud ValueError at construction instead of a silent misalignment thirty
milliseconds later inside a paid ``forward_backward``. These tests pin the loudness.
"""

from __future__ import annotations

import pytest

from flowcode.types import (
    Segment,
    TokenUsage,
    Trajectory,
    segments_from_boundaries,
    token_level_segments,
)


def make_trajectory(
    *,
    completion: list[int] | None = None,
    segments: list[Segment] | None = None,
    logprobs: list[float] | None = None,
    prompt: list[int] | None = None,
) -> Trajectory:
    """A valid trajectory, with any one part swapped out to test a failure mode."""
    completion = [10, 11, 12] if completion is None else completion
    return Trajectory(
        task_id="t0",
        prompt_tokens=[1, 2, 3, 4] if prompt is None else prompt,
        completion_tokens=completion,
        sampling_logprobs=[-0.1] * len(completion) if logprobs is None else logprobs,
        log_reward=-1.5,
        segments=[Segment(0, len(completion))] if segments is None else segments,
    )


class TestSegment:
    def test_rejects_empty_span(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            Segment(start=2, end=2)

    def test_rejects_reversed_span(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            Segment(start=5, end=3)

    def test_rejects_negative_start(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            Segment(start=-1, end=3)

    def test_len_is_the_span(self) -> None:
        assert len(Segment(start=2, end=7)) == 5

    def test_partial_log_reward_defaults_to_none(self) -> None:
        assert Segment(0, 1).partial_log_reward is None

    def test_is_frozen(self) -> None:
        seg = Segment(0, 1)
        with pytest.raises(Exception):  # noqa: B017 - dataclasses raise FrozenInstanceError
            seg.start = 3  # ty: ignore[invalid-assignment]


class TestTrajectoryValidation:
    def test_valid_trajectory_round_trips(self) -> None:
        traj = make_trajectory()
        assert traj.num_completion_tokens == 3
        assert traj.num_prompt_tokens == 4
        assert traj.total_tokens == 7

    def test_logprob_length_must_match_completion(self) -> None:
        with pytest.raises(ValueError, match="one-to-one"):
            make_trajectory(completion=[10, 11, 12], logprobs=[-0.1, -0.2])

    def test_empty_completion_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty completion"):
            make_trajectory(completion=[], segments=[Segment(0, 1)])

    def test_empty_prompt_rejected(self) -> None:
        # ob_len = len(prompt) - 1, so a zero-length prompt cannot build a model input.
        with pytest.raises(ValueError, match="empty prompt"):
            make_trajectory(prompt=[])

    def test_no_segments_rejected(self) -> None:
        with pytest.raises(ValueError, match="no segments"):
            make_trajectory(segments=[])

    def test_gap_between_segments_rejected(self) -> None:
        with pytest.raises(ValueError, match="contiguous"):
            make_trajectory(segments=[Segment(0, 1), Segment(2, 3)])

    def test_overlapping_segments_rejected(self) -> None:
        with pytest.raises(ValueError, match="contiguous"):
            make_trajectory(segments=[Segment(0, 2), Segment(1, 3)])

    def test_out_of_order_segments_rejected(self) -> None:
        with pytest.raises(ValueError, match="contiguous"):
            make_trajectory(segments=[Segment(1, 3), Segment(0, 1)])

    def test_under_covering_segments_rejected(self) -> None:
        with pytest.raises(ValueError, match="tile the completion"):
            make_trajectory(segments=[Segment(0, 2)])

    def test_over_covering_segments_rejected(self) -> None:
        with pytest.raises(ValueError, match="tile the completion"):
            make_trajectory(segments=[Segment(0, 3), Segment(3, 5)])

    def test_token_level_segments_are_accepted(self) -> None:
        completion = [7, 8, 9, 10]
        traj = make_trajectory(completion=completion, segments=token_level_segments(4))
        assert len(traj.segments) == 4

    def test_metadata_defaults_to_empty_and_is_per_instance(self) -> None:
        a = make_trajectory()
        b = make_trajectory()
        a.metadata["x"] = 1
        assert b.metadata == {}


class TestTokenLevelSegments:
    def test_tiles_exactly(self) -> None:
        segs = token_level_segments(4)
        assert [(s.start, s.end) for s in segs] == [(0, 1), (1, 2), (2, 3), (3, 4)]

    @pytest.mark.parametrize("n", [0, -1])
    def test_rejects_non_positive(self, n: int) -> None:
        with pytest.raises(ValueError, match="positive length"):
            token_level_segments(n)


class TestSegmentsFromBoundaries:
    def test_empty_boundaries_gives_single_segment(self) -> None:
        # This is the plain-TB layout: one segment, the terminal state only.
        assert [(s.start, s.end) for s in segments_from_boundaries([], 5)] == [(0, 5)]

    def test_interior_cuts(self) -> None:
        segs = segments_from_boundaries([2, 4], 6)
        assert [(s.start, s.end) for s in segs] == [(0, 2), (2, 4), (4, 6)]

    def test_unsorted_duplicated_and_terminal_boundaries_are_tolerated(self) -> None:
        segs = segments_from_boundaries([4, 2, 2, 6, 0], 6)
        assert [(s.start, s.end) for s in segs] == [(0, 2), (2, 4), (4, 6)]

    def test_result_is_a_valid_trajectory_layout(self) -> None:
        completion = list(range(6))
        traj = make_trajectory(completion=completion, segments=segments_from_boundaries([3], 6))
        assert [(s.start, s.end) for s in traj.segments] == [(0, 3), (3, 6)]

    def test_rejects_out_of_range_boundary(self) -> None:
        with pytest.raises(ValueError, match="outside"):
            segments_from_boundaries([7], 6)

    def test_rejects_negative_boundary(self) -> None:
        with pytest.raises(ValueError, match="outside"):
            segments_from_boundaries([-1], 6)

    @pytest.mark.parametrize("n", [0, -3])
    def test_rejects_non_positive_length(self, n: int) -> None:
        with pytest.raises(ValueError, match="positive length"):
            segments_from_boundaries([], n)


class TestTokenUsage:
    def test_starts_at_zero(self) -> None:
        assert TokenUsage() == TokenUsage(0, 0, 0, 0, 0, 0)

    def test_adds_fieldwise(self) -> None:
        total = TokenUsage(train_tokens=10, sample_tokens=5, num_forward_passes=1) + TokenUsage(
            train_tokens=3, prompt_cache_hit_tokens=2, num_backward_passes=1
        )
        assert total.train_tokens == 13
        assert total.sample_tokens == 5
        assert total.prompt_cache_hit_tokens == 2
        assert total.num_forward_passes == 1
        assert total.num_backward_passes == 1

    def test_billable_sample_tokens_nets_off_cache_hits(self) -> None:
        assert (
            TokenUsage(sample_tokens=100, prompt_cache_hit_tokens=30).billable_sample_tokens == 70
        )

    def test_billable_sample_tokens_floors_at_zero(self) -> None:
        # Tinker's cache-hit accounting is not multiplied across samples, so it can
        # exceed our gross count in odd cases; never report a negative bill.
        assert TokenUsage(sample_tokens=10, prompt_cache_hit_tokens=40).billable_sample_tokens == 0
