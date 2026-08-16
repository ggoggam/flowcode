"""The prompt/completion alignment contract, pinned independently of any backend.

Every off-by-one here is silent if it is wrong: the loss still computes, the gradient
still flows, and the model is trained to predict tokens one position out of step. So the
arithmetic is asserted directly rather than inferred from a passing training run.
"""

from __future__ import annotations

import pytest
import torch

from flowcode.alignment import (
    build_input_tokens,
    build_target_tokens,
    observation_length,
    pad_completion_values,
    slice_completion_values,
)
from flowcode.types import Segment, Trajectory

PROMPT = [101, 102, 103, 104]  # P = 4, so ob_len = 3
COMPLETION = [201, 202, 203, 204, 205]  # N = 5


def make_trajectory(
    prompt: list[int] | None = None,
    completion: list[int] | None = None,
) -> Trajectory:
    prompt = PROMPT if prompt is None else prompt
    completion = COMPLETION if completion is None else completion
    return Trajectory(
        task_id="t0",
        prompt_tokens=prompt,
        completion_tokens=completion,
        sampling_logprobs=[-0.1] * len(completion),
        log_reward=-2.0,
        segments=[Segment(0, len(completion))],
    )


class TestObservationLength:
    def test_is_prompt_minus_one(self) -> None:
        assert observation_length(make_trajectory()) == 3

    def test_single_token_prompt_has_no_padding(self) -> None:
        assert observation_length(make_trajectory(prompt=[7], completion=[42])) == 0


class TestInputTokens:
    def test_is_prompt_plus_completion_minus_last(self) -> None:
        assert build_input_tokens(make_trajectory()) == PROMPT + COMPLETION[:-1]

    def test_length_equals_ob_len_plus_completion(self) -> None:
        traj = make_trajectory()
        assert len(build_input_tokens(traj)) == observation_length(traj) + len(COMPLETION)

    def test_single_token_completion_is_exactly_the_prompt(self) -> None:
        # completion[:-1] is empty: the last completion token is a target only, and
        # including it would ask the model to predict past the end of the trajectory.
        assert build_input_tokens(make_trajectory(prompt=[7, 8, 9], completion=[42])) == [7, 8, 9]


class TestTargetTokens:
    def test_is_zero_padded_completion(self) -> None:
        assert build_target_tokens(make_trajectory()) == [0, 0, 0, *COMPLETION]

    def test_first_completion_token_lands_at_ob_len(self) -> None:
        # The whole point of the padding: index ob_len of the output scores completion[0].
        traj = make_trajectory()
        targets = build_target_tokens(traj)
        assert targets[observation_length(traj)] == COMPLETION[0]
        assert targets[-1] == COMPLETION[-1]

    def test_agrees_in_length_with_the_input(self) -> None:
        traj = make_trajectory()
        assert len(build_target_tokens(traj)) == len(build_input_tokens(traj))

    def test_two_token_prompt(self) -> None:
        traj = make_trajectory(prompt=[7, 8], completion=[42, 43])
        assert build_input_tokens(traj) == [7, 8, 42]
        assert build_target_tokens(traj) == [0, 42, 43]


class TestPadCompletionValues:
    def test_left_pads_with_zeros(self) -> None:
        assert pad_completion_values(make_trajectory(), [1.0, 2.0, 3.0, 4.0, 5.0]) == [
            0.0,
            0.0,
            0.0,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
        ]

    def test_prompt_positions_are_always_inert(self) -> None:
        traj = make_trajectory()
        ob_len = observation_length(traj)
        assert pad_completion_values(traj, [9.0] * 5)[:ob_len] == [0.0] * ob_len

    def test_rejects_wrong_length(self) -> None:
        with pytest.raises(ValueError, match="do not pre-pad"):
            pad_completion_values(make_trajectory(), [1.0, 2.0])


class TestSliceCompletionValues:
    def test_round_trips_padding(self) -> None:
        traj = make_trajectory()
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        padded = torch.tensor(pad_completion_values(traj, values))
        assert slice_completion_values(traj, padded).tolist() == values

    def test_rejects_wrong_length(self) -> None:
        with pytest.raises(ValueError, match="alignment contract is broken"):
            slice_completion_values(make_trajectory(), torch.zeros(3))

    def test_rejects_wrong_rank(self) -> None:
        with pytest.raises(ValueError, match="alignment contract is broken"):
            slice_completion_values(make_trajectory(), torch.zeros(2, 8))
