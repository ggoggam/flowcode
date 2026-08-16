"""The prompt/completion alignment contract, independent of any backend.

Every backend has to answer the same question — *which output position carries the logprob
of which completion token* — and every backend gets it wrong the same way if it guesses.
So the arithmetic lives here once, as free functions over
:class:`~flowcode.types.Trajectory`, and both the hosted and the local backends build on
it rather than each rederiving it.

The contract
------------
For a trajectory with prompt ``p`` (length ``P``) and completion ``c`` (length ``N``)::

    ob_len        = P - 1
    input_tokens  = p + c[:-1]                    # length ob_len + N
    target_tokens = [0] * ob_len + c              # length ob_len + N
    weights       = [0.0] * ob_len + <N values>   # length ob_len + N

A causal LM predicts position ``i + 1`` from position ``i``, so the logprob of ``c[0]``
shows up at index ``P - 1 = ob_len`` of the output and the logprob of ``c[N-1]`` at index
``ob_len + N - 1``. Positions ``[0, ob_len)`` score prompt tokens we do not train on; the
zero weights there make them contribute nothing, and the padded ``0`` target tokens are
never read for gradient purposes.

The last completion token is deliberately absent from ``input_tokens``: it is a *target*
only. Including it would ask the model to predict a token past the end of the trajectory.

:func:`slice_completion_values` strips that padding off, so objectives only ever see
arrays of length ``N`` indexed the same way ``Trajectory.completion_tokens`` is — which is
why nothing under ``objectives/`` has to know a prompt exists.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from flowcode.types import Trajectory

__all__ = [
    "PAD_TARGET_TOKEN",
    "build_input_tokens",
    "build_target_tokens",
    "observation_length",
    "pad_completion_values",
    "slice_completion_values",
]

PAD_TARGET_TOKEN = 0
"""Filler for the prompt-side target positions. Never contributes: its weight is 0.0."""


def observation_length(trajectory: Trajectory) -> int:
    """``ob_len``: how many leading output positions score prompt tokens, not completion.

    Args:
        trajectory: The trajectory to measure.

    Returns:
        ``len(prompt_tokens) - 1``. The model input is ``prompt + completion[:-1]``, and
        position ``i`` predicts token ``i + 1``, so the first completion token's logprob
        lands at index ``len(prompt_tokens) - 1``.
    """
    return trajectory.num_prompt_tokens - 1


def build_input_tokens(trajectory: Trajectory) -> list[int]:
    """Build ``prompt + completion[:-1]`` as plain token ids.

    Args:
        trajectory: The trajectory to encode.

    Returns:
        Token ids of length ``ob_len + len(completion_tokens)``, ready to be wrapped in
        whatever input type a backend wants.
    """
    return [*trajectory.prompt_tokens, *trajectory.completion_tokens[:-1]]


def build_target_tokens(trajectory: Trajectory) -> list[int]:
    """Build ``[0] * ob_len + completion_tokens``.

    Args:
        trajectory: The trajectory to encode.

    Returns:
        Target token ids aligned to :func:`build_input_tokens`'s output positions. The
        leading zeros are filler for prompt-side positions and are inert because the
        matching weights are zero.
    """
    ob_len = observation_length(trajectory)
    return [PAD_TARGET_TOKEN] * ob_len + list(trajectory.completion_tokens)


def pad_completion_values(trajectory: Trajectory, values: Sequence[float]) -> list[float]:
    """Left-pad a completion-shaped float array out to full model-input length.

    This is the inverse of :func:`slice_completion_values` and the reason objectives never
    have to think about the prompt at all.

    Args:
        trajectory: The trajectory the values belong to.
        values: One float per completion token.

    Returns:
        ``[0.0] * ob_len + list(values)``.

    Raises:
        ValueError: If ``values`` is not exactly ``len(completion_tokens)`` long.
    """
    n = trajectory.num_completion_tokens
    if len(values) != n:
        raise ValueError(
            f"Trajectory {trajectory.task_id!r}: expected {n} per-completion-token values "
            f"(one per completion token) but got {len(values)}. Objectives work in "
            "completion-token space; do not pre-pad."
        )
    return [0.0] * observation_length(trajectory) + [float(v) for v in values]


def slice_completion_values(trajectory: Trajectory, full: torch.Tensor) -> torch.Tensor:
    """Drop the prompt-side padding from a full-length per-position array.

    Args:
        trajectory: The trajectory the array belongs to.
        full: A 1-D tensor of length ``ob_len + len(completion_tokens)``.

    Returns:
        A view of length ``len(completion_tokens)``, indexed like
        ``Trajectory.completion_tokens``.

    Raises:
        ValueError: If ``full`` has the wrong rank or length, which in practice means the
            alignment contract broke somewhere upstream.
    """
    ob_len = observation_length(trajectory)
    expected = ob_len + trajectory.num_completion_tokens
    if full.ndim != 1 or full.shape[0] != expected:
        raise ValueError(
            f"Trajectory {trajectory.task_id!r}: expected a 1-D array of length "
            f"{expected} (= ob_len {ob_len} + {trajectory.num_completion_tokens} "
            f"completion tokens) but got shape {tuple(full.shape)}. The backend returned a "
            "different number of positions than the model input had; the alignment "
            "contract is broken."
        )
    return full[ob_len:]
