"""Shared vocabulary for flowcode.

Every module in the package speaks in terms of the two dataclasses here. A
:class:`Trajectory` is one sampled completion together with everything an objective
needs to score it; a :class:`Segment` marks where the GFlowNet state boundaries fall
inside that completion.

The GFlowNet framing: sampling a completion token-by-token is a trajectory through a
DAG whose states are token prefixes. Trajectory balance and its relatives need to know
which prefixes count as "states" for the purposes of the balance constraint. For plain
TB that is only the terminal state, so the whole completion is one segment. For SubTB
and DB the intermediate boundaries matter, and each boundary may carry a partial log
reward if the environment can score a prefix.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "SampleResponse",
    "SampledSequence",
    "SamplingParams",
    "Segment",
    "TokenUsage",
    "Trajectory",
    "TrajectoryBatch",
    "segments_from_boundaries",
    "token_level_segments",
]


@dataclass(frozen=True)
class Segment:
    """A contiguous run of completion tokens ending at a GFlowNet state boundary.

    Indices are into ``Trajectory.completion_tokens``, half-open in the usual Python
    way: ``[start, end)``.

    Args:
        start: First completion-token index in the segment, inclusive.
        end: One past the last completion-token index, exclusive.
        partial_log_reward: ``log R(s)`` for the state reached at ``end``, if the
            environment can score this intermediate state. ``None`` means unscorable,
            which is the common case for everything but the terminal segment.
    """

    start: int
    end: int
    partial_log_reward: float | None = None

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"Segment.start must be non-negative, got {self.start}")
        if self.end <= self.start:
            raise ValueError(
                f"Segment must be non-empty: got start={self.start}, end={self.end} "
                "(end is exclusive and must exceed start)"
            )

    def __len__(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class Trajectory:
    """One sampled completion, scored, with its GFlowNet state boundaries marked.

    Args:
        task_id: Identifier of the task/prompt this completion answers. Objectives that
            need a per-prompt normaliser (log Z is conditioned on the prompt) group by
            this field, so it must be stable across a group.
        prompt_tokens: The conditioning prompt, already tokenised. Never scored.
        completion_tokens: The sampled tokens. Everything else in this class is indexed
            against this list.
        sampling_logprobs: Behaviour-policy logprobs straight off the sampler, one per
            completion token. These are the logprobs of the policy *at sampling time*,
            which is the current policy only when training is strictly on-policy.
        log_reward: ``log R(x)`` at the terminal state. Log space, not raw reward: the
            objectives all work in logs and a raw reward of 0 is representable here as
            ``-inf``-ish only through the caller's own floor.
        segments: State boundaries, contiguous and covering the completion exactly.
        metadata: Free-form extras (stop reason, raw reward components, decoded text...).
            Never read by the objectives.

    Raises:
        ValueError: If the lengths disagree or the segments do not tile
            ``[0, len(completion_tokens))`` exactly.
    """

    task_id: str
    prompt_tokens: list[int]
    completion_tokens: list[int]
    sampling_logprobs: list[float]
    log_reward: float
    segments: list[Segment]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.completion_tokens)
        if n == 0:
            raise ValueError(
                f"Trajectory {self.task_id!r} has an empty completion; a trajectory must "
                "contain at least one sampled token"
            )
        if len(self.sampling_logprobs) != n:
            raise ValueError(
                f"Trajectory {self.task_id!r}: sampling_logprobs has length "
                f"{len(self.sampling_logprobs)} but completion_tokens has length {n}; "
                "they must agree one-to-one"
            )
        if not self.prompt_tokens:
            raise ValueError(
                f"Trajectory {self.task_id!r} has an empty prompt; the alignment contract "
                "needs at least one prompt token to build a model input"
            )
        if not self.segments:
            raise ValueError(
                f"Trajectory {self.task_id!r} has no segments; expected at least one "
                f"covering [0, {n})"
            )

        cursor = 0
        for i, seg in enumerate(self.segments):
            if seg.start != cursor:
                raise ValueError(
                    f"Trajectory {self.task_id!r}: segment {i} starts at {seg.start} but the "
                    f"previous segment ended at {cursor}; segments must be contiguous and "
                    "in ascending order with no gaps or overlaps"
                )
            cursor = seg.end
        if cursor != n:
            raise ValueError(
                f"Trajectory {self.task_id!r}: segments cover [0, {cursor}) but "
                f"completion_tokens has length {n}; segments must tile the completion exactly"
            )

    @property
    def num_completion_tokens(self) -> int:
        """Length of the completion, i.e. the shape objectives and gradients use."""
        return len(self.completion_tokens)

    @property
    def num_prompt_tokens(self) -> int:
        """Length of the prompt."""
        return len(self.prompt_tokens)

    @property
    def total_tokens(self) -> int:
        """Prompt plus completion, i.e. what Tinker bills a forward pass for."""
        return len(self.prompt_tokens) + len(self.completion_tokens)


TrajectoryBatch = Sequence[Trajectory]
"""A batch of trajectories. Deliberately a Sequence, not a list: objectives only iterate."""


@dataclass(frozen=True)
class TokenUsage:
    """Cumulative tokens a backend has actually pushed through Tinker.

    Lives here rather than in :mod:`flowcode.tinker_backend` so that
    :mod:`flowcode.cost` can price a run without importing the Tinker SDK.
    :meth:`flowcode.tinker_backend.TinkerBackend.token_usage` returns a snapshot of one
    of these; :func:`flowcode.cost.from_usage` turns it into dollars.

    Immutable on purpose: a snapshot taken mid-run must not mutate underneath the caller.
    Accumulate with ``+``.

    Args:
        train_tokens: Tokens processed by ``forward`` / ``forward_backward``, counted as
            ``model_input.length`` per datum and summed over every pass. Both the oracle
            forward and the gradient push count, so an off-policy step costs twice an
            on-policy one.
        sample_tokens: Tokens processed by sampling, counted as
            ``num_samples * prompt_length + total_generated``. This is the *gross*
            figure; subtract :attr:`prompt_cache_hit_tokens` for what is billed at the
            full sample rate.
        prompt_cache_hit_tokens: Prompt tokens Tinker reported as prefix-cache hits,
            summed verbatim from ``SampleResponse.prompt_cache_hit_tokens``.
        num_forward_passes: How many ``forward`` oracle passes were submitted.
        num_backward_passes: How many ``forward_backward`` gradient pushes were submitted.
        num_optim_steps: How many ``optim_step`` calls were submitted.
    """

    train_tokens: int = 0
    sample_tokens: int = 0
    prompt_cache_hit_tokens: int = 0
    num_forward_passes: int = 0
    num_backward_passes: int = 0
    num_optim_steps: int = 0

    @property
    def billable_sample_tokens(self) -> int:
        """Sample tokens net of reported cache hits, floored at zero."""
        return max(0, self.sample_tokens - self.prompt_cache_hit_tokens)

    def __add__(self, other: TokenUsage) -> TokenUsage:
        if not isinstance(other, TokenUsage):
            return NotImplemented
        return TokenUsage(
            train_tokens=self.train_tokens + other.train_tokens,
            sample_tokens=self.sample_tokens + other.sample_tokens,
            prompt_cache_hit_tokens=(self.prompt_cache_hit_tokens + other.prompt_cache_hit_tokens),
            num_forward_passes=self.num_forward_passes + other.num_forward_passes,
            num_backward_passes=self.num_backward_passes + other.num_backward_passes,
            num_optim_steps=self.num_optim_steps + other.num_optim_steps,
        )


@dataclass(frozen=True)
class SamplingParams:
    """How to draw a completion. Backend-neutral by design.

    Every sampler in this package takes one of these rather than its engine's own params
    object, so :func:`flowcode.rollout.rollout` does not know or care whether it is
    driving a hosted API or a local engine. Adapting to the engine's own type is the
    backend's job and happens at its edge.

    Args:
        max_tokens: Hard cap on generated tokens. A sequence that hits it is reported with
            ``stop_reason="length"`` and counted as truncated.
        temperature: Softmax temperature. ``0.0`` means greedy, which the eval path uses.
        top_p: Nucleus sampling mass. ``1.0`` disables it.
        stop: Stop conditions, either as strings or as token ids — whichever
            :func:`flowcode.render.stop_sequences` produced for the renderer. Engines that
            accept only one form convert at their edge.
        seed: Sampler seed, or ``None`` to leave it unseeded.
    """

    max_tokens: int
    temperature: float = 1.0
    top_p: float = 1.0
    stop: Sequence[str] | Sequence[int] = ()
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")
        if self.temperature < 0.0:
            raise ValueError(f"temperature must be non-negative, got {self.temperature}")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")


@dataclass(frozen=True)
class SampledSequence:
    """One completion off a sampler, with the behaviour policy's own logprobs.

    The logprobs are not optional decoration. They become
    :attr:`Trajectory.sampling_logprobs`, which is what the importance-weighting path in
    :func:`flowcode.objectives.base.importance_weights` divides by and what
    ``train.on_policy_only`` substitutes for the oracle forward pass. A sampler that
    returned tokens without them, and was silently given zeros instead, would be handing
    the objective a behaviour policy that assigns probability 1 to everything — so the
    length agreement is enforced here, once, for every backend that builds one.

    Args:
        tokens: Generated token ids. May be empty: a sampler that returns nothing is a
            degenerate sample, which :func:`flowcode.rollout.rollout` drops and counts
            rather than raising on.
        logprobs: ``log P(token)`` under the sampling policy, one per entry of ``tokens``.
        stop_reason: Why generation ended. ``"length"`` specifically is counted as a
            truncation in :class:`~flowcode.rollout.RolloutStats`; anything else is passed
            through to the trajectory metadata untouched.

    Raises:
        ValueError: If ``logprobs`` and ``tokens`` disagree in length.
    """

    tokens: list[int]
    logprobs: list[float]
    stop_reason: str = "unknown"

    def __post_init__(self) -> None:
        if len(self.logprobs) != len(self.tokens):
            raise ValueError(
                f"SampledSequence has {len(self.tokens)} tokens but {len(self.logprobs)} "
                "logprobs; a sampler must return one logprob per sampled token"
            )


@dataclass(frozen=True)
class SampleResponse:
    """Every completion drawn for one prompt.

    Args:
        sequences: The completions, ``num_samples`` of them for a healthy sampler. One
            response per prompt, so a GFlowNet group is exactly one of these.
        prompt_cache_hit_tokens: Prompt tokens the engine served from a prefix cache.
            Reported for throughput accounting and, on billed backends, because they are
            not charged at the full rate. Samplers without a prefix cache report ``0``.
    """

    sequences: list[SampledSequence]
    prompt_cache_hit_tokens: int = 0


def token_level_segments(n: int) -> list[Segment]:
    """Every token is its own state boundary — the finest DB/SubTB decomposition.

    Args:
        n: Number of completion tokens.

    Returns:
        ``n`` unit-length segments tiling ``[0, n)``.

    Raises:
        ValueError: If ``n`` is not positive.
    """
    if n <= 0:
        raise ValueError(f"token_level_segments needs a positive length, got {n}")
    return [Segment(start=i, end=i + 1) for i in range(n)]


def segments_from_boundaries(boundaries: Sequence[int], n: int) -> list[Segment]:
    """Build segments from a list of cut points.

    Args:
        boundaries: Interior cut points, each the exclusive end of a segment. Duplicates
            and a trailing ``n`` are tolerated; 0 is ignored. Need not be sorted.
        n: Number of completion tokens. A final segment ending at ``n`` is always
            appended, so passing ``[]`` gives the single-segment (plain TB) layout.

    Returns:
        Segments tiling ``[0, n)`` exactly.

    Raises:
        ValueError: If ``n`` is not positive or a boundary falls outside ``(0, n]``.
    """
    if n <= 0:
        raise ValueError(f"segments_from_boundaries needs a positive length, got {n}")
    for b in boundaries:
        if not 0 <= b <= n:
            raise ValueError(
                f"boundary {b} is outside [0, {n}]; boundaries index into completion_tokens"
            )
    cuts = sorted({b for b in boundaries if 0 < b < n})
    cuts.append(n)
    segments: list[Segment] = []
    start = 0
    for cut in cuts:
        segments.append(Segment(start=start, end=cut))
        start = cut
    return segments
