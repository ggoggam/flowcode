"""Turning an engine's output into :class:`~flowcode.types.SampleResponse`, purely.

Everything here is a plain function over plain data with **no vLLM import**, which is the
point: ``vllm`` has no macOS wheel and does not import without an accelerator, so if the
translation lived inside the engine wrapper it could not be tested anywhere except on the
hardware it is meant to run on. The engine wrappers in :mod:`flowcode.samplers.vllm` do
nothing but talk to the engine and hand its output here.

The structural types below describe the shape of a vLLM ``RequestOutput`` rather than
importing it, so the tests can substitute an ordinary object.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from flowcode.types import SampledSequence, SampleResponse, SamplingParams

__all__ = [
    "STOP_REASON_UNKNOWN",
    "CompletionOutputLike",
    "RequestOutputLike",
    "to_sample_response",
    "to_sampled_sequence",
    "vllm_sampling_kwargs",
]

STOP_REASON_UNKNOWN = "unknown"
"""What an engine that declines to say gets recorded as. Only ``"length"`` is given any
meaning downstream — :class:`~flowcode.rollout.RolloutStats` counts it as a truncation."""


class CompletionOutputLike(Protocol):
    """One completion inside a vLLM ``RequestOutput``.

    The members are declared read-only. A mutable protocol attribute is invariant, which
    would reject a ``list[int]`` where ``Sequence[int]`` is declared and make every caller
    — including the engine's own type — fail to match.
    """

    @property
    def token_ids(self) -> Sequence[int]: ...

    @property
    def logprobs(self) -> Sequence[Mapping[int, Any]] | None: ...

    @property
    def finish_reason(self) -> str | None: ...


class RequestOutputLike(Protocol):
    """A vLLM ``RequestOutput``: every completion drawn for a single prompt."""

    @property
    def outputs(self) -> Sequence[CompletionOutputLike]: ...


def vllm_sampling_kwargs(params: SamplingParams, num_samples: int) -> dict[str, Any]:
    """Translate neutral sampling params into vLLM ``SamplingParams`` keyword arguments.

    Returned as a dict rather than a constructed object so this stays importable without
    vLLM; the engine wrapper splats it into the real class.

    Two choices worth stating:

    * ``n=num_samples`` rather than ``num_samples`` separate requests. A GFlowNet group
      shares one prompt, so this lets the engine prefill once and fan out — the single
      biggest saving available in group-based sampling, and free.
    * ``logprobs=0`` asks for the logprob of the *sampled* token only, not a top-k table.
      That is exactly what :attr:`~flowcode.types.Trajectory.sampling_logprobs` needs, and
      any larger value would move megabytes per step for nothing.

    Args:
        params: The neutral parameters a rollout built.
        num_samples: Completions per prompt.

    Returns:
        Keyword arguments for ``vllm.SamplingParams``.

    Raises:
        ValueError: If ``num_samples`` is not positive.
    """
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")

    kwargs: dict[str, Any] = {
        "n": num_samples,
        "temperature": params.temperature,
        "top_p": params.top_p,
        "max_tokens": params.max_tokens,
        "logprobs": 0,
    }
    if params.seed is not None:
        kwargs["seed"] = params.seed

    # stop_sequences() returns strings for renderers with textual stops and token ids for
    # those whose stop is a special token; vLLM takes them under different names.
    stop = list(params.stop)
    if stop:
        if all(isinstance(s, int) for s in stop):
            kwargs["stop_token_ids"] = stop
        else:
            kwargs["stop"] = [str(s) for s in stop]
    return kwargs


def to_sampled_sequence(output: CompletionOutputLike) -> SampledSequence:
    """Adapt one vLLM ``CompletionOutput``.

    vLLM reports logprobs as one mapping per position, keyed by token id, holding whatever
    top-k was requested plus the sampled token itself. With ``logprobs=0`` each mapping
    holds a single entry, but it is looked up by the sampled id rather than assumed to be
    the only one, so raising ``logprobs`` for debugging does not silently corrupt training.

    Args:
        output: The engine's completion.

    Returns:
        The neutral sequence.

    Raises:
        ValueError: If tokens came back without logprobs, or a position's table does not
            contain its own sampled token. Substituting zeros would tell the objective the
            behaviour policy assigned probability 1 to everything it emitted.
    """
    token_ids = [int(t) for t in output.token_ids]
    finish_reason = output.finish_reason or STOP_REASON_UNKNOWN

    if not token_ids:
        return SampledSequence(tokens=[], logprobs=[], stop_reason=finish_reason)

    if output.logprobs is None:
        raise ValueError(
            "the engine returned tokens without logprobs; flowcode requests logprobs=0 on "
            "every sampling call, so this response is malformed. Sampling logprobs are the "
            "behaviour policy the objectives reweight against and cannot be defaulted."
        )
    if len(output.logprobs) != len(token_ids):
        raise ValueError(
            f"the engine returned {len(token_ids)} tokens but "
            f"{len(output.logprobs)} logprob tables; they must correspond position-by-position"
        )

    logprobs: list[float] = []
    for position, (token_id, table) in enumerate(zip(token_ids, output.logprobs, strict=True)):
        entry = table.get(token_id)
        if entry is None:
            raise ValueError(
                f"position {position} sampled token {token_id} but its logprob table holds "
                f"only {sorted(table)}; the engine's sampled token and its logprobs disagree"
            )
        # vLLM wraps the value in a Logprob object; a plain float is accepted too so the
        # tests do not have to model the wrapper.
        logprobs.append(float(getattr(entry, "logprob", entry)))

    return SampledSequence(tokens=token_ids, logprobs=logprobs, stop_reason=finish_reason)


def to_sample_response(output: RequestOutputLike, cached_tokens: int = 0) -> SampleResponse:
    """Adapt a whole vLLM ``RequestOutput`` — one prompt's entire group.

    Args:
        output: The engine's response for one prompt.
        cached_tokens: Prompt tokens served from the prefix cache, if the engine reported
            any. Counted once per prompt, not once per sample: the group shares a prefill.

    Returns:
        The neutral response.
    """
    return SampleResponse(
        sequences=[to_sampled_sequence(o) for o in output.outputs],
        prompt_cache_hit_tokens=int(cached_tokens),
    )
