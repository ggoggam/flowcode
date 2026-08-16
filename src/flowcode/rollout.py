"""Sampling completions and turning them into scored :class:`~flowcode.types.Trajectory`.

One call to :func:`rollout` is one GFlowNet batch: render every task's prompt, draw
``group_size`` completions per prompt, execute them against the hidden tests, and package
the result with everything an objective needs.

Two things here are about throughput, not correctness, and both are easy to get wrong.

**Scoring must leave the event loop.** ``env.batch_log_reward`` forks one subprocess per
completion and blocks in ``waitpid``; called directly from a coroutine it stalls the loop,
and with it every other in-flight Tinker request. :func:`asyncio.to_thread` hands the whole
batch to a worker thread, where the environment's own :class:`ThreadPoolExecutor` fans it
out. The GIL is not a factor: the workers are blocked in the kernel the entire time.

**Sampling is one call, not one per prompt.**
:meth:`flowcode.tinker_backend.TinkerBackend.sample` already gathers concurrently over
prompts, so a per-prompt loop here would serialise what the backend deliberately
parallelised.

Segments
--------
Trajectories are built with :func:`~flowcode.types.token_level_segments`, i.e. a state
boundary after every completion token. This environment is single-turn — one prompt, one
reply, no tool calls — so there are no genuine turn boundaries to mark, and inventing some
(after the first fence, after the docstring...) would be fabricating structure the sampler
does not have. Token level is the finest *honest* decomposition and the standard setting
for LLM GFlowNets. Note the consequence: because the environment offers no coarser
segmentation, ``objective.granularity=turn`` sees the same boundaries as ``token`` rather
than a single terminal state. TB and VarGrad ignore segments entirely.

Degenerate samples
------------------
A zero-token completion cannot be a :class:`~flowcode.types.Trajectory` (the type refuses
one, correctly — there is nothing to compute a logprob for) and cannot be scored. Those
samples are dropped and counted in :class:`RolloutStats` rather than raised on: a sampler
that returns one empty sequence in a thousand should not end a paid run.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from tinker import types as tinker_types

from flowcode.config import RootConfig
from flowcode.envs.base import RewardResult, Task
from flowcode.envs.extract import extract_code
from flowcode.render import ChatTokenizer, completion_to_text, render_prompt, stop_sequences
from flowcode.types import Trajectory, token_level_segments

__all__ = ["RolloutStats", "SamplerLike", "ScorerLike", "rollout", "solution_fingerprint"]

logger = logging.getLogger(__name__)


class SamplerLike(Protocol):
    """The part of :class:`~flowcode.tinker_backend.TinkerBackend` a rollout uses."""

    async def sample(
        self,
        prompts: Sequence[Sequence[int]],
        num_samples: int,
        sampling_params: tinker_types.SamplingParams,
    ) -> list[tinker_types.SampleResponse]:
        """Draw ``num_samples`` completions per prompt, in prompt order."""
        ...


class ScorerLike(Protocol):
    """The part of :class:`~flowcode.envs.base.Environment` a rollout uses."""

    def batch_log_reward(self, pairs: Sequence[tuple[Task, str]]) -> list[RewardResult]:
        """Score ``(task, completion_text)`` pairs concurrently, in input order."""
        ...


@dataclass
class RolloutStats:
    """What one rollout did, beyond the trajectories themselves.

    Args:
        num_requested: ``len(tasks) * num_samples``.
        num_returned: Sequences the sampler actually returned.
        num_empty: Sequences dropped for having no tokens.
        num_truncated: Sequences that hit ``max_tokens`` instead of a stop condition. A
            high rate means the completion budget is clipping real answers, which shows up
            as an unexplained pile of syntax errors in the reward.
        completion_tokens: Total sampled tokens kept.
        prompt_cache_hit_tokens: Prompt tokens Tinker served from its prefix cache.
    """

    num_requested: int = 0
    num_returned: int = 0
    num_empty: int = 0
    num_truncated: int = 0
    completion_tokens: int = 0
    prompt_cache_hit_tokens: int = 0

    def as_metrics(self, prefix: str = "rollout") -> dict[str, float]:
        """Flatten to ``{f"{prefix}/{field}": value}`` for the logger."""
        return {
            f"{prefix}/num_requested": float(self.num_requested),
            f"{prefix}/num_returned": float(self.num_returned),
            f"{prefix}/num_empty": float(self.num_empty),
            f"{prefix}/num_truncated": float(self.num_truncated),
            f"{prefix}/completion_tokens": float(self.completion_tokens),
            f"{prefix}/prompt_cache_hit_tokens": float(self.prompt_cache_hit_tokens),
        }


@dataclass(frozen=True)
class _Candidate:
    """One sampled sequence on its way to becoming a trajectory."""

    task: Task
    prompt_tokens: list[int]
    completion_tokens: list[int]
    sampling_logprobs: list[float]
    stop_reason: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def solution_fingerprint(completion_text: str) -> str:
    """A short, stable hash of the *code* in a completion.

    Two completions that differ only in prose or whitespace map to the same fingerprint, so
    counting distinct fingerprints among the passing samples of one task is a usable
    "how many modes did we find" statistic — the property a GFlowNet is supposed to win on
    against a reward-maximising policy, and therefore the one worth measuring.

    Args:
        completion_text: Raw completion text.

    Returns:
        16 hex characters, or ``""`` when no code could be extracted.
    """
    code = extract_code(completion_text)
    normalised = "\n".join(line.rstrip() for line in code.split("\n") if line.strip())
    if not normalised:
        return ""
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:16]


def _sequence_arrays(sequence: tinker_types.SampledSequence) -> tuple[list[int], list[float]]:
    """Pull ``(tokens, logprobs)`` off a sampled sequence, trimmed to a common length.

    Raises:
        ValueError: If the sampler returned no logprobs. They arrive unconditionally on the
            real API, so their absence means the response is not what this code thinks it
            is — and silently substituting zeros would hand the objective a behaviour
            policy that assigns probability 1 to everything.
    """
    tokens_np = sequence.tokens_np
    logprobs_np = sequence.logprobs_np
    tokens = [int(t) for t in (np.asarray(tokens_np).reshape(-1) if tokens_np is not None else [])]
    if logprobs_np is None:
        if not tokens:
            return [], []
        raise ValueError(
            "the sampler returned tokens without logprobs; SampledSequence.logprobs_np is "
            "populated unconditionally by the Tinker API, so this response is malformed"
        )
    logprobs = [float(v) for v in np.asarray(logprobs_np).reshape(-1)]
    if len(logprobs) != len(tokens):
        # Defensive: keep the pair aligned rather than letting Trajectory reject the batch.
        keep = min(len(tokens), len(logprobs))
        logger.warning(
            "sampler returned %d tokens but %d logprobs; truncating both to %d",
            len(tokens),
            len(logprobs),
            keep,
        )
        tokens, logprobs = tokens[:keep], logprobs[:keep]
    return tokens, logprobs


def _build_sampling_params(
    cfg: RootConfig,
    tokenizer: ChatTokenizer,
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int | None,
) -> tinker_types.SamplingParams:
    """Assemble ``SamplingParams``, including the renderer's stop conditions."""
    return tinker_types.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop_sequences(tokenizer, cfg.model.renderer),
        seed=seed,
    )


async def rollout(
    backend: SamplerLike,
    env: ScorerLike,
    tokenizer: ChatTokenizer,
    tasks: Sequence[Task],
    cfg: RootConfig,
    *,
    num_samples: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    seed: int | None = None,
    stats: RolloutStats | None = None,
) -> list[Trajectory]:
    """Sample completions for ``tasks`` and score them into trajectories.

    Args:
        backend: Anything with :meth:`SamplerLike.sample` — the real
            :class:`~flowcode.tinker_backend.TinkerBackend` in production.
        env: The scoring environment.
        tokenizer: The model's tokenizer, for rendering and decoding.
        tasks: Prompts to condition on. One group per task.
        cfg: The composed run config; supplies ``model.renderer`` and the ``train.*``
            sampling knobs.
        num_samples: Completions per task. Defaults to ``cfg.train.group_size``. The eval
            path passes 1.
        temperature: Overrides ``cfg.train.temperature``. The eval path passes ``0.0`` for
            greedy decoding.
        top_p: Overrides ``cfg.train.top_p``.
        max_tokens: Overrides ``cfg.train.max_tokens``.
        seed: Sampler seed, forwarded to Tinker. ``None`` leaves it unseeded.
        stats: Optional :class:`RolloutStats` to accumulate into; a fresh one is used when
            omitted.

    Returns:
        Scored trajectories, grouped by task in task order. Shorter than
        ``len(tasks) * num_samples`` when degenerate samples were dropped.

    Raises:
        ValueError: If ``tasks`` is empty, ``num_samples`` is not positive, or the sampler
            returns a malformed response.
    """
    if not tasks:
        raise ValueError("rollout() needs at least one task")
    group_size = cfg.train.group_size if num_samples is None else num_samples
    if group_size <= 0:
        raise ValueError(f"num_samples must be positive, got {group_size}")

    tracker = RolloutStats() if stats is None else stats
    renderer = cfg.model.renderer

    prompts = [render_prompt(tokenizer, task, renderer) for task in tasks]
    params = _build_sampling_params(
        cfg,
        tokenizer,
        max_tokens=cfg.train.max_tokens if max_tokens is None else max_tokens,
        temperature=cfg.train.temperature if temperature is None else temperature,
        top_p=cfg.train.top_p if top_p is None else top_p,
        seed=seed,
    )

    responses = await backend.sample(prompts, num_samples=group_size, sampling_params=params)
    if len(responses) != len(tasks):
        raise ValueError(
            f"sampler returned {len(responses)} responses for {len(tasks)} prompts; they must "
            "correspond one-to-one and in order"
        )
    tracker.num_requested += len(tasks) * group_size

    candidates: list[_Candidate] = []
    for task, prompt_tokens, response in zip(tasks, prompts, responses, strict=True):
        tracker.prompt_cache_hit_tokens += int(response.prompt_cache_hit_tokens)
        for sequence in response.sequences:
            tracker.num_returned += 1
            tokens, logprobs = _sequence_arrays(sequence)
            stop_reason = str(getattr(sequence, "stop_reason", "unknown"))
            if not tokens:
                # Nothing was sampled: unscoreable and unrepresentable. Drop it.
                tracker.num_empty += 1
                continue
            if stop_reason == "length":
                tracker.num_truncated += 1
            tracker.completion_tokens += len(tokens)
            candidates.append(
                _Candidate(
                    task=task,
                    prompt_tokens=list(prompt_tokens),
                    completion_tokens=tokens,
                    sampling_logprobs=logprobs,
                    stop_reason=stop_reason,
                    text=completion_to_text(tokenizer, tokens, renderer),
                )
            )

    if not candidates:
        logger.warning("rollout produced no usable completions for %d task(s)", len(tasks))
        return []

    # Off the event loop: batch_log_reward blocks in waitpid for the whole batch.
    pairs = [(candidate.task, candidate.text) for candidate in candidates]
    results = await asyncio.to_thread(env.batch_log_reward, pairs)
    if len(results) != len(pairs):
        raise ValueError(
            f"env.batch_log_reward returned {len(results)} results for {len(pairs)} pairs; "
            "results must be in input order and one-to-one"
        )

    trajectories: list[Trajectory] = []
    for candidate, result in zip(candidates, results, strict=True):
        trajectories.append(
            Trajectory(
                task_id=candidate.task.task_id,
                prompt_tokens=candidate.prompt_tokens,
                completion_tokens=candidate.completion_tokens,
                sampling_logprobs=candidate.sampling_logprobs,
                log_reward=result.log_reward,
                segments=token_level_segments(len(candidate.completion_tokens)),
                metadata={
                    "pass_fraction": result.pass_fraction,
                    "passed": result.passed,
                    "error": result.error,
                    "stop_reason": candidate.stop_reason,
                    # Not the completion text: the buffer holds thousands of these and the
                    # text is the largest field. A fingerprint is all the mode-diversity
                    # metric needs.
                    "solution_fingerprint": solution_fingerprint(candidate.text),
                },
            )
        )
    return trajectories
