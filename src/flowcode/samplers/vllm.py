"""vLLM samplers, in the two topologies a run can be deployed in.

Both satisfy :class:`~flowcode.local_backend.WeightSyncSampler`, so
:class:`~flowcode.local_backend.LocalBackend` drives either without knowing which it has,
and the choice is a config flag rather than a code path through the trainer.

``colocated``
    An ``AsyncLLMEngine`` in this process, sharing devices with the trainer under a capped
    ``gpu_memory_utilization``. Runs on a single GPU or chip and syncs weights by
    registering a new adapter in-process. The binding constraint is memory: base model,
    LoRA, activations and KV cache all on the same device.

``remote``
    An engine already running elsewhere, reached over its OpenAI-compatible HTTP server.
    Sampling and training never contend. Note this mode does **not** require ``vllm`` to be
    installed locally — the engine is on the other host, and all that is needed here is an
    HTTP client, which is why ``httpx`` lives in the ``local`` extra.

Why vLLM at all
---------------
Continuous batching and paged KV are what make a decoupled producer worth building: the
default step asks for only 64 concurrent sequences, which leaves any modern accelerator
mostly idle during decode, and the fix is to keep many more requests in flight than one
step needs. The engine also has the same API on CUDA and TPU, which is what lets the
hardware decision stay open.

**Unverified on hardware.** The pure translation layer in
:mod:`flowcode.samplers.adapters` and the request/response shaping below are unit-tested,
but neither engine path has been run against a real vLLM. Two things to check first on a
box that can: whether the deployed version accepts token-id prompts on the completions
endpoint and returns token ids back, and whether it will load an adapter that includes
``lm_head`` (``model.train_unembed`` defaults to ``true``, and vLLM's LoRA support does not
reliably cover the unembedding — expect to set it ``false``).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import Any

from flowcode.samplers.adapters import to_sample_response, vllm_sampling_kwargs
from flowcode.types import SampledSequence, SampleResponse, SamplingParams

__all__ = ["ColocatedVllmSampler", "RemoteVllmSampler", "parse_completion_choice"]

logger = logging.getLogger(__name__)


class ColocatedVllmSampler:
    """An ``AsyncLLMEngine`` in this process, sharing devices with the trainer."""

    def __init__(self, engine: Any, *, lora_name: str = "policy") -> None:
        """Wrap an already-constructed engine.

        Args:
            engine: A ``vllm.AsyncLLMEngine``. Taken as a constructed object rather than
                built here so tests can substitute a fake.
            lora_name: Name the adapter is registered under. The *id* is the policy
                version, which is what actually selects a revision.
        """
        self._engine = engine
        self._lora_name = lora_name
        self._lora_request: Any = None
        self._version = 0

    @classmethod
    def create(
        cls,
        model_name: str,
        *,
        max_lora_rank: int = 32,
        gpu_memory_utilization: float = 0.4,
        max_model_len: int | None = None,
        enable_prefix_caching: bool = True,
        tensor_parallel_size: int = 1,
        dtype: str = "bfloat16",
        seed: int = 0,
        **engine_kwargs: Any,
    ) -> ColocatedVllmSampler:
        """Build the in-process engine.

        Args:
            model_name: Base model identifier, matching the trainer's.
            max_lora_rank: Must be at least ``model.lora_rank``; the engine rejects an
                adapter above whatever it was built for.
            gpu_memory_utilization: Fraction of device memory the engine may claim for
                weights and KV cache. The trainer needs the rest, so this is well below
                the single-tenant default.
            max_model_len: Context window; ``None`` takes the model's own.
            enable_prefix_caching: Share prefills across requests. A GFlowNet group already
                shares its prefill via ``n=group_size``; this additionally shares across
                the repeated sampling of the same task over a run.
            tensor_parallel_size: Devices to shard the engine over.
            dtype: Engine dtype. Should match the trainer's, or sampled logprobs and
                recomputed ones will disagree by more than arithmetic noise.
            seed: Engine seed.
            **engine_kwargs: Passed through to ``AsyncEngineArgs`` untouched.

        Returns:
            A ready sampler.

        Raises:
            ImportError: If the ``vllm`` extra is not installed.
        """
        try:
            from vllm import AsyncEngineArgs, AsyncLLMEngine
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "Colocated sampling needs the `vllm` extra, which has no macOS wheel. "
                "Install it on a Linux GPU/TPU host with `mise run sync:vllm`, or use "
                "sampler.mode=remote to reach an engine on another host."
            ) from exc

        args = AsyncEngineArgs(
            model=model_name,
            enable_lora=True,
            max_lora_rank=max_lora_rank,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enable_prefix_caching=enable_prefix_caching,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            seed=seed,
            **engine_kwargs,
        )
        engine = AsyncLLMEngine.from_engine_args(args)
        logger.info(
            "colocated vLLM engine ready: model=%s tp=%d gpu_mem=%.2f",
            model_name,
            tensor_parallel_size,
            gpu_memory_utilization,
        )
        return cls(engine)

    async def sample(
        self,
        prompts: Sequence[Sequence[int]],
        num_samples: int,
        sampling_params: SamplingParams,
    ) -> list[SampleResponse]:
        """Draw ``num_samples`` completions per prompt, all prompts concurrently.

        Args:
            prompts: One token-id list per prompt. Order is preserved in the result.
            num_samples: Completions per prompt.
            sampling_params: Neutral sampling parameters.

        Returns:
            One response per prompt, in the same order.

        Raises:
            ValueError: If ``prompts`` is empty.
        """
        if not prompts:
            raise ValueError("sample() needs at least one prompt")

        from vllm import SamplingParams as VllmSamplingParams
        from vllm.inputs import TokensPrompt

        params = VllmSamplingParams(**vllm_sampling_kwargs(sampling_params, num_samples))
        version = self._version
        lora_request = self._lora_request

        async def one(index: int, prompt: Sequence[int]) -> SampleResponse:
            # Every request carries the adapter that was current when the batch started,
            # so a mid-batch sync cannot split a group across two policy revisions.
            request_id = f"flowcode-{version}-{index}"
            final = None
            async for output in self._engine.generate(
                TokensPrompt(prompt_token_ids=list(prompt)),
                params,
                request_id,
                lora_request=lora_request,
            ):
                final = output
            if final is None:
                raise RuntimeError(f"the engine produced no output for request {request_id}")
            return to_sample_response(final, getattr(final, "num_cached_tokens", 0) or 0)

        return list(await asyncio.gather(*(one(i, p) for i, p in enumerate(prompts))))

    async def sync_weights(self, adapter_path: str, version: int) -> None:
        """Register the adapter so subsequent requests use it.

        Requests already in flight finish on the previous adapter, which is staleness
        rather than error — see :meth:`flowcode.local_backend.LocalBackend.sync_sampler`.

        Args:
            adapter_path: Directory a PEFT adapter was saved to.
            version: Monotonic policy revision, used as the engine's LoRA id.
        """
        from vllm.lora.request import LoRARequest

        self._version = version
        self._lora_request = LoRARequest(self._lora_name, version, adapter_path)
        logger.debug("colocated sampler now serving policy version %d", version)


def parse_completion_choice(choice: dict[str, Any]) -> SampledSequence:
    """Adapt one ``choice`` from an OpenAI-compatible ``/v1/completions`` response.

    vLLM's server reports per-position logprobs under ``logprobs``, with ``token_logprobs``
    holding the sampled token's own logprob and ``top_logprobs`` holding the requested
    table. Token *ids* come back only when the server is asked for them; when they are
    absent this falls back to the tokens the request already knows it sent, which is why
    :meth:`RemoteVllmSampler.sample` keeps them.

    Args:
        choice: One element of the response's ``choices`` array.

    Returns:
        The neutral sequence.

    Raises:
        ValueError: If the response carries no per-token logprobs.
    """
    logprob_block = choice.get("logprobs") or {}
    token_logprobs = logprob_block.get("token_logprobs")
    token_ids = logprob_block.get("token_ids") or choice.get("token_ids")
    finish_reason = choice.get("finish_reason") or "unknown"

    if token_logprobs is None:
        raise ValueError(
            "the server returned no token_logprobs; flowcode requests logprobs on every "
            "call because they are the behaviour policy the objectives reweight against. "
            "Check that the deployed vLLM accepts `logprobs` on /v1/completions."
        )
    if token_ids is None:
        raise ValueError(
            "the server returned logprobs without token ids. flowcode trains on token ids, "
            "not detokenised text, so a response it cannot map back to ids is unusable. "
            "The deployed vLLM must support returning token ids on /v1/completions."
        )

    tokens = [int(t) for t in token_ids]
    # A leading null appears when the server echoes the prompt; drop the pair together.
    logprobs = [0.0 if lp is None else float(lp) for lp in token_logprobs]
    if len(logprobs) != len(tokens):
        raise ValueError(
            f"the server returned {len(tokens)} token ids but {len(logprobs)} logprobs; "
            "they must correspond position-by-position"
        )
    return SampledSequence(tokens=tokens, logprobs=logprobs, stop_reason=finish_reason)


class RemoteVllmSampler:
    """An engine on another host, reached over its OpenAI-compatible HTTP server.

    Does not import ``vllm``: the engine is elsewhere, so only an HTTP client is needed.
    """

    def __init__(
        self,
        client: Any,
        model_name: str,
        *,
        lora_name: str = "policy",
        request_timeout: float = 600.0,
    ) -> None:
        """Wrap an HTTP client pointed at the server's base URL.

        Args:
            client: An ``httpx.AsyncClient`` whose ``base_url`` is the server root. Taken
                as a constructed object so tests can substitute a transport.
            model_name: Model name to request. Replaced by the adapter's name once one has
                been synced, which is how the server routes to the LoRA.
            lora_name: Name the adapter is registered under on the server.
            request_timeout: Per-request timeout in seconds. Generous by default: a group
                of long completions is not a fast request, and a timeout mid-generation
                wastes everything already decoded.
        """
        self._client = client
        self._base_model = model_name
        self._lora_name = lora_name
        self._request_timeout = request_timeout
        self._version = 0

    @classmethod
    def create(
        cls,
        base_url: str,
        model_name: str,
        *,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> RemoteVllmSampler:
        """Build an HTTP client for the server at ``base_url``.

        Args:
            base_url: Server root, e.g. ``http://10.0.0.4:8000``.
            model_name: Base model name the server was started with.
            api_key: Bearer token, if the server was started with one.
            **kwargs: Passed to the constructor.

        Returns:
            A ready sampler.

        Raises:
            ImportError: If ``httpx`` is not installed.
        """
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "Remote sampling needs `httpx`, part of the `local` extra. Install it with "
                "`mise run sync`."
            ) from exc

        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        client = httpx.AsyncClient(base_url=base_url, headers=headers, timeout=None)
        return cls(client, model_name, **kwargs)

    @property
    def _served_model(self) -> str:
        """The adapter once one has been synced, else the base model."""
        return self._lora_name if self._version else self._base_model

    async def sample(
        self,
        prompts: Sequence[Sequence[int]],
        num_samples: int,
        sampling_params: SamplingParams,
    ) -> list[SampleResponse]:
        """Draw ``num_samples`` completions per prompt, all prompts concurrently.

        Args:
            prompts: One token-id list per prompt. Sent as token ids, not text: a
                round-trip through detokenisation is not guaranteed to be the identity and
                a prompt that shifts by one token silently breaks the alignment contract.
            num_samples: Completions per prompt.
            sampling_params: Neutral sampling parameters.

        Returns:
            One response per prompt, in the same order.

        Raises:
            ValueError: If ``prompts`` is empty.
            RuntimeError: If the server returns a non-2xx response.
        """
        if not prompts:
            raise ValueError("sample() needs at least one prompt")

        kwargs = vllm_sampling_kwargs(sampling_params, num_samples)
        model = self._served_model

        async def one(prompt: Sequence[int]) -> SampleResponse:
            payload: dict[str, Any] = {
                "model": model,
                "prompt": list(prompt),
                "return_token_ids": True,
                **kwargs,
            }
            response = await self._client.post(
                "/v1/completions", json=payload, timeout=self._request_timeout
            )
            if response.status_code >= 400:
                raise RuntimeError(
                    f"vLLM server returned {response.status_code} for a sampling request: "
                    f"{response.text[:500]}"
                )
            body = response.json()
            return SampleResponse(
                sequences=[parse_completion_choice(c) for c in body.get("choices", [])],
                prompt_cache_hit_tokens=int(
                    (body.get("usage") or {}).get("prompt_cache_hit_tokens", 0)
                ),
            )

        return list(await asyncio.gather(*(one(p) for p in prompts)))

    async def sync_weights(self, adapter_path: str, version: int) -> None:
        """Load the adapter onto the server, replacing the previous revision.

        The server must have been started with ``--enable-lora`` and
        ``VLLM_ALLOW_RUNTIME_LORA_UPDATING=1``; without the latter the load endpoint is not
        mounted and this raises rather than silently continuing to serve stale weights.

        The path is interpreted *by the server*, so on a genuinely remote host it has to be
        on shared storage both sides can see.

        Args:
            adapter_path: Directory a PEFT adapter was saved to.
            version: Monotonic policy revision.

        Raises:
            RuntimeError: If the server rejects the load.
        """
        if self._version:
            # Unload first: re-registering a live name is rejected by the server.
            await self._client.post("/v1/unload_lora_adapter", json={"lora_name": self._lora_name})
        response = await self._client.post(
            "/v1/load_lora_adapter",
            json={"lora_name": self._lora_name, "lora_path": adapter_path},
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"vLLM server refused the adapter at {adapter_path!r} "
                f"({response.status_code}): {response.text[:500]}. The server needs "
                "--enable-lora and VLLM_ALLOW_RUNTIME_LORA_UPDATING=1, and the path must be "
                "readable from the server's host. If the adapter includes lm_head "
                "(model.train_unembed=true), try disabling it: vLLM's LoRA support does not "
                "reliably cover the unembedding."
            )
        self._version = version
        logger.debug("remote sampler now serving policy version %d", version)

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()
