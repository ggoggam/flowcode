"""Sampler translation and request shaping, without an engine.

``vllm`` is not installed here and cannot be — no macOS wheel — so what is pinned is
everything *around* the engine: how neutral sampling params become engine params, how
engine output becomes a :class:`~flowcode.types.SampleResponse`, and how the remote
sampler shapes its HTTP calls. The engine calls themselves are exercised against fakes.

What this file does **not** prove is that a real vLLM behaves as modelled. That needs a
GPU or TPU host; see the note at the top of :mod:`flowcode.samplers.vllm`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from flowcode.samplers import SAMPLER_MODES, build_sampler
from flowcode.samplers.adapters import (
    to_sample_response,
    to_sampled_sequence,
    vllm_sampling_kwargs,
)
from flowcode.samplers.vllm import RemoteVllmSampler, parse_completion_choice
from flowcode.types import SamplingParams


@dataclass
class FakeLogprob:
    """vLLM wraps each logprob in an object with a ``.logprob`` attribute."""

    logprob: float


@dataclass
class FakeCompletionOutput:
    token_ids: list[int]
    logprobs: list[dict[int, Any]] | None
    finish_reason: str | None = "stop"


@dataclass
class FakeRequestOutput:
    outputs: list[FakeCompletionOutput]
    num_cached_tokens: int = 0


def wrapped(tokens: list[int], values: list[float]) -> list[dict[int, FakeLogprob]]:
    """The per-position logprob tables vLLM returns for ``logprobs=0``."""
    return [{t: FakeLogprob(v)} for t, v in zip(tokens, values, strict=True)]


# --------------------------------------------------------------------- sampling params


class TestVllmSamplingKwargs:
    def test_group_becomes_n_not_repeated_requests(self) -> None:
        # A GFlowNet group shares one prompt, so n=group_size lets the engine prefill once
        # and fan out. Issuing group_size separate requests would prefill group_size times.
        assert vllm_sampling_kwargs(SamplingParams(max_tokens=8), 8)["n"] == 8

    def test_requests_only_the_sampled_tokens_logprob(self) -> None:
        assert vllm_sampling_kwargs(SamplingParams(max_tokens=8), 1)["logprobs"] == 0

    def test_carries_temperature_top_p_and_max_tokens(self) -> None:
        kwargs = vllm_sampling_kwargs(SamplingParams(max_tokens=256, temperature=0.7, top_p=0.9), 4)
        assert kwargs["temperature"] == pytest.approx(0.7)
        assert kwargs["top_p"] == pytest.approx(0.9)
        assert kwargs["max_tokens"] == 256

    def test_seed_is_omitted_when_unset(self) -> None:
        assert "seed" not in vllm_sampling_kwargs(SamplingParams(max_tokens=8), 1)
        assert vllm_sampling_kwargs(SamplingParams(max_tokens=8, seed=17), 1)["seed"] == 17

    def test_string_stops_go_to_stop(self) -> None:
        kwargs = vllm_sampling_kwargs(SamplingParams(max_tokens=8, stop=["<|im_end|>"]), 1)
        assert kwargs["stop"] == ["<|im_end|>"]
        assert "stop_token_ids" not in kwargs

    def test_token_id_stops_go_to_stop_token_ids(self) -> None:
        # stop_sequences() returns ids for renderers whose stop is a special token; vLLM
        # takes the two forms under different names and silently ignores the wrong one.
        kwargs = vllm_sampling_kwargs(SamplingParams(max_tokens=8, stop=[151645]), 1)
        assert kwargs["stop_token_ids"] == [151645]
        assert "stop" not in kwargs

    def test_empty_stop_sets_neither(self) -> None:
        kwargs = vllm_sampling_kwargs(SamplingParams(max_tokens=8), 1)
        assert "stop" not in kwargs
        assert "stop_token_ids" not in kwargs

    def test_non_positive_num_samples_rejected(self) -> None:
        with pytest.raises(ValueError, match="num_samples must be positive"):
            vllm_sampling_kwargs(SamplingParams(max_tokens=8), 0)


# ------------------------------------------------------------------- engine translation


class TestToSampledSequence:
    def test_pulls_the_sampled_tokens_logprob(self) -> None:
        output = FakeCompletionOutput([7, 8, 9], wrapped([7, 8, 9], [-0.1, -0.2, -0.3]))
        sequence = to_sampled_sequence(output)
        assert sequence.tokens == [7, 8, 9]
        assert sequence.logprobs == pytest.approx([-0.1, -0.2, -0.3])

    def test_looks_up_by_token_id_not_by_position(self) -> None:
        # With logprobs>0 for debugging, each table holds several entries. Taking "the
        # only one" would then silently pick a token the policy did not emit.
        output = FakeCompletionOutput(
            [7], [{3: FakeLogprob(-0.01), 7: FakeLogprob(-2.5), 11: FakeLogprob(-9.0)}]
        )
        assert to_sampled_sequence(output).logprobs == pytest.approx([-2.5])

    def test_accepts_bare_floats(self) -> None:
        assert to_sampled_sequence(
            FakeCompletionOutput([4], [{4: -1.25}])
        ).logprobs == pytest.approx([-1.25])

    def test_carries_finish_reason(self) -> None:
        output = FakeCompletionOutput([1], wrapped([1], [-0.5]), finish_reason="length")
        assert to_sampled_sequence(output).stop_reason == "length"

    def test_missing_finish_reason_becomes_unknown(self) -> None:
        output = FakeCompletionOutput([1], wrapped([1], [-0.5]), finish_reason=None)
        assert to_sampled_sequence(output).stop_reason == "unknown"

    def test_empty_completion_is_tolerated(self) -> None:
        # Degenerate, not malformed: rollout drops and counts these.
        sequence = to_sampled_sequence(FakeCompletionOutput([], None))
        assert sequence.tokens == []
        assert sequence.logprobs == []

    def test_tokens_without_logprobs_are_an_error(self) -> None:
        with pytest.raises(ValueError, match="without logprobs"):
            to_sampled_sequence(FakeCompletionOutput([1, 2], None))

    def test_table_missing_its_own_token_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="sampled token and its logprobs disagree"):
            to_sampled_sequence(FakeCompletionOutput([7], [{3: FakeLogprob(-0.5)}]))

    def test_length_mismatch_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="position-by-position"):
            to_sampled_sequence(FakeCompletionOutput([1, 2, 3], wrapped([1], [-0.5])))


class TestToSampleResponse:
    def test_adapts_every_completion_in_the_group(self) -> None:
        output = FakeRequestOutput(
            [
                FakeCompletionOutput([1, 2], wrapped([1, 2], [-0.1, -0.2])),
                FakeCompletionOutput([3], wrapped([3], [-0.3])),
            ]
        )
        response = to_sample_response(output)
        assert len(response.sequences) == 2
        assert response.sequences[1].tokens == [3]

    def test_carries_cached_tokens(self) -> None:
        output = FakeRequestOutput([FakeCompletionOutput([1], wrapped([1], [-0.1]))])
        assert to_sample_response(output, 42).prompt_cache_hit_tokens == 42


# ----------------------------------------------------------------------- remote server


class TestParseCompletionChoice:
    def test_reads_token_ids_and_logprobs(self) -> None:
        choice = {
            "finish_reason": "stop",
            "logprobs": {"token_ids": [5, 6], "token_logprobs": [-0.5, -1.5]},
        }
        sequence = parse_completion_choice(choice)
        assert sequence.tokens == [5, 6]
        assert sequence.logprobs == pytest.approx([-0.5, -1.5])
        assert sequence.stop_reason == "stop"

    def test_a_null_logprob_becomes_zero(self) -> None:
        choice = {"logprobs": {"token_ids": [5], "token_logprobs": [None]}}
        assert parse_completion_choice(choice).logprobs == pytest.approx([0.0])

    def test_missing_logprobs_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="no token_logprobs"):
            parse_completion_choice({"finish_reason": "stop"})

    def test_missing_token_ids_is_an_error(self) -> None:
        # Detokenised text is not a substitute: flowcode trains on ids, and a text
        # round-trip is not guaranteed to be the identity.
        with pytest.raises(ValueError, match="token ids"):
            parse_completion_choice({"logprobs": {"token_logprobs": [-0.5]}})

    def test_length_mismatch_is_an_error(self) -> None:
        choice = {"logprobs": {"token_ids": [5, 6], "token_logprobs": [-0.5]}}
        with pytest.raises(ValueError, match="position-by-position"):
            parse_completion_choice(choice)


@dataclass
class FakeResponse:
    status_code: int = 200
    payload: dict[str, Any] = field(default_factory=dict)
    text: str = ""

    def json(self) -> dict[str, Any]:
        return self.payload


class FakeHttpClient:
    """Records posts and replays canned responses keyed by path."""

    def __init__(self, responses: dict[str, FakeResponse] | None = None) -> None:
        self.posts: list[tuple[str, dict[str, Any]]] = []
        self.responses = responses or {}

    async def post(self, path: str, json: dict[str, Any], **kwargs: Any) -> FakeResponse:
        self.posts.append((path, json))
        return self.responses.get(
            path,
            FakeResponse(
                payload={
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "logprobs": {"token_ids": [1, 2], "token_logprobs": [-0.1, -0.2]},
                        }
                    ],
                    "usage": {"prompt_cache_hit_tokens": 3},
                }
            ),
        )

    async def aclose(self) -> None:
        return None


class TestRemoteVllmSampler:
    async def test_sends_token_ids_not_text(self) -> None:
        # A round-trip through detokenisation is not guaranteed to be the identity, and a
        # prompt that shifts by one token breaks the alignment contract silently.
        client = FakeHttpClient()
        sampler = RemoteVllmSampler(client, "Qwen/Qwen3-8B")
        await sampler.sample([[10, 11, 12]], 2, SamplingParams(max_tokens=8))
        path, payload = client.posts[0]
        assert path == "/v1/completions"
        assert payload["prompt"] == [10, 11, 12]
        assert payload["n"] == 2

    async def test_one_request_per_prompt_in_order(self) -> None:
        client = FakeHttpClient()
        sampler = RemoteVllmSampler(client, "m")
        responses = await sampler.sample([[1], [2], [3]], 1, SamplingParams(max_tokens=8))
        assert len(responses) == 3
        assert [payload["prompt"] for _, payload in client.posts] == [[1], [2], [3]]

    async def test_reads_cache_hits_off_usage(self) -> None:
        sampler = RemoteVllmSampler(FakeHttpClient(), "m")
        (response,) = await sampler.sample([[1]], 1, SamplingParams(max_tokens=8))
        assert response.prompt_cache_hit_tokens == 3

    async def test_server_error_is_loud(self) -> None:
        client = FakeHttpClient({"/v1/completions": FakeResponse(status_code=500, text="boom")})
        sampler = RemoteVllmSampler(client, "m")
        with pytest.raises(RuntimeError, match="returned 500"):
            await sampler.sample([[1]], 1, SamplingParams(max_tokens=8))

    async def test_empty_prompts_rejected(self) -> None:
        sampler = RemoteVllmSampler(FakeHttpClient(), "m")
        with pytest.raises(ValueError, match="at least one prompt"):
            await sampler.sample([], 1, SamplingParams(max_tokens=8))

    async def test_sync_loads_the_adapter(self) -> None:
        client = FakeHttpClient()
        sampler = RemoteVllmSampler(client, "m")
        await sampler.sync_weights("/shared/adapters/v000001", 1)
        path, payload = client.posts[0]
        assert path == "/v1/load_lora_adapter"
        assert payload["lora_path"] == "/shared/adapters/v000001"

    async def test_second_sync_unloads_first(self) -> None:
        # Re-registering a live name is rejected by the server.
        client = FakeHttpClient()
        sampler = RemoteVllmSampler(client, "m")
        await sampler.sync_weights("/a", 1)
        await sampler.sync_weights("/b", 2)
        assert [path for path, _ in client.posts] == [
            "/v1/load_lora_adapter",
            "/v1/unload_lora_adapter",
            "/v1/load_lora_adapter",
        ]

    async def test_requests_route_to_the_adapter_once_synced(self) -> None:
        client = FakeHttpClient()
        sampler = RemoteVllmSampler(client, "Qwen/Qwen3-8B", lora_name="policy")
        await sampler.sample([[1]], 1, SamplingParams(max_tokens=8))
        assert client.posts[-1][1]["model"] == "Qwen/Qwen3-8B"
        await sampler.sync_weights("/a", 1)
        await sampler.sample([[1]], 1, SamplingParams(max_tokens=8))
        assert client.posts[-1][1]["model"] == "policy"

    async def test_a_refused_adapter_explains_the_usual_causes(self) -> None:
        client = FakeHttpClient(
            {"/v1/load_lora_adapter": FakeResponse(status_code=400, text="nope")}
        )
        sampler = RemoteVllmSampler(client, "m")
        with pytest.raises(RuntimeError, match="VLLM_ALLOW_RUNTIME_LORA_UPDATING"):
            await sampler.sync_weights("/a", 1)


class TestBuildSampler:
    def test_unknown_mode_lists_the_valid_ones(self) -> None:
        with pytest.raises(ValueError, match=r"sampler\.mode must be one of"):
            build_sampler("inprocess", "m")

    def test_remote_without_a_base_url_says_so(self) -> None:
        with pytest.raises(ValueError, match=r"sampler\.base_url"):
            build_sampler("remote", "m")

    def test_modes_are_the_documented_pair(self) -> None:
        assert SAMPLER_MODES == ("colocated", "remote")
