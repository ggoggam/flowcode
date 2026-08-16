"""Do the sampler and the trainer agree about what the policy is?

**Run this before any real training run on new hardware.** It is marked ``hardware`` and
deselected by default because it needs a real accelerator and a real vLLM, so it will
otherwise never run — and it is the check that catches the worst failure mode in the whole
local stack.

What breaks when they disagree
------------------------------
``Trajectory.sampling_logprobs`` comes from the engine; ``LocalBackend.compute_logprobs``
recomputes the same tokens through the training model. Two independent implementations of
the same forward pass, with different kernels, different attention implementations,
possibly different dtypes. When they drift:

* ``train.on_policy_only=true`` substitutes the sampler's logprobs for the current
  policy's and skips the oracle pass entirely. Every balance residual is then computed
  against a ``log P_F`` the trainer never produced. Nothing raises.
* :func:`flowcode.objectives.base.importance_weights` divides one by the other. A constant
  offset becomes a constant reweighting; a token-dependent offset becomes noise that looks
  like exploration.
* The GFlowNet target ``p(x) ∝ R(x)`` is defined in terms of the policy's own
  probabilities. If the trainer is fitting a different function than the sampler draws
  from, the thing that converges is not the thing that was asked for.

None of this shows up as an error. It shows up as a run that trains, logs plausible
metrics, and does not reproduce.

What "agree" means
------------------
Not bit-identical — that is not achievable across two implementations. The tolerances
below are set so that ordinary kernel and reduction-order differences pass while anything
structural (an off-by-one in the alignment contract, a chat-template mismatch, a dtype
collapse in the log-softmax normaliser) fails. An off-by-one is the one to worry about:
it produces logprobs that are individually plausible and completely wrong.

Setup
-----
Needs the ``vllm`` extra and an accelerator::

    mise run sync:vllm
    uv run pytest tests/test_logprob_agreement.py -m hardware

``FLOWCODE_TEST_MODEL`` overrides the model; the default is small enough to run anywhere
with a device and big enough to be a real transformer.
"""

from __future__ import annotations

import os

import pytest
import torch

from flowcode.config import ModelConfig
from flowcode.types import SamplingParams, Segment, Trajectory

pytestmark = pytest.mark.hardware

MODEL_NAME = os.environ.get("FLOWCODE_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

MEAN_ABS_TOLERANCE = 0.02
"""Average per-token disagreement, in nats. Kernel and reduction-order differences live
well inside this; an alignment bug does not."""

MAX_ABS_TOLERANCE = 0.25
"""Worst single token. Looser than the mean because one unlucky position near a numerical
cliff is not evidence of anything, but a systematic shift would fail the mean first."""

SEQUENCE_TOLERANCE = 0.5
"""Whole-sequence sum. This is the quantity the objectives actually use — trajectory
balance sums logprobs over the completion — so it gets its own bound rather than being
left implied by the per-token one."""


@pytest.fixture(scope="module")
def model_cfg() -> ModelConfig:
    # LoRA on the unembedding is disabled: vLLM's LoRA support does not reliably cover
    # lm_head, and an adapter it refuses to load makes this test fail for a reason that
    # has nothing to do with logprob agreement.
    return ModelConfig(name=MODEL_NAME, renderer="qwen3", lora_rank=8, train_unembed=False)


@pytest.fixture(scope="module")
def backend(model_cfg: ModelConfig):  # noqa: ANN201 - LocalBackend, imported lazily
    from flowcode.local_backend import LocalBackend

    return LocalBackend.create(model_cfg, micro_batch_size=4, dtype="bfloat16")


@pytest.fixture(scope="module")
def sampler(model_cfg: ModelConfig):  # noqa: ANN201 - ColocatedVllmSampler, lazily
    from flowcode.samplers.vllm import ColocatedVllmSampler

    return ColocatedVllmSampler.create(
        model_cfg.name,
        max_lora_rank=model_cfg.lora_rank,
        gpu_memory_utilization=0.35,
        max_model_len=1024,
        dtype="bfloat16",
    )


PROMPTS = [
    "Write a Python function that reverses a string.",
    "Write a Python function that returns the nth Fibonacci number.",
]


@pytest.fixture(scope="module")
async def sampled(backend, sampler) -> list[Trajectory]:  # noqa: ANN001
    """Sample real completions and package them as trajectories."""
    tokenizer = backend.tokenizer
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], add_generation_prompt=True, tokenize=True
        )
        for p in PROMPTS
    ]
    responses = await sampler.sample(prompts, 2, SamplingParams(max_tokens=64, temperature=1.0))

    trajectories: list[Trajectory] = []
    for prompt, response in zip(prompts, responses, strict=True):
        for sequence in response.sequences:
            if not sequence.tokens:
                continue
            trajectories.append(
                Trajectory(
                    task_id="agreement",
                    prompt_tokens=list(prompt),
                    completion_tokens=list(sequence.tokens),
                    sampling_logprobs=list(sequence.logprobs),
                    log_reward=0.0,
                    segments=[Segment(0, len(sequence.tokens))],
                )
            )
    assert trajectories, "the engine returned nothing usable; fix that before reading on"
    return trajectories


class TestLogprobAgreement:
    async def test_per_token_logprobs_agree(self, backend, sampled) -> None:  # noqa: ANN001
        recomputed = await backend.compute_logprobs(sampled)
        for trajectory, ours in zip(sampled, recomputed, strict=True):
            theirs = torch.tensor(trajectory.sampling_logprobs, dtype=torch.float32)
            diff = (ours.detach().float().cpu() - theirs).abs()
            assert diff.mean().item() < MEAN_ABS_TOLERANCE, (
                f"mean per-token disagreement {diff.mean().item():.4f} nats. This is the "
                "signature of an alignment or template mismatch, not of kernel noise — "
                "check flowcode.alignment against how the engine was prompted."
            )
            assert diff.max().item() < MAX_ABS_TOLERANCE, (
                f"worst-token disagreement {diff.max().item():.4f} nats at position "
                f"{int(diff.argmax().item())}"
            )

    async def test_sequence_sums_agree(self, backend, sampled) -> None:  # noqa: ANN001
        # The quantity trajectory balance actually consumes.
        recomputed = await backend.compute_logprobs(sampled)
        for trajectory, ours in zip(sampled, recomputed, strict=True):
            theirs = sum(trajectory.sampling_logprobs)
            assert abs(ours.detach().float().sum().item() - theirs) < SEQUENCE_TOLERANCE

    async def test_the_disagreement_is_not_a_constant_shift(self, backend, sampled) -> None:  # noqa: ANN001
        # A constant offset would pass a loose per-token check while silently rescaling
        # every importance weight by the same factor. It is also exactly what an
        # off-by-one in the prompt boundary does not produce, so this separates the two
        # failure modes rather than lumping them together.
        recomputed = await backend.compute_logprobs(sampled)
        for trajectory, ours in zip(sampled, recomputed, strict=True):
            theirs = torch.tensor(trajectory.sampling_logprobs, dtype=torch.float32)
            residual = ours.detach().float().cpu() - theirs
            assert abs(residual.mean().item()) < MEAN_ABS_TOLERANCE

    async def test_shifting_by_one_token_would_fail_this_test(self, backend, sampled) -> None:  # noqa: ANN001
        # Calibration. If the tolerances above are loose enough to admit an off-by-one,
        # every other assertion in this file is decoration.
        recomputed = await backend.compute_logprobs(sampled)
        checked = 0
        for trajectory, ours in zip(sampled, recomputed, strict=True):
            if len(trajectory.sampling_logprobs) < 8:
                continue
            theirs = torch.tensor(trajectory.sampling_logprobs, dtype=torch.float32)
            shifted = (ours.detach().float().cpu()[1:] - theirs[:-1]).abs()
            assert shifted.mean().item() > MEAN_ABS_TOLERANCE
            checked += 1
        assert checked, "no completion long enough to calibrate against"
