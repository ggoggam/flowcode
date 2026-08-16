"""The local policy backend, against a tiny randomly-initialised model.

No network and no download: :func:`tiny_backend` builds a two-layer Llama from a config,
so every test here runs on CPU in milliseconds. What matters is not that the model is any
good but that the *gradient plumbing* is exact, and that is checkable at any scale.

The load-bearing test is :class:`TestGradientEquivalence`. ``compute_logprobs`` runs
under ``no_grad`` and ``apply_gradient`` forwards a second time, so nothing in the code
path structurally guarantees the two passes see the same function — a dropout left on, an
eval/train mismatch, a padding bug that only bites at one batch shape, and the gradient
silently becomes something other than the objective's. Comparing against a single-pass
autograd reference is the only way to catch that.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from flowcode.config import ModelConfig
from flowcode.local_backend import (
    LocalBackend,
    lora_target_modules,
    resolve_dtype,
)
from flowcode.types import SampledSequence, SampleResponse, SamplingParams, Segment, Trajectory

pytest.importorskip("peft", reason="LocalBackend needs the `local` extra")
pytest.importorskip("transformers", reason="LocalBackend needs the `local` extra")

MODEL_CFG = ModelConfig(name="tiny-llama-test", renderer="qwen3", lora_rank=4)
VOCAB = 64


def make_trajectory(
    task_id: str = "t0",
    prompt: list[int] | None = None,
    completion: list[int] | None = None,
) -> Trajectory:
    prompt = [5, 6, 7, 8] if prompt is None else prompt
    completion = [9, 10, 11] if completion is None else completion
    return Trajectory(
        task_id=task_id,
        prompt_tokens=prompt,
        completion_tokens=completion,
        sampling_logprobs=[-0.5] * len(completion),
        log_reward=-1.0,
        segments=[Segment(0, len(completion))],
    )


def tiny_backend(micro_batch_size: int = 8, **kwargs: object) -> LocalBackend:
    """A LocalBackend over a two-layer random Llama. No download, no accelerator."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.manual_seed(0)
    config = AutoConfig.for_model(
        "llama",
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
    )
    model = AutoModelForCausalLM.from_config(config)
    peft_model = get_peft_model(
        model,
        LoraConfig(
            r=MODEL_CFG.lora_rank,
            lora_alpha=8,
            lora_dropout=0.0,
            target_modules=lora_target_modules(MODEL_CFG),
            task_type="CAUSAL_LM",
            bias="none",
        ),
    )
    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)
    return LocalBackend(
        peft_model,
        tokenizer=None,
        optimizer=optimizer,
        model_cfg=MODEL_CFG,
        micro_batch_size=micro_batch_size,
        **kwargs,  # ty: ignore[invalid-argument-type]
    )


def trainable_grads(backend: LocalBackend) -> list[torch.Tensor]:
    return [
        p.grad.detach().clone() if p.grad is not None else torch.zeros(())
        for p in backend.parameters()
    ]


class FakeSampler:
    """Records sync calls and returns canned completions."""

    def __init__(self) -> None:
        self.synced: list[tuple[str, int]] = []
        self.sample_calls = 0

    async def sample(
        self, prompts: object, num_samples: int, sampling_params: SamplingParams
    ) -> list[SampleResponse]:
        self.sample_calls += 1
        return [
            SampleResponse(
                sequences=[
                    SampledSequence(tokens=[1, 2], logprobs=[-0.1, -0.2], stop_reason="stop")
                    for _ in range(num_samples)
                ],
                prompt_cache_hit_tokens=4,
            )
            for _ in prompts  # ty: ignore[not-iterable]
        ]

    async def sync_weights(self, adapter_path: str, version: int) -> None:
        self.synced.append((adapter_path, version))


# ------------------------------------------------------------------------ config helpers


class TestResolveDtype:
    def test_known_names(self) -> None:
        assert resolve_dtype("bfloat16") is torch.bfloat16
        assert resolve_dtype("float32") is torch.float32

    def test_unknown_name_is_loud(self) -> None:
        # Defaulting to float32 here would quietly cost 4x the memory on an 8B model.
        with pytest.raises(ValueError, match="dtype must be one of"):
            resolve_dtype("bf16")


class TestLoraTargetModules:
    def test_all_families(self) -> None:
        targets = lora_target_modules(MODEL_CFG)
        assert "q_proj" in targets
        assert "gate_proj" in targets
        assert "lm_head" in targets

    def test_attention_only(self) -> None:
        cfg = ModelConfig(name="m", renderer="qwen3", train_mlp=False, train_unembed=False)
        assert lora_target_modules(cfg) == ["q_proj", "k_proj", "v_proj", "o_proj"]

    def test_no_families_is_rejected(self) -> None:
        cfg = ModelConfig(
            name="m", renderer="qwen3", train_mlp=False, train_attn=False, train_unembed=False
        )
        with pytest.raises(ValueError, match="no LoRA parameters to train"):
            lora_target_modules(cfg)


# --------------------------------------------------------------------- compute_logprobs


class TestComputeLogprobs:
    async def test_shapes_match_completions(self) -> None:
        backend = tiny_backend()
        trajectories = [
            make_trajectory("a", completion=[9, 10, 11]),
            make_trajectory("b", prompt=[1, 2], completion=[3, 4, 5, 6, 7]),
        ]
        logprobs = await backend.compute_logprobs(trajectories)
        assert [lp.shape[0] for lp in logprobs] == [3, 5]

    async def test_returns_detached_leaves(self) -> None:
        # The objective calls .backward() on a loss built from these and then reads .grad,
        # so they must be leaves. A graph-connected tensor would backprop into the model
        # early and then be counted a second time by apply_gradient.
        backend = tiny_backend()
        (logprobs,) = await backend.compute_logprobs([make_trajectory()])
        assert logprobs.requires_grad
        assert logprobs.is_leaf
        assert logprobs.grad_fn is None

    async def test_values_are_real_logprobs(self) -> None:
        backend = tiny_backend()
        (logprobs,) = await backend.compute_logprobs([make_trajectory()])
        assert torch.all(logprobs < 0.0)
        # A uniform distribution over the vocabulary is the floor for a random init.
        assert torch.all(logprobs > torch.log(torch.tensor(1.0 / VOCAB)) - 2.0)

    async def test_empty_batch_rejected(self) -> None:
        backend = tiny_backend()
        with pytest.raises(ValueError, match="at least one trajectory"):
            await backend.compute_logprobs([])

    async def test_padding_does_not_change_a_trajectorys_logprobs(self) -> None:
        # Ragged batches are right-padded and masked. If the mask were wrong, a short
        # trajectory batched next to a long one would score differently than alone.
        backend = tiny_backend()
        short = make_trajectory("a", prompt=[1, 2], completion=[3, 4])
        long = make_trajectory("b", prompt=[5, 6, 7, 8, 9], completion=[10, 11, 12, 13, 14, 15])
        (alone,) = await backend.compute_logprobs([short])
        together, _ = await backend.compute_logprobs([short, long])
        assert torch.allclose(alone, together, atol=1e-5)

    async def test_leaves_the_model_in_its_prior_mode(self) -> None:
        backend = tiny_backend()
        backend._model.train()
        await backend.compute_logprobs([make_trajectory()])
        assert backend._model.training


# ------------------------------------------------------------------ THE GRADIENT


class TestGradientEquivalence:
    """apply_gradient's two-pass recompute must equal single-pass autograd, exactly."""

    @staticmethod
    def reference_grads(
        backend: LocalBackend, trajectories: list[Trajectory], cotangents: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """dC/dtheta computed in one pass, with C = sum(logprobs * cotangent)."""
        backend._optimizer.zero_grad(set_to_none=True)
        was_training = backend._model.training
        backend._model.eval()
        per_trajectory = backend._forward_logprobs(trajectories)
        loss = torch.stack(
            [(lp * g).sum() for lp, g in zip(per_trajectory, cotangents, strict=True)]
        ).sum()
        loss.backward()
        backend._model.train(was_training)
        grads = trainable_grads(backend)
        backend._optimizer.zero_grad(set_to_none=True)
        return grads

    async def test_matches_single_pass_autograd(self) -> None:
        backend = tiny_backend()
        trajectories = [
            make_trajectory("a", completion=[9, 10, 11]),
            make_trajectory("b", prompt=[1, 2, 3], completion=[4, 5, 6, 7]),
        ]
        cotangents = [
            torch.tensor([0.5, -1.5, 2.0]),
            torch.tensor([-0.25, 0.75, 1.0, -2.0]),
        ]
        expected = self.reference_grads(backend, trajectories, cotangents)

        await backend.apply_gradient(trajectories, cotangents)
        actual = trainable_grads(backend)

        assert len(actual) == len(expected)
        for got, want in zip(actual, expected, strict=True):
            assert torch.allclose(got, want, atol=1e-6), (got - want).abs().max()

    async def test_micro_batching_changes_nothing(self) -> None:
        # The batch is split into forward passes of micro_batch_size. If padding or
        # masking depended on the chunk's shape, this would diverge.
        trajectories = [
            make_trajectory("a", prompt=[1, 2], completion=[3, 4]),
            make_trajectory("b", prompt=[5, 6, 7, 8], completion=[9, 10, 11, 12, 13]),
            make_trajectory("c", prompt=[14, 15, 16], completion=[17]),
        ]
        cotangents = [
            torch.tensor([1.0, -1.0]),
            torch.tensor([0.5, 0.25, -0.5, 2.0, -1.0]),
            torch.tensor([3.0]),
        ]

        whole = tiny_backend(micro_batch_size=8)
        await whole.apply_gradient(trajectories, cotangents)
        one_at_a_time = tiny_backend(micro_batch_size=1)
        await one_at_a_time.apply_gradient(trajectories, cotangents)

        for got, want in zip(trainable_grads(one_at_a_time), trainable_grads(whole), strict=True):
            assert torch.allclose(got, want, atol=1e-6)

    async def test_grad_scale_scales_the_gradient(self) -> None:
        trajectories = [make_trajectory()]
        cotangents = [torch.tensor([1.0, 1.0, 1.0])]

        unscaled = tiny_backend()
        await unscaled.apply_gradient(trajectories, cotangents)
        halved = tiny_backend()
        await halved.apply_gradient(trajectories, cotangents, grad_scale=0.5)

        for full, half in zip(trainable_grads(unscaled), trainable_grads(halved), strict=True):
            assert torch.allclose(half, full * 0.5, atol=1e-6)

    async def test_gradients_accumulate_until_optim_step(self) -> None:
        # Trainer pushes then steps; two pushes without a step must sum.
        trajectories = [make_trajectory()]
        cotangents = [torch.tensor([1.0, 1.0, 1.0])]

        once = tiny_backend()
        await once.apply_gradient(trajectories, cotangents)
        single = trainable_grads(once)

        twice = tiny_backend()
        await twice.apply_gradient(trajectories, cotangents)
        await twice.apply_gradient(trajectories, cotangents)

        for doubled, one in zip(trainable_grads(twice), single, strict=True):
            assert torch.allclose(doubled, one * 2.0, atol=1e-6)

    async def test_sign_is_not_flipped(self) -> None:
        # The Tinker bridge negates because that server's cross-entropy is
        # sum(-logprobs * weights). Locally there is no such convention to cancel, and a
        # stray negation would train the policy to maximise exactly what it should minimise.
        backend = tiny_backend()
        trajectory = make_trajectory()
        (logprobs,) = await backend.compute_logprobs([trajectory])
        before = logprobs.detach().clone()

        # Ascend the logprob of every completion token: dC/dlogprobs = +1 with a
        # gradient *descent* step means C = -sum(logprobs) should fall, i.e. logprobs rise.
        await backend.apply_gradient([trajectory], [torch.full((3,), -1.0)])
        await backend.optim_step(lr=0.1)

        (after,) = await backend.compute_logprobs([trajectory])
        assert after.sum() > before.sum()


class TestApplyGradientValidation:
    async def test_empty_batch_rejected(self) -> None:
        backend = tiny_backend()
        with pytest.raises(ValueError, match="at least one trajectory"):
            await backend.apply_gradient([], [])

    async def test_misaligned_lengths_rejected(self) -> None:
        backend = tiny_backend()
        with pytest.raises(ValueError, match="align index-for-index"):
            await backend.apply_gradient([make_trajectory()], [])

    async def test_wrong_shaped_gradient_rejected(self) -> None:
        backend = tiny_backend()
        with pytest.raises(ValueError, match="completion-only"):
            await backend.apply_gradient([make_trajectory()], [torch.zeros(99)])

    async def test_defer_is_accepted_and_ignored(self) -> None:
        backend = tiny_backend()
        metrics = await backend.apply_gradient([make_trajectory()], [torch.ones(3)], defer=True)
        assert metrics["num_trajectories"] == 1.0


# ------------------------------------------------------------------------- optim_step


class TestOptimStep:
    async def test_changes_the_trainable_parameters(self) -> None:
        backend = tiny_backend()
        before = [p.detach().clone() for p in backend.parameters()]
        await backend.apply_gradient([make_trajectory()], [torch.ones(3)])
        await backend.optim_step(lr=1e-2)
        after = list(backend.parameters())
        assert any(not torch.allclose(a, b) for a, b in zip(after, before, strict=True))

    async def test_clears_gradients(self) -> None:
        backend = tiny_backend()
        await backend.apply_gradient([make_trajectory()], [torch.ones(3)])
        await backend.optim_step(lr=1e-2)
        assert all(p.grad is None for p in backend.parameters())

    async def test_writes_the_scheduled_learning_rate(self) -> None:
        # The schedule in flowcode.train owns the LR; a torch scheduler would duplicate it.
        backend = tiny_backend()
        await backend.apply_gradient([make_trajectory()], [torch.ones(3)])
        metrics = await backend.optim_step(lr=3e-4)
        assert metrics["lr"] == pytest.approx(3e-4)
        assert backend._optimizer.param_groups[0]["lr"] == pytest.approx(3e-4)

    async def test_reports_the_clipped_norm(self) -> None:
        backend = tiny_backend()
        await backend.apply_gradient([make_trajectory()], [torch.full((3,), 100.0)])
        metrics = await backend.optim_step(lr=1e-3, grad_clip_norm=1.0)
        assert metrics["grad_norm"] > 1.0  # the pre-clip norm is what gets reported

    async def test_non_positive_lr_rejected(self) -> None:
        backend = tiny_backend()
        with pytest.raises(ValueError, match="positive learning rate"):
            await backend.optim_step(lr=0.0)


# ----------------------------------------------------------------- sampling and sync


class TestSampling:
    async def test_delegates_to_the_sampler(self) -> None:
        sampler = FakeSampler()
        backend = tiny_backend(sampler=sampler)
        responses = await backend.sample([[1, 2, 3]], 2, SamplingParams(max_tokens=8))
        assert sampler.sample_calls == 1
        assert len(responses) == 1
        assert len(responses[0].sequences) == 2

    async def test_counts_prompt_and_generated_tokens(self) -> None:
        backend = tiny_backend(sampler=FakeSampler())
        await backend.sample([[1, 2, 3]], 2, SamplingParams(max_tokens=8))
        usage = backend.token_usage()
        assert usage.sample_tokens == 2 * 3 + 2 * 2
        assert usage.prompt_cache_hit_tokens == 4

    async def test_without_a_sampler_it_says_so(self) -> None:
        backend = tiny_backend()
        with pytest.raises(RuntimeError, match="without a sampler"):
            await backend.sample([[1]], 1, SamplingParams(max_tokens=8))

    async def test_sync_writes_an_adapter_and_bumps_the_version(self, tmp_path: Path) -> None:
        sampler = FakeSampler()
        backend = tiny_backend(sampler=sampler, adapter_dir=str(tmp_path))
        assert backend.policy_version == 0
        await backend.sync_sampler()
        await backend.sync_sampler()
        assert backend.policy_version == 2
        assert [version for _, version in sampler.synced] == [1, 2]
        assert all(Path(path).is_dir() for path, _ in sampler.synced)

    async def test_sync_without_a_sampler_says_so(self) -> None:
        backend = tiny_backend()
        with pytest.raises(RuntimeError, match="nothing to sync"):
            await backend.sync_sampler()


class TestTokenUsage:
    async def test_starts_empty(self) -> None:
        backend = tiny_backend()
        usage = backend.token_usage()
        assert usage.train_tokens == 0
        assert usage.num_forward_passes == 0

    async def test_forward_and_backward_each_cost_the_input_length(self) -> None:
        # This is the passes=2 the cost estimator prices off-policy runs at.
        backend = tiny_backend()
        trajectory = make_trajectory()
        (logprobs,) = await backend.compute_logprobs([trajectory])
        await backend.apply_gradient([trajectory], [torch.ones_like(logprobs)])
        usage = backend.token_usage()
        # prompt 4 + completion 3 - 1 = 6 input positions, counted once per pass
        assert usage.train_tokens == 12
        assert usage.num_forward_passes == usage.num_backward_passes == 1

    async def test_optim_steps_are_counted(self) -> None:
        backend = tiny_backend()
        await backend.apply_gradient([make_trajectory()], [torch.ones(3)])
        await backend.optim_step(lr=1e-3)
        assert backend.token_usage().num_optim_steps == 1
