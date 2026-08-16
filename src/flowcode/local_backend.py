"""The policy in this process: a local LoRA model standing in for the hosted one.

:class:`LocalBackend` satisfies the same structural contract as
:class:`~flowcode.tinker_backend.TinkerBackend` — ``sample`` / ``compute_logprobs`` /
``apply_gradient`` / ``optim_step`` / ``sync_sampler`` / ``token_usage`` — so
:class:`~flowcode.train.Trainer` drives either without knowing which it has. Nothing under
``objectives/`` changes: they consume per-token logprobs and hand back
``dC/dlogprobs``, and where those logprobs came from was never their business.

Why ``accelerate`` and not a training framework
-----------------------------------------------
The loop already exists (:meth:`flowcode.train.Trainer.step`) and the backward is
*cotangent-driven*: the objective differentiates a scalar it built out of logprobs, and
what reaches this module is ``dC/dlogprobs``, not a loss. A framework whose contract is
"implement ``training_step``, return a loss" has to be fought to accept that. What is
actually wanted from the stack is device placement, precision and distribution, which is
exactly ``Accelerator``'s remit and nothing more. Lightning Fabric would fit identically.

Two passes, not one retained graph
----------------------------------
:meth:`compute_logprobs` runs under ``torch.no_grad()`` and returns detached leaves;
:meth:`apply_gradient` runs the forward *again*, with grad, and backpropagates the
cotangent. The obvious alternative — keep the first pass's graph alive and reuse it — is
rejected on memory: the objective's loss couples every trajectory in the batch (VarGrad's
group baseline, TB's shared ``log Z``), so all logprobs must exist before any backward
can start, and retaining graphs for a whole 85-trajectory batch means holding every
activation of an 8B model at once. Recomputing costs one extra forward, roughly a third
more compute, and turns peak memory from a function of ``batch`` into a function of
``micro_batch_size``.

This is also precisely the shape the hosted backend already had — a ``forward`` oracle
pass followed by a ``forward_backward`` push — so the accounting in
:class:`~flowcode.types.TokenUsage` ("an off-policy step costs twice an on-policy one")
stays true, and ``train.on_policy_only`` still skips the oracle pass and saves the same
third.

The surrogate scalar
--------------------
``torch.autograd`` wants a scalar. Given the objective's ``g = dC/dlogprobs``, this module
backpropagates::

    S = sum(logprobs * g.detach())

whose derivative with respect to ``logprobs`` is exactly ``g``, so the gradient deposited
in the LoRA parameters is the one the objective asked for. Note there is no sign flip
here: the negation in :mod:`flowcode.tinker_backend` exists only to cancel the sign in
that server's cross-entropy convention, and locally there is no such convention to cancel.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import torch

from flowcode.alignment import (
    build_input_tokens,
    build_target_tokens,
    slice_completion_values,
)
from flowcode.config import ModelConfig
from flowcode.types import SampleResponse, SamplingParams, TokenUsage, Trajectory, TrajectoryBatch

if TYPE_CHECKING:  # pragma: no cover - import-time only
    from accelerate import Accelerator

__all__ = [
    "ATTN_MODULES",
    "MLP_MODULES",
    "UNEMBED_MODULES",
    "LocalBackend",
    "WeightSyncSampler",
    "lora_target_modules",
    "resolve_dtype",
]

logger = logging.getLogger(__name__)

ATTN_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
"""Attention projections to attach LoRAs to, in the Llama/Qwen/Mistral naming convention."""

MLP_MODULES = ("gate_proj", "up_proj", "down_proj")
"""MLP projections, same convention. MoE architectures name their experts differently and
need ``lora_target_modules`` overridden."""

UNEMBED_MODULES = ("lm_head",)
"""The unembedding. Worth knowing before enabling: vLLM's LoRA support does not reliably
cover ``lm_head``, so a sampler may refuse an adapter that includes it."""


class WeightSyncSampler(Protocol):
    """A sampler that can be handed fresh policy weights mid-run.

    Split out from :class:`~flowcode.rollout.SamplerLike` because a rollout only ever
    samples, while a *training* backend additionally has to push its updated adapter over.
    """

    async def sample(
        self,
        prompts: Sequence[Sequence[int]],
        num_samples: int,
        sampling_params: SamplingParams,
    ) -> list[SampleResponse]:
        """See :meth:`flowcode.rollout.SamplerLike.sample`."""
        ...

    async def sync_weights(self, adapter_path: str, version: int) -> None:
        """Adopt the LoRA adapter at ``adapter_path`` for subsequent requests.

        Args:
            adapter_path: Directory a PEFT adapter was just saved to.
            version: Monotonic counter identifying this policy revision. Samplers use it
                to key their adapter registry and to stamp trajectories for staleness
                accounting.
        """
        ...


def resolve_dtype(name: str) -> torch.dtype:
    """Map a config string to a torch dtype.

    Args:
        name: One of ``bfloat16`` / ``float16`` / ``float32``.

    Returns:
        The corresponding dtype.

    Raises:
        ValueError: On an unknown name, rather than silently defaulting to float32 and
            quietly costing four times the memory.
    """
    dtypes = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if name not in dtypes:
        raise ValueError(f"dtype must be one of {sorted(dtypes)}, got {name!r}")
    return dtypes[name]


def lora_target_modules(model_cfg: ModelConfig) -> list[str]:
    """Translate the ``train_*`` flags into PEFT ``target_modules``.

    Args:
        model_cfg: The model group, whose ``train_mlp`` / ``train_attn`` /
            ``train_unembed`` flags select module families.

    Returns:
        Module name suffixes for :class:`peft.LoraConfig`.

    Raises:
        ValueError: If every family is disabled, which would build a model with no
            trainable parameters and train silently forever.
    """
    targets: list[str] = []
    if model_cfg.train_attn:
        targets.extend(ATTN_MODULES)
    if model_cfg.train_mlp:
        targets.extend(MLP_MODULES)
    if model_cfg.train_unembed:
        targets.extend(UNEMBED_MODULES)
    if not targets:
        raise ValueError(
            "At least one of model.train_mlp / model.train_attn / model.train_unembed "
            "must be true; with all three false there are no LoRA parameters to train."
        )
    return targets


class LocalBackend:
    """A LoRA policy held in this process, driven through the :class:`BackendLike` contract.

    Construct with :meth:`create` in production. The plain constructor takes the already
    built pieces, which is how the offline tests drive it with a tiny random model.

    Attributes:
        model_cfg: The model group this backend was built from.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        optimizer: torch.optim.Optimizer,
        model_cfg: ModelConfig,
        *,
        accelerator: Accelerator | None = None,
        sampler: WeightSyncSampler | None = None,
        micro_batch_size: int = 8,
        adapter_dir: str = "adapters",
    ) -> None:
        """Wrap an already-constructed model, tokenizer and optimiser.

        Args:
            model: A PEFT-wrapped causal LM. Only its LoRA parameters should require grad.
            tokenizer: The model's tokenizer, exposed for prompt rendering.
            optimizer: Optimiser over the trainable parameters.
            model_cfg: The model group, kept so :mod:`flowcode.cost` can price usage.
            accelerator: An ``Accelerator`` to route ``backward`` through. ``None`` uses
                plain autograd, which is what the CPU tests want.
            sampler: Where :meth:`sync_sampler` pushes updated weights. ``None`` makes
                both :meth:`sample` and :meth:`sync_sampler` raise.
            micro_batch_size: Trajectories per forward pass. This, not the batch size,
                is what peak activation memory scales with.
            adapter_dir: Directory adapters are written to on each sampler sync.

        Raises:
            ValueError: If ``micro_batch_size`` is not positive.
        """
        if micro_batch_size <= 0:
            raise ValueError(f"micro_batch_size must be positive, got {micro_batch_size}")
        self._model = model
        self._tokenizer = tokenizer
        self._optimizer = optimizer
        self.model_cfg = model_cfg
        self._accelerator = accelerator
        self._sampler = sampler
        self._micro_batch_size = micro_batch_size
        self._adapter_dir = Path(adapter_dir)
        self._usage = TokenUsage()
        self._policy_version = 0

    @property
    def tokenizer(self) -> Any:
        """The model's tokenizer, for rendering prompts and decoding completions."""
        return self._tokenizer

    @property
    def policy_version(self) -> int:
        """How many times the sampler has been synced. Stamped onto trajectories."""
        return self._policy_version

    # ------------------------------------------------------------------------ sampling

    async def sample(
        self,
        prompts: Sequence[Sequence[int]],
        num_samples: int,
        sampling_params: SamplingParams,
    ) -> list[SampleResponse]:
        """Delegate to the configured sampler and account for the tokens it burned.

        Args:
            prompts: One token-id list per prompt. Order is preserved in the result.
            num_samples: Completions per prompt — the GFlowNet group size.
            sampling_params: Neutral sampling parameters.

        Returns:
            One :class:`~flowcode.types.SampleResponse` per prompt, in the same order.

        Raises:
            RuntimeError: If no sampler was configured.
            ValueError: If ``prompts`` is empty or ``num_samples`` is not positive.
        """
        if self._sampler is None:
            raise RuntimeError(
                "LocalBackend was built without a sampler, so it cannot generate. Pass "
                "one to LocalBackend.create(), or drive compute_logprobs directly."
            )
        if not prompts:
            raise ValueError("sample() needs at least one prompt")
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")

        responses = await self._sampler.sample(prompts, num_samples, sampling_params)

        gross_tokens = 0
        cache_hits = 0
        for prompt, response in zip(prompts, responses, strict=True):
            gross_tokens += num_samples * len(prompt)
            gross_tokens += sum(len(seq.tokens) for seq in response.sequences)
            cache_hits += response.prompt_cache_hit_tokens
        self._usage += TokenUsage(sample_tokens=gross_tokens, prompt_cache_hit_tokens=cache_hits)
        return responses

    async def sync_sampler(self) -> None:
        """Save the current adapter and hand it to the sampler.

        In-flight requests finishing on the previous adapter are not waited for. That is
        staleness, not error: the balance conditions the objectives regress on hold for
        trajectories from any behaviour policy (see
        :mod:`flowcode.objectives.base`), which is exactly what makes a decoupled sampler
        affordable here and would not be for PPO.

        Raises:
            RuntimeError: If no sampler was configured.
        """
        if self._sampler is None:
            raise RuntimeError("LocalBackend was built without a sampler; nothing to sync")
        self._policy_version += 1
        path = self._adapter_dir / f"v{self._policy_version:06d}"
        path.parent.mkdir(parents=True, exist_ok=True)
        self._unwrapped().save_pretrained(str(path))
        await self._sampler.sync_weights(str(path), self._policy_version)
        logger.info("synced sampler to policy version %d (%s)", self._policy_version, path)

    # ------------------------------------------------------------------ the oracle pass

    async def compute_logprobs(self, trajectories: TrajectoryBatch) -> list[torch.Tensor]:
        """Score each trajectory's completion tokens under the *current* policy weights.

        Args:
            trajectories: The batch to score. Must be non-empty.

        Returns:
            One tensor per trajectory, shape ``(len(completion_tokens),)``, prompt padding
            already sliced off, ``requires_grad=True``. These are autograd *leaves* with
            no connection to the model: the objective builds its loss out of them, calls
            ``.backward()``, and hands ``tensor.grad`` back to :meth:`apply_gradient`,
            which is where the chain rule is rejoined.

        Raises:
            ValueError: If the batch is empty.
        """
        if not trajectories:
            raise ValueError("compute_logprobs() needs at least one trajectory")

        was_training = self._model.training
        self._model.eval()
        try:
            with torch.no_grad():
                per_trajectory = self._forward_logprobs(trajectories)
        finally:
            self._model.train(was_training)

        self._usage += TokenUsage(
            train_tokens=sum(len(build_input_tokens(t)) for t in trajectories),
            num_forward_passes=1,
        )
        return [lp.detach().clone().float().requires_grad_(True) for lp in per_trajectory]

    # ------------------------------------------------------------------- the gradient

    async def apply_gradient(
        self,
        trajectories: TrajectoryBatch,
        grads: Sequence[torch.Tensor],
        *,
        grad_scale: float = 1.0,
        defer: bool = False,
    ) -> dict[str, float]:
        """Backpropagate ``dC/dlogprobs`` into the LoRA parameters.

        Args:
            trajectories: The same batch, in the same order, that produced ``grads``.
            grads: One gradient tensor per trajectory, each shaped
                ``(len(completion_tokens),)`` — completion-only, exactly as
                :meth:`compute_logprobs` returned.
            grad_scale: Multiplier applied to the surrogate before backward. Gradients
                *accumulate* into ``.grad`` until :meth:`optim_step` clears them, so this
                is where a batch-size denominator goes.
            defer: Accepted and ignored. It exists so a hosted backend can submit a
                gradient without draining it and let the server batch it with the
                following optimiser step; locally there is no submission to pipeline.

        Returns:
            Metrics: the surrogate value, the scale applied, and how many trajectories
            were pushed.

        Raises:
            ValueError: If the batch is empty or ``grads`` does not align with it.
        """
        if not trajectories:
            raise ValueError("apply_gradient() needs at least one trajectory")
        if len(grads) != len(trajectories):
            raise ValueError(
                f"got {len(grads)} gradient tensors for {len(trajectories)} trajectories; "
                "they must align index-for-index"
            )
        for i, (g, traj) in enumerate(zip(grads, trajectories, strict=True)):
            if g.ndim != 1 or g.shape[0] != traj.num_completion_tokens:
                raise ValueError(
                    f"grads[{i}] has shape {tuple(g.shape)} but trajectory "
                    f"{traj.task_id!r} has {traj.num_completion_tokens} completion tokens; "
                    "gradients are completion-only, exactly as compute_logprobs returned "
                    "them"
                )

        total = 0.0
        for start in range(0, len(trajectories), self._micro_batch_size):
            chunk = list(trajectories[start : start + self._micro_batch_size])
            chunk_grads = list(grads[start : start + self._micro_batch_size])
            per_trajectory = self._forward_logprobs(chunk)
            # S = sum(logprobs * dC/dlogprobs), so dS/dlogprobs is the objective's own
            # gradient and the model's backward deposits exactly dC/dtheta.
            surrogate = torch.stack(
                [
                    (lp.float() * g.detach().to(lp.device, torch.float32)).sum()
                    for lp, g in zip(per_trajectory, chunk_grads, strict=True)
                ]
            ).sum()
            scaled = surrogate * grad_scale
            if self._accelerator is not None:
                self._accelerator.backward(scaled)
            else:
                scaled.backward()
            total += float(surrogate.detach().item())

        self._usage += TokenUsage(
            train_tokens=sum(len(build_input_tokens(t)) for t in trajectories),
            num_backward_passes=1,
        )
        return {
            "surrogate": total,
            "grad_scale": float(grad_scale),
            "num_trajectories": float(len(trajectories)),
        }

    async def optim_step(self, lr: float, *, grad_clip_norm: float = 0.0) -> dict[str, float]:
        """Step the policy optimiser and clear the accumulated gradient.

        Args:
            lr: Learning rate for this step, from the schedule in
                :func:`flowcode.train.learning_rate`. Written into every param group, so
                the schedule owns the LR rather than a torch scheduler duplicating it.
            grad_clip_norm: Global grad-norm clip; ``0.0`` disables.

        Returns:
            Metrics: the learning rate used and the pre-clip gradient norm.

        Raises:
            ValueError: If ``lr`` is not positive.
        """
        if lr <= 0.0:
            raise ValueError(f"optim_step needs a positive learning rate, got {lr}")

        for group in self._optimizer.param_groups:
            group["lr"] = lr

        grad_norm = 0.0
        if grad_clip_norm > 0.0:
            params = [p for g in self._optimizer.param_groups for p in g["params"]]
            if self._accelerator is not None:
                clipped = self._accelerator.clip_grad_norm_(params, grad_clip_norm)
                grad_norm = float(clipped) if clipped is not None else 0.0
            else:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(params, grad_clip_norm).item())

        self._optimizer.step()
        self._optimizer.zero_grad(set_to_none=True)
        self._usage += TokenUsage(num_optim_steps=1)
        return {"lr": float(lr), "grad_norm": grad_norm}

    def token_usage(self) -> TokenUsage:
        """An immutable snapshot of cumulative token counts."""
        return self._usage

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        """The trainable (LoRA) parameters, for checkpointing."""
        return (p for p in self._model.parameters() if p.requires_grad)

    # ------------------------------------------------------------------- checkpointing

    def save_checkpoint(self, path: str) -> str:
        """Write the adapter and the optimiser state to ``path``.

        The optimiser goes with the adapter deliberately. Adam's second-moment estimate
        over LoRA parameters takes hundreds of steps to settle, and a resume that restores
        only the weights spends those steps taking badly-scaled updates on an otherwise
        converged policy.

        Args:
            path: Directory to write into. Created if absent.

        Returns:
            The directory written.
        """
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        self._unwrapped().save_pretrained(str(target / "adapter"))
        torch.save(
            {
                "optimizer": self._optimizer.state_dict(),
                "policy_version": self._policy_version,
                "usage": self._usage,
            },
            target / "backend.pt",
        )
        logger.info("wrote backend checkpoint to %s", target)
        return str(target)

    def load_checkpoint(self, path: str) -> None:
        """Restore the adapter and optimiser state written by :meth:`save_checkpoint`.

        Args:
            path: A directory :meth:`save_checkpoint` wrote to.

        Raises:
            FileNotFoundError: If either half is missing. Restoring one without the other
                is not a degraded resume, it is a differently-behaved run.
            ImportError: If the ``local`` extra is not installed.
        """
        from peft import load_peft_weights, set_peft_model_state_dict

        target = Path(path)
        adapter = target / "adapter"
        backend_state = target / "backend.pt"
        if not adapter.is_dir():
            raise FileNotFoundError(f"no adapter directory in {target}")
        if not backend_state.is_file():
            raise FileNotFoundError(
                f"no backend.pt in {target}; the adapter alone would resume with a cold "
                "optimiser, which takes badly-scaled steps on a converged policy"
            )

        set_peft_model_state_dict(self._unwrapped(), load_peft_weights(str(adapter)))
        payload = torch.load(backend_state, map_location="cpu", weights_only=False)
        self._optimizer.load_state_dict(payload["optimizer"])
        self._policy_version = int(payload.get("policy_version", 0))
        usage = payload.get("usage")
        if isinstance(usage, TokenUsage):
            self._usage = usage
        logger.info("restored backend from %s at policy version %d", target, self._policy_version)

    # --------------------------------------------------------------------- internals

    def _unwrapped(self) -> Any:
        """The bare PEFT model, with any Accelerate/DDP wrapper peeled off."""
        if self._accelerator is not None:
            return self._accelerator.unwrap_model(self._model)
        return self._model

    def _device(self) -> torch.device:
        return next(self._model.parameters()).device

    def _forward_logprobs(self, trajectories: Sequence[Trajectory]) -> list[torch.Tensor]:
        """One micro-batch forward, returning completion-only logprobs per trajectory.

        Right-pads the batch and masks the padding out. The returned tensors keep whatever
        autograd graph the caller's context allows: under ``no_grad`` they are constants,
        otherwise they are differentiable back into the LoRA parameters.
        """
        device = self._device()
        inputs = [build_input_tokens(t) for t in trajectories]
        targets = [build_target_tokens(t) for t in trajectories]
        width = max(len(row) for row in inputs)

        input_ids = torch.zeros((len(inputs), width), dtype=torch.long, device=device)
        target_ids = torch.zeros((len(inputs), width), dtype=torch.long, device=device)
        attention_mask = torch.zeros((len(inputs), width), dtype=torch.long, device=device)
        for i, (row, target) in enumerate(zip(inputs, targets, strict=True)):
            n = len(row)
            input_ids[i, :n] = torch.tensor(row, dtype=torch.long, device=device)
            target_ids[i, :n] = torch.tensor(target, dtype=torch.long, device=device)
            attention_mask[i, :n] = 1

        logits = self._model(input_ids=input_ids, attention_mask=attention_mask).logits
        # float32 before the log_softmax, always. In bf16 the normaliser carries ~3 decimal
        # digits, and these logprobs are compared against the sampler's own — a difference
        # that shows up as phantom importance weights rather than as an error.
        token_logprobs = (
            torch.log_softmax(logits.float(), dim=-1)
            .gather(-1, target_ids.unsqueeze(-1))
            .squeeze(-1)
        )

        out: list[torch.Tensor] = []
        for i, trajectory in enumerate(trajectories):
            row = token_logprobs[i, : len(inputs[i])]
            out.append(slice_completion_values(trajectory, row))
        return out

    @classmethod
    def create(
        cls,
        model_cfg: ModelConfig,
        *,
        sampler: WeightSyncSampler | None = None,
        dtype: str = "bfloat16",
        lora_alpha: int | None = None,
        lora_dropout: float = 0.0,
        micro_batch_size: int = 8,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.0,
        gradient_checkpointing: bool = False,
        attn_implementation: str | None = None,
        adapter_dir: str = "adapters",
        mixed_precision: str | None = None,
    ) -> LocalBackend:
        """Load the base model, attach a LoRA, and build the optimiser.

        Args:
            model_cfg: Base model, LoRA rank and which module families get adapters.
            sampler: Where :meth:`sync_sampler` pushes weights.
            dtype: Parameter dtype — ``bfloat16`` on anything modern.
            lora_alpha: LoRA scaling. Defaults to ``2 * lora_rank``, the usual convention.
            lora_dropout: Dropout on the LoRA path.
            micro_batch_size: Trajectories per forward pass.
            learning_rate: Initial LR. Overwritten every :meth:`optim_step` by the
                schedule, so it only matters before the first one.
            weight_decay: AdamW weight decay.
            gradient_checkpointing: Trade compute for activation memory.
            attn_implementation: Passed to ``from_pretrained``; ``None`` lets
                transformers choose.
            adapter_dir: Where sampler-sync adapters are written.
            mixed_precision: Forwarded to ``Accelerator``; ``None`` means no autocast,
                which is right when the parameters are already bf16.

        Returns:
            A ready backend.

        Raises:
            ImportError: If the ``local`` extra is not installed.
        """
        try:
            from accelerate import Accelerator
            from peft import LoraConfig, get_peft_model
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "LocalBackend needs the `local` extra: transformers, peft and accelerate. "
                "Install it with `uv sync --extra local` or `mise run sync`."
            ) from exc

        torch_dtype = resolve_dtype(dtype)
        targets = lora_target_modules(model_cfg)

        load_kwargs: dict[str, Any] = {"dtype": torch_dtype}
        if attn_implementation is not None:
            load_kwargs["attn_implementation"] = attn_implementation
        model = AutoModelForCausalLM.from_pretrained(model_cfg.name, **load_kwargs)
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()

        lora = LoraConfig(
            r=model_cfg.lora_rank,
            lora_alpha=2 * model_cfg.lora_rank if lora_alpha is None else lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=targets,
            task_type="CAUSAL_LM",
            bias="none",
        )
        peft_model = get_peft_model(model, lora)
        tokenizer = AutoTokenizer.from_pretrained(model_cfg.name)

        trainable = [p for p in peft_model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)

        accelerator = Accelerator(mixed_precision=mixed_precision)
        prepared_model, prepared_optimizer = accelerator.prepare(peft_model, optimizer)
        logger.info(
            "LocalBackend ready: base_model=%s lora_rank=%d trainable=%d device=%s",
            model_cfg.name,
            model_cfg.lora_rank,
            sum(p.numel() for p in trainable),
            accelerator.device,
        )
        return cls(
            prepared_model,
            tokenizer,
            prepared_optimizer,
            model_cfg,
            accelerator=accelerator,
            sampler=sampler,
            micro_batch_size=micro_batch_size,
            adapter_dir=adapter_dir,
        )

    def __repr__(self) -> str:
        return (
            f"LocalBackend(model={self.model_cfg.name!r}, "
            f"lora_rank={self.model_cfg.lora_rank}, "
            f"micro_batch_size={self._micro_batch_size}, "
            f"policy_version={self._policy_version})"
        )
