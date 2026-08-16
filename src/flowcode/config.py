"""Hydra structured configs, plus the two secrets that deliberately bypass them.

Every dataclass here mirrors a YAML file under ``conf/`` field-for-field. Registering
them in Hydra's :class:`~hydra.core.config_store.ConfigStore` turns the YAML into a
*typed* config: ``train.group_size=notanint`` and ``train.grop_size=16`` both fail during
composition rather than ten minutes into a paid run.

Two things about this file are non-obvious and were both settled by experiment against
hydra 1.3.5 / omegaconf 2.3.1 rather than by reading docs:

**The dataclasses are not frozen.** ``@dataclass(frozen=True)`` makes OmegaConf mark the
corresponding node read-only, and Hydra applies command-line overrides by *mutating* the
composed config. A frozen schema therefore turns every documented override in
``conf/config.yaml`` into ``ReadonlyConfigError: Cannot change read-only config
container``. Clearing the flag at store time does not help either — the flag is re-derived
from the reference type on every merge. Treat these as read-only by convention.

**Group-backed fields default to ``MISSING``, not to a nested default.** ``conf/config.yaml``
ends its defaults list with ``- _self_``, so the root schema is merged *after* the group
selections. A schema carrying real nested defaults would silently clobber
``model=qwen3-8b`` back to the dataclass defaults. ``MISSING`` is skipped by OmegaConf's
merge, so the group's values survive while the *type* still propagates — which is what
makes ``cfg.model`` come out as a ``ModelConfig`` and get validated.

Secrets do not live in Hydra at all. A composed config is dumped verbatim into
``outputs/.../.hydra/config.yaml`` on every run, and an API key does not belong in a file
that a sweep writes a hundred copies of. Use :func:`get_api_key` and
:func:`get_project_id`, which read the process environment that ``mise`` has already
populated from the repo-root ``.env``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

__all__ = [
    "API_KEY_ENV_VAR",
    "PROJECT_ID_ENV_VAR",
    "BackendConfig",
    "CostConfig",
    "ModelConfig",
    "ProducerConfig",
    "ReplayConfig",
    "RootConfig",
    "SamplerConfig",
    "TrainConfig",
    "get_api_key",
    "get_project_id",
    "register_configs",
]


@dataclass
class ModelConfig:
    """Which base model to fine-tune and which parts of it get a LoRA.

    Mirrors ``conf/model/*.yaml``.

    Args:
        name: Tinker base-model identifier, e.g. ``Qwen/Qwen3-8B``. Also the key into
            the price table in :mod:`flowcode.cost`.
        renderer: Chat-template/renderer name used to build prompts for this model.
        max_context: Context window in tokens. Prompt plus completion must fit.
        lora_rank: LoRA rank passed to ``create_lora_training_client``.
        train_mlp: Attach LoRAs to the MLP (and MoE) layers.
        train_attn: Attach LoRAs to the attention layers.
        train_unembed: Attach a LoRA to the unembedding layer.
    """

    name: str = MISSING
    renderer: str = MISSING
    max_context: int = 32768
    lora_rank: int = 32
    train_mlp: bool = True
    train_attn: bool = True
    train_unembed: bool = True


@dataclass
class ReplayConfig:
    """Off-policy replay buffer. Mirrors the ``replay`` block of ``conf/train/*.yaml``.

    Args:
        enabled: Draw part of each batch from the buffer.
        capacity: Maximum trajectories retained.
        fraction: Share of each batch taken from replay rather than fresh rollouts.
        prioritize: Sampling strategy — ``reward`` | ``uniform`` | ``recency``.
    """

    enabled: bool = True
    capacity: int = 10000
    fraction: float = 0.25
    prioritize: str = "reward"


@dataclass
class ProducerConfig:
    """The decoupled sampler. Mirrors the ``producer`` block of ``conf/train/*.yaml``.

    See :mod:`flowcode.producer` for why running the sampler ahead of the trainer is sound
    for a GFlowNet and would not be for PPO.

    Args:
        enabled: Run sampling and scoring continuously in the background. When ``false``
            the loop samples inside the step, as it always did.
        concurrency: Workers sampling in parallel, each holding one group. Sequences in
            flight is ``concurrency * group_size``, and the point of the whole exercise is
            for that to be much larger than one step's worth.
        queue_size: Bound on queued groups. Backpressure once full, which is what stops a
            fast sampler running arbitrarily far ahead of a slow trainer.
        max_staleness: Discard groups more than this many policy versions behind.
            ``0`` keeps everything, which is the right default — the objectives do not
            need the bound; it exists for ablations and for capping drift.
    """

    enabled: bool = False
    concurrency: int = 8
    queue_size: int = 32
    max_staleness: int = 0


@dataclass
class BackendConfig:
    """Where the policy lives. Mirrors ``conf/backend/*.yaml``.

    Args:
        kind: ``tinker`` for the hosted LoRA service, ``local`` for a model in this
            process (see :mod:`flowcode.local_backend`).
        micro_batch_size: Trajectories per forward pass, ``local`` only. Peak activation
            memory scales with this rather than with the batch size.
        dtype: Parameter dtype — ``bfloat16`` on anything modern.
        mixed_precision: Passed to ``Accelerator``. ``null`` means no autocast, which is
            right when the parameters are already bf16.
        gradient_checkpointing: Trade compute for activation memory.
        attn_implementation: ``flash_attention_2`` / ``sdpa`` / ``eager``; ``null`` lets
            transformers choose.
        weight_decay: AdamW weight decay on the LoRA parameters.
        lora_alpha: LoRA scaling; ``null`` uses the usual ``2 * lora_rank``.
        lora_dropout: Dropout on the LoRA path.
        adapter_dir: Where sampler-sync adapters are written. On a remote sampler this
            must be a path the engine's host can read too.
        checkpoint_dir: Where run checkpoints go.
    """

    kind: str = "tinker"
    micro_batch_size: int = 8
    dtype: str = "bfloat16"
    mixed_precision: str | None = None
    gradient_checkpointing: bool = False
    attn_implementation: str | None = None
    weight_decay: float = 0.0
    lora_alpha: int | None = None
    lora_dropout: float = 0.0
    adapter_dir: str = "adapters"
    checkpoint_dir: str = "checkpoints"


@dataclass
class SamplerConfig:
    """Where completions come from. Mirrors ``conf/sampler/*.yaml``.

    Only consulted when ``backend.kind=local``; the hosted backend samples from the same
    service it trains on.

    Args:
        mode: ``colocated`` shares devices with the trainer; ``remote`` reaches an engine
            already running elsewhere over its OpenAI-compatible server.
        base_url: Engine URL, ``remote`` only.
        auth_token_env: Environment variable holding the server's bearer token, if it
            needs one. Named for the *variable*, not the value: a composed config is dumped
            verbatim into the run directory, and tests/test_config.py enforces that nothing
            resembling a credential ever appears there.
        gpu_memory_utilization: Fraction of device memory the colocated engine may claim.
            Well below the single-tenant default because the trainer needs the rest.
        max_model_len: Engine context window; ``null`` takes the model's own.
        tensor_parallel_size: Devices to shard the colocated engine over.
        enable_prefix_caching: Share prefills across requests.
        max_lora_rank: Must be at least ``model.lora_rank``.
    """

    mode: str = "colocated"
    base_url: str | None = None
    auth_token_env: str | None = None
    gpu_memory_utilization: float = 0.4
    max_model_len: int | None = None
    tensor_parallel_size: int = 1
    enable_prefix_caching: bool = True
    max_lora_rank: int = 32


@dataclass
class TrainConfig:
    """The training loop's shape. Mirrors ``conf/train/*.yaml``.

    The two learning rates are deliberate. ``policy_lr`` steps the LoRA on Tinker's side
    via ``optim_step``; ``flow_lr`` steps the client-side ``log Z`` / ``log F`` parameters,
    which fit a handful of scalars and need one to three orders of magnitude more.

    Args:
        steps: Number of optimiser steps.
        groups_per_step: Distinct prompts sampled per step.
        group_size: Completions per prompt. Group-baseline estimators (VarGrad) need
            this above 1.
        temperature: Rollout sampling temperature.
        top_p: Nucleus sampling probability.
        max_tokens: Completion-length cap per rollout.
        policy_lr: Learning rate for the LoRA policy weights.
        flow_lr: Learning rate for the client-side flow parameters.
        lr_schedule: ``linear`` | ``cosine`` | ``constant``.
        warmup_steps: Steps of linear warmup before the schedule proper.
        grad_clip_norm: Global grad-norm clip; ``0.0`` disables (Tinker's default).
        replay: Off-policy replay settings.
        producer: Decoupled-sampler settings.
        on_policy_only: Reuse the sampler's own logprobs as the current-policy logprobs
            and skip the ``forward()`` oracle pass. Roughly halves the training-token
            bill; incompatible with replay.
        sync_sampler_every: Steps between sampler weight syncs.
        log_every: Steps between metric dumps.
        eval_every: Steps between evals; ``0`` disables.
        checkpoint_every: Steps between checkpoints; ``0`` disables.
    """

    steps: int = 1000
    groups_per_step: int = 8
    group_size: int = 8
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 512
    policy_lr: float = 1e-5
    flow_lr: float = 1e-2
    lr_schedule: str = "cosine"
    warmup_steps: int = 10
    grad_clip_norm: float = 1.0
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    producer: ProducerConfig = field(default_factory=ProducerConfig)
    on_policy_only: bool = False
    sync_sampler_every: int = 1
    log_every: int = 1
    eval_every: int = 50
    checkpoint_every: int = 100


@dataclass
class CostConfig:
    """Assumptions the cost estimator prices a run with. Mirrors ``conf/cost/*.yaml``.

    Args:
        prompt_tokens: Assumed prompt length per rollout.
        completion_tokens: Assumed completion length per rollout.
        prompt_cache_hit_rate: Fraction of prompt tokens expected to hit Tinker's prefix
            cache during sampling. Within a group every sample shares one prompt, so all
            but the first should hit; across groups there is no sharing.
        price_overrides: Per-model ``{name: [train_usd_per_mtok, sample_usd_per_mtok]}``
            replacements for the built-in table. ``null`` for a sample price means the
            model cannot be sampled from / the price is unpublished.
    """

    prompt_tokens: int = 300
    completion_tokens: int = 400
    prompt_cache_hit_rate: float = 0.8
    # Hydra puts the whole composed config in struct mode, so adding a key that is not
    # already present needs a `+` prefix on the command line no matter how this field is
    # typed (`Any` behaves identically):
    #
    #   flowcode-cost '+cost.price_overrides={Qwen/Qwen3-8B: [0.50, 0.65]}'
    #
    # Contents are validated by flowcode.cost rather than here, so a malformed entry
    # produces a message about prices instead of about OmegaConf.
    price_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass
class RootConfig:
    """The composed run config. Mirrors ``conf/config.yaml`` plus its defaults list.

    ``objective`` and ``env`` are typed ``Any`` on purpose: both YAML groups carry a
    ``_target_`` and are built by ``hydra.utils.instantiate``, whose recursive
    instantiation fights a declared schema (the nested ``log_z`` / ``flow`` blocks differ
    per objective, so no single dataclass fits all four). They are validated by
    instantiation failing loudly instead.

    Args:
        seed: Global RNG seed.
        output_dir: Where checkpoints and metric dumps go. Hydra's own run dir holds the
            composed config and logs; this outlives a sweep.
        logger: ``rich`` | ``wandb`` | ``none``.
        model: Selected ``model`` group.
        objective: Selected ``objective`` group, un-instantiated.
        env: Selected ``env`` group, un-instantiated.
        train: Selected ``train`` group.
        cost: Selected ``cost`` group.
        backend: Selected ``backend`` group — where the policy lives.
        sampler: Selected ``sampler`` group — where completions come from.
    """

    seed: int = 0
    output_dir: str = "runs"
    logger: str = "rich"
    # MISSING, not a default_factory — see this module's docstring. `_self_` is last in
    # conf/config.yaml's defaults list, so anything concrete here would overwrite the
    # group that was just selected.
    model: ModelConfig = MISSING
    objective: Any = MISSING
    env: Any = MISSING
    train: TrainConfig = MISSING
    cost: CostConfig = MISSING
    backend: BackendConfig = MISSING
    sampler: SamplerConfig = MISSING


def register_configs() -> None:
    """Register every schema with the global :class:`ConfigStore`.

    Idempotent, and called at import time of this module, so anything that imports
    :mod:`flowcode.config` before composing gets validation for free.

    The entry that does the real work is ``name="config"``: it shares a name with
    ``conf/config.yaml``, so Hydra applies it as that file's schema, and because
    :class:`RootConfig` declares typed ``model`` / ``train`` / ``cost`` fields the types
    propagate down into the group nodes. The per-group entries are registered under
    ``_schema_`` so a future YAML can pull one in explicitly via its own defaults list
    (``- /model/_schema_@_here_``); nothing in ``conf/`` needs that today.

    The root schema is registered as ``_root_`` and named explicitly as the first entry of
    ``conf/config.yaml``'s defaults list. Registering it as ``config`` instead would also
    work — Hydra matches a schema whose name equals the config file's — but that implicit
    match is deprecated and warns on every run ("'config' is validated against ConfigStore
    schema with the same name"). Verified against hydra 1.3.5: composition, group selection,
    override validation and the propagated node types are identical either way.
    """
    cs = ConfigStore.instance()
    cs.store(group="model", name="_schema_", node=ModelConfig)
    cs.store(group="train", name="_schema_", node=TrainConfig)
    cs.store(group="cost", name="_schema_", node=CostConfig)
    cs.store(group="backend", name="_schema_", node=BackendConfig)
    cs.store(group="sampler", name="_schema_", node=SamplerConfig)
    cs.store(name="_root_", node=RootConfig)


register_configs()


API_KEY_ENV_VAR = "TINKER_API_KEY"
"""Environment variable holding the Tinker credential."""

PROJECT_ID_ENV_VAR = "TINKER_PROJECT_ID"
"""Environment variable holding the optional Tinker project id."""

_MISSING_KEY_MESSAGE = (
    f"{API_KEY_ENV_VAR} is not set, so flowcode cannot talk to Tinker.\n"
    'The key lives in .env at the repo root, which mise loads via `_.file = ".env"` in '
    "mise.toml.\n"
    "Fix it with either of:\n"
    f"  echo '{API_KEY_ENV_VAR}=tk-...' >> .env   # then cd out and back so mise reloads\n"
    f"  export {API_KEY_ENV_VAR}=tk-...           # one-off, this shell only\n"
    "See .env.example for the full list of variables flowcode reads."
)


def get_api_key() -> str:
    """Read the Tinker API key from the environment.

    Returns:
        The key.

    Raises:
        RuntimeError: If the variable is unset or empty, with instructions naming the
            repo-root ``.env``.
    """
    key = os.environ.get(API_KEY_ENV_VAR, "").strip()
    if not key:
        raise RuntimeError(_MISSING_KEY_MESSAGE)
    return key


def get_project_id() -> str | None:
    """Read the optional Tinker project id from the environment.

    Returns:
        The project id, or ``None`` if unset or empty. Tinker's ``ServiceClient`` reads
        the same variable itself, so passing ``None`` through is harmless.
    """
    project_id = os.environ.get(PROJECT_ID_ENV_VAR, "").strip()
    return project_id or None
