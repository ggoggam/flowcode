"""What a run will cost, and what it did cost.

Tinker bills per token, separately for training and sampling, at a rate that depends on
the base model. A GFlowNet loop has an unusual shape compared to plain SFT, and the shape
is what dominates the bill:

* Every step samples ``groups_per_step * group_size`` completions, not one per prompt.
  Group size is not a batching convenience — VarGrad estimates ``log Z`` from within the
  group — so it cannot be turned down to save money without changing the objective.
* Every step runs **two** training passes over the same tokens, not one. The first is the
  all-zero-weight ``forward()`` oracle that fetches current-policy logprobs; the second is
  the ``forward_backward()`` that pushes ``weights = -dC/dlogprobs``. See
  :mod:`flowcode.tinker_backend`. Setting ``train.on_policy_only=true`` reuses the
  sampler's own logprobs and drops the oracle pass, which halves the training-token bill
  exactly.

So the headline knobs are ``groups_per_step``, ``group_size`` and ``on_policy_only``, and
this module exists so you can see that before you spend rather than after.

Prices are a hardcoded snapshot and will drift; override them via ``cost.price_overrides``
rather than editing this file. Where a price is genuinely unpublished the table holds
``None`` and every number downstream of it reports ``"unknown"`` — an invented number that
looks authoritative is worse than an admitted gap.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from rich.table import Table

from flowcode.config import RootConfig
from flowcode.types import TokenUsage

__all__ = [
    "PRICES",
    "CostEstimate",
    "ModelPrice",
    "estimate",
    "from_usage",
    "resolve_price",
]

_USD_PER_MTOK = 1_000_000.0


@dataclass(frozen=True)
class ModelPrice:
    """USD per million tokens for one base model.

    Args:
        train: Price per Mtok for ``forward`` / ``forward_backward``.
        sample: Price per Mtok for sampling, or ``None`` when Tinker publishes no sampling
            price for this model (several of the large models are train-only).
    """

    train: float
    sample: float | None


# Snapshot of Tinker's published per-Mtok pricing as of **August 2026**. This WILL go
# stale. Nothing reads it without first consulting `cost.price_overrides`, so a stale row
# is always fixable from config:
#
#   flowcode-cost '+cost.price_overrides={Qwen/Qwen3-8B: [0.50, 0.65]}'
#
# The `+` is Hydra's, not ours: the composed config is in struct mode, so a model name
# the schema has never seen has to be added rather than overridden.
PRICES: dict[str, ModelPrice] = {
    "Qwen/Qwen3-8B": ModelPrice(train=0.44, sample=0.60),
    "Qwen/Qwen3.5-4B": ModelPrice(train=0.737, sample=1.005),
    "Qwen/Qwen3.5-9B": ModelPrice(train=1.463, sample=None),
    "Qwen/Qwen3.5-9B-Base": ModelPrice(train=1.463, sample=None),
    "openai/gpt-oss-20b": ModelPrice(train=0.396, sample=0.45),
    "openai/gpt-oss-120b": ModelPrice(train=0.737, sample=None),
    "Qwen/Qwen3.6-27B": ModelPrice(train=4.103, sample=None),
    "deepseek-ai/DeepSeek-V3.1": ModelPrice(train=3.718, sample=None),
    "moonshotai/Kimi-K2.6": ModelPrice(train=4.84, sample=None),
}


def _parse_override(model: str, raw: Any) -> ModelPrice:
    """Turn one ``cost.price_overrides`` entry into a :class:`ModelPrice`.

    Accepts plain Python and OmegaConf containers alike — a value straight off a composed
    config arrives as ``ListConfig``/``DictConfig``, not ``list``/``dict``, which is why
    the checks are against the ABCs rather than the concrete types.
    """
    if isinstance(raw, Mapping):
        missing = {"train", "sample"} - set(raw)
        if missing:
            raise ValueError(
                f"cost.price_overrides[{model!r}] is missing {sorted(missing)}; a mapping "
                "override needs both 'train' and 'sample' (use null for an unknown "
                "sample price)"
            )
        return ModelPrice(train=float(raw["train"]), sample=_opt_float(raw["sample"]))
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and len(raw) == 2:
        train, sample = raw[0], raw[1]
        return ModelPrice(train=float(train), sample=_opt_float(sample))
    raise ValueError(
        f"cost.price_overrides[{model!r}] must be [train_usd_per_mtok, "
        f"sample_usd_per_mtok] or {{train: ..., sample: ...}}, got {raw!r}"
    )


def _opt_float(value: Any) -> float | None:
    """``None`` stays ``None`` — an unpublished price must never become 0.0."""
    return None if value is None else float(value)


def resolve_price(model: str, price_overrides: Mapping[str, Any] | None = None) -> ModelPrice:
    """Look up a model's price, letting config override the built-in table.

    Args:
        model: Base model name, e.g. ``Qwen/Qwen3-8B``.
        price_overrides: Contents of ``cfg.cost.price_overrides``. Checked first.

    Returns:
        The price to bill at.

    Raises:
        KeyError: If the model is in neither the overrides nor :data:`PRICES`, with the
            known names listed. A wrong guess here silently misprices a whole sweep, so
            there is no fallback default.
    """
    if price_overrides and model in price_overrides:
        return _parse_override(model, price_overrides[model])
    if model in PRICES:
        return PRICES[model]
    raise KeyError(
        f"No price for base model {model!r}. Known models: {sorted(PRICES)}. If Tinker "
        "added it after the Aug 2026 snapshot in flowcode/cost.py, supply it with "
        f"'+cost.price_overrides={{{model}: [train_usd_per_mtok, sample_usd_per_mtok]}}'."
    )


def _usd(tokens: float, price_per_mtok: float | None) -> float | None:
    """Tokens at a per-Mtok price, or ``None`` when the price is unknown."""
    if price_per_mtok is None:
        return None
    return tokens / _USD_PER_MTOK * price_per_mtok


@dataclass(frozen=True)
class CostEstimate:
    """A priced run, per step and in total.

    Every ``*_usd_*`` field is ``None`` exactly when the underlying price is unpublished;
    :meth:`render` prints ``unknown`` for those rather than a zero that would read as free.

    Args:
        model: Base model priced.
        steps: Number of optimiser steps covered by the ``*_per_run`` fields.
        price: The resolved price table row.
        samples_per_step: ``groups_per_step * group_size``.
        seq_len: Assumed ``prompt_tokens + completion_tokens`` per rollout.
        passes: Training passes over each token per step — 2 normally, 1 for
            ``on_policy_only``.
        sample_tokens_per_step: Billable sampling tokens, net of the assumed prompt-cache
            hit rate.
        train_tokens_per_step: Billable training tokens.
        sample_usd_per_step: Sampling cost for one step.
        train_usd_per_step: Training cost for one step.
    """

    model: str
    steps: int
    price: ModelPrice
    samples_per_step: int
    seq_len: int
    passes: int
    sample_tokens_per_step: float
    train_tokens_per_step: float
    sample_usd_per_step: float | None
    train_usd_per_step: float | None

    @property
    def total_usd_per_step(self) -> float | None:
        """Sampling plus training for one step, or ``None`` if either price is unknown."""
        if self.sample_usd_per_step is None or self.train_usd_per_step is None:
            return None
        return self.sample_usd_per_step + self.train_usd_per_step

    @property
    def sample_tokens_per_run(self) -> float:
        """Billable sampling tokens across all ``steps``."""
        return self.sample_tokens_per_step * self.steps

    @property
    def train_tokens_per_run(self) -> float:
        """Billable training tokens across all ``steps``."""
        return self.train_tokens_per_step * self.steps

    @property
    def sample_usd_per_run(self) -> float | None:
        """Sampling cost across all ``steps``."""
        return None if self.sample_usd_per_step is None else self.sample_usd_per_step * self.steps

    @property
    def train_usd_per_run(self) -> float | None:
        """Training cost across all ``steps``."""
        return None if self.train_usd_per_step is None else self.train_usd_per_step * self.steps

    @property
    def total_usd_per_run(self) -> float | None:
        """Total cost across all ``steps``, or ``None`` if any price is unknown."""
        total = self.total_usd_per_step
        return None if total is None else total * self.steps

    def render(self, title: str | None = None) -> Table:
        """Format as a rich table.

        Args:
            title: Table title; defaults to naming the model and step count.

        Returns:
            A :class:`rich.table.Table` for the caller to print. Returning rather than
            printing keeps this usable from the CLI, from a notebook and from tests.
        """
        table = Table(
            title=title or f"{self.model} — {self.steps} steps, {self.passes} train pass(es)/step",
            title_justify="left",
        )
        table.add_column("", style="bold")
        table.add_column("tokens/step", justify="right")
        table.add_column("USD/step", justify="right")
        table.add_column("tokens/run", justify="right")
        table.add_column("USD/run", justify="right")

        def money(value: float | None) -> str:
            return "unknown" if value is None else f"${value:,.4f}"

        table.add_row(
            "sample",
            f"{self.sample_tokens_per_step:,.0f}",
            money(self.sample_usd_per_step),
            f"{self.sample_tokens_per_run:,.0f}",
            money(self.sample_usd_per_run),
        )
        table.add_row(
            "train",
            f"{self.train_tokens_per_step:,.0f}",
            money(self.train_usd_per_step),
            f"{self.train_tokens_per_run:,.0f}",
            money(self.train_usd_per_run),
        )
        table.add_section()
        table.add_row(
            "total",
            f"{self.sample_tokens_per_step + self.train_tokens_per_step:,.0f}",
            money(self.total_usd_per_step),
            f"{self.sample_tokens_per_run + self.train_tokens_per_run:,.0f}",
            money(self.total_usd_per_run),
        )
        if self.price.sample is None:
            table.caption = (
                f"Tinker publishes no sampling price for {self.model}; sampling cost is "
                "reported as unknown, not as zero."
            )
        return table


def estimate(cfg: RootConfig, steps: int | None = None) -> CostEstimate:
    """Price a run before it happens, from the same config that will run it.

    The arithmetic, in full::

        seq_len            = cost.prompt_tokens + cost.completion_tokens
        samples_per_step   = train.groups_per_step * train.group_size
        billable_prompt    = cost.prompt_tokens * (1 - cost.prompt_cache_hit_rate)
        sample_tokens/step = samples_per_step * (billable_prompt + cost.completion_tokens)
        passes             = 1 if train.on_policy_only else 2
        train_tokens/step  = passes * samples_per_step * seq_len

    Training tokens get no cache discount: ``forward``/``forward_backward`` re-prefill the
    whole sequence every pass. Sampling tokens do, on the assumption that a cache hit is
    not billed at the sampling rate — an assumption worth checking against
    :func:`from_usage` after your first real job, since the estimate and the actual are
    the two numbers this module exists to let you compare.

    Worked example (the one the test pins). ``Qwen/Qwen3-8B`` at 8 groups x 8 samples,
    300-token prompts and 400-token completions — ~700 tokens a rollout — off-policy, and
    ``prompt_cache_hit_rate=0.0`` so nothing is discounted away::

        samples_per_step   = 8 * 8                     = 64
        seq_len            = 300 + 400                 = 700
        sample_tokens/step = 64 * 700                  = 44,800
        train_tokens/step  = 2 * 64 * 700              = 89,600
        sample USD/step    = 0.0448 Mtok * $0.60/Mtok  = $0.026880
        train  USD/step    = 0.0896 Mtok * $0.44/Mtok  = $0.039424
        total  USD/step                                = $0.066304

    So roughly **$0.066 a step**, or $66 for a 1000-step run — and $0.0466 a step with
    ``train.on_policy_only=true``, which is where the halved training pass shows up. With
    the shipped ``cost.prompt_cache_hit_rate=0.8`` the sampling side drops to $0.017664
    and the step to $0.057088.

    Args:
        cfg: The composed run config.
        steps: Steps to price. Defaults to ``cfg.train.steps``.

    Returns:
        The estimate.

    Raises:
        KeyError: If the model has no known price.
        ValueError: If ``steps`` or any config quantity is non-positive, or the cache hit
            rate is outside ``[0, 1]``.
    """
    steps = cfg.train.steps if steps is None else steps
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    if not 0.0 <= cfg.cost.prompt_cache_hit_rate <= 1.0:
        raise ValueError(
            f"cost.prompt_cache_hit_rate must be in [0, 1], got {cfg.cost.prompt_cache_hit_rate}"
        )
    if cfg.cost.prompt_tokens <= 0 or cfg.cost.completion_tokens <= 0:
        raise ValueError(
            "cost.prompt_tokens and cost.completion_tokens must both be positive; got "
            f"{cfg.cost.prompt_tokens} and {cfg.cost.completion_tokens}"
        )
    if cfg.train.groups_per_step <= 0 or cfg.train.group_size <= 0:
        raise ValueError(
            "train.groups_per_step and train.group_size must both be positive; got "
            f"{cfg.train.groups_per_step} and {cfg.train.group_size}"
        )

    price = resolve_price(cfg.model.name, cfg.cost.price_overrides)
    seq_len = cfg.cost.prompt_tokens + cfg.cost.completion_tokens
    samples_per_step = cfg.train.groups_per_step * cfg.train.group_size
    billable_prompt = cfg.cost.prompt_tokens * (1.0 - cfg.cost.prompt_cache_hit_rate)
    sample_tokens = samples_per_step * (billable_prompt + cfg.cost.completion_tokens)
    # The forward() oracle pass plus the forward_backward() push. on_policy_only reuses
    # the sampler's logprobs instead of the oracle, so it pays for exactly one.
    passes = 1 if cfg.train.on_policy_only else 2
    train_tokens = float(passes * samples_per_step * seq_len)

    return CostEstimate(
        model=cfg.model.name,
        steps=steps,
        price=price,
        samples_per_step=samples_per_step,
        seq_len=seq_len,
        passes=passes,
        sample_tokens_per_step=sample_tokens,
        train_tokens_per_step=train_tokens,
        sample_usd_per_step=_usd(sample_tokens, price.sample),
        train_usd_per_step=_usd(train_tokens, price.train),
    )


def from_usage(
    usage: TokenUsage,
    model: str,
    price_overrides: Mapping[str, Any] | None = None,
) -> CostEstimate:
    """Price what a backend actually spent.

    Unlike :func:`estimate` this makes no assumptions: the sampling figure uses the real
    ``prompt_cache_hit_tokens`` Tinker reported, and the training figure counts the passes
    that actually happened. Compare the two to find out how wrong ``conf/cost/default.yaml``
    was for your workload.

    The result is shaped as a one-"step" estimate — ``steps=1`` and the per-step fields
    holding the cumulative totals — so :meth:`CostEstimate.render` works unchanged.

    Args:
        usage: Snapshot from :meth:`flowcode.tinker_backend.TinkerBackend.token_usage`.
        model: Base model the tokens were spent on.
        price_overrides: Contents of ``cfg.cost.price_overrides``, if any.

    Returns:
        The actual spend.

    Raises:
        KeyError: If the model has no known price.
    """
    price = resolve_price(model, price_overrides)
    sample_tokens = float(usage.billable_sample_tokens)
    train_tokens = float(usage.train_tokens)
    passes = usage.num_forward_passes + usage.num_backward_passes
    # Zero sampling costs zero regardless of whether a sample price is published, so a
    # train-only model does not report an unknown total for a train-only run.
    sample_usd = 0.0 if sample_tokens == 0.0 else _usd(sample_tokens, price.sample)
    return CostEstimate(
        model=model,
        steps=1,
        price=price,
        samples_per_step=0,
        seq_len=0,
        passes=passes,
        sample_tokens_per_step=sample_tokens,
        train_tokens_per_step=train_tokens,
        sample_usd_per_step=sample_usd,
        train_usd_per_step=_usd(train_tokens, price.train),
    )
