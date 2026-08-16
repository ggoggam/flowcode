"""Cost arithmetic, pinned so it cannot drift silently.

A cost estimator that is quietly wrong is worse than no estimator: it is the thing you
check before authorising a sweep. The numbers here are computed by hand in the test and
compared against the module, rather than snapshotted from the module's own output.
"""

from __future__ import annotations

import pytest
from rich.console import Console
from rich.table import Table

from flowcode.config import CostConfig, ModelConfig, RootConfig, TrainConfig
from flowcode.cost import PRICES, ModelPrice, estimate, from_usage, resolve_price
from flowcode.types import TokenUsage


def usd(value: float | None) -> float:
    """Narrow a cost to a float.

    ``total_usd_per_step`` is ``float | None`` because a model with no published sampling
    price cannot be totalled. Tests that do arithmetic on it are all using models that DO
    have one, so this asserts that rather than suppressing the type error — if a price table
    edit ever makes one of them None, the test fails loudly instead of silently.
    """
    assert value is not None
    return value


def make_cfg(
    *,
    model: str = "Qwen/Qwen3-8B",
    groups_per_step: int = 8,
    group_size: int = 8,
    prompt_tokens: int = 300,
    completion_tokens: int = 400,
    prompt_cache_hit_rate: float = 0.0,
    on_policy_only: bool = False,
    steps: int = 1000,
    price_overrides: dict[str, object] | None = None,
) -> RootConfig:
    """A RootConfig instance shaped like the worked example in :func:`estimate`."""
    return RootConfig(
        model=ModelConfig(name=model, renderer="qwen3"),
        objective=None,
        env=None,
        train=TrainConfig(
            steps=steps,
            groups_per_step=groups_per_step,
            group_size=group_size,
            on_policy_only=on_policy_only,
        ),
        cost=CostConfig(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prompt_cache_hit_rate=prompt_cache_hit_rate,
            price_overrides=dict(price_overrides or {}),
        ),
    )


class TestWorkedExample:
    """The number in :func:`flowcode.cost.estimate`'s docstring, asserted.

    Qwen3-8B, 8 groups x 8 samples, 300-token prompts + 400-token completions, off-policy,
    no prompt-cache discount:

        sample: 64 * 700           = 44,800 tok  * $0.60/Mtok = $0.026880
        train:  2 * 64 * 700       = 89,600 tok  * $0.44/Mtok = $0.039424
        total                                                 = $0.066304
    """

    def test_lands_near_six_point_six_cents_per_step(self) -> None:
        est = estimate(make_cfg(), steps=1)
        assert est.total_usd_per_step == pytest.approx(0.066, abs=5e-4)

    def test_exact_arithmetic(self) -> None:
        est = estimate(make_cfg(), steps=1)
        assert est.samples_per_step == 64
        assert est.seq_len == 700
        assert est.passes == 2
        assert est.sample_tokens_per_step == pytest.approx(44_800.0)
        assert est.train_tokens_per_step == pytest.approx(89_600.0)
        assert est.sample_usd_per_step == pytest.approx(0.026880, abs=1e-9)
        assert est.train_usd_per_step == pytest.approx(0.039424, abs=1e-9)
        assert est.total_usd_per_step == pytest.approx(0.066304, abs=1e-9)

    def test_thousand_step_run_is_about_sixty_six_dollars(self) -> None:
        est = estimate(make_cfg(steps=1000))
        assert est.steps == 1000
        assert est.total_usd_per_run == pytest.approx(66.304, abs=1e-6)

    def test_shipped_cache_hit_rate_lowers_the_sampling_side(self) -> None:
        # conf/cost/default.yaml assumes 0.8; only prompt tokens are discounted.
        est = estimate(make_cfg(prompt_cache_hit_rate=0.8), steps=1)
        assert est.sample_tokens_per_step == pytest.approx(64 * (300 * 0.2 + 400))
        assert est.sample_usd_per_step == pytest.approx(0.017664, abs=1e-9)
        assert est.total_usd_per_step == pytest.approx(0.057088, abs=1e-9)

    def test_training_tokens_get_no_cache_discount(self) -> None:
        a = estimate(make_cfg(prompt_cache_hit_rate=0.0), steps=1)
        b = estimate(make_cfg(prompt_cache_hit_rate=1.0), steps=1)
        assert a.train_tokens_per_step == b.train_tokens_per_step
        assert b.sample_tokens_per_step == pytest.approx(64 * 400)


class TestPassesHalving:
    """``on_policy_only`` drops the forward() oracle pass — exactly half the train bill."""

    def test_passes_flag(self) -> None:
        assert estimate(make_cfg(on_policy_only=False), steps=1).passes == 2
        assert estimate(make_cfg(on_policy_only=True), steps=1).passes == 1

    def test_train_tokens_halve(self) -> None:
        off = estimate(make_cfg(on_policy_only=False), steps=1)
        on = estimate(make_cfg(on_policy_only=True), steps=1)
        assert on.train_tokens_per_step == pytest.approx(off.train_tokens_per_step / 2)
        assert on.train_usd_per_step == pytest.approx(usd(off.train_usd_per_step) / 2)

    def test_sampling_is_untouched_by_the_flag(self) -> None:
        off = estimate(make_cfg(on_policy_only=False), steps=1)
        on = estimate(make_cfg(on_policy_only=True), steps=1)
        assert on.sample_tokens_per_step == off.sample_tokens_per_step

    def test_on_policy_step_cost(self) -> None:
        est = estimate(make_cfg(on_policy_only=True), steps=1)
        assert est.total_usd_per_step == pytest.approx(0.026880 + 0.019712, abs=1e-9)


class TestScaling:
    def test_linear_in_groups(self) -> None:
        a = estimate(make_cfg(groups_per_step=8), steps=1)
        b = estimate(make_cfg(groups_per_step=16), steps=1)
        assert b.total_usd_per_step == pytest.approx(2 * usd(a.total_usd_per_step))

    def test_linear_in_group_size(self) -> None:
        a = estimate(make_cfg(group_size=8), steps=1)
        b = estimate(make_cfg(group_size=4), steps=1)
        assert b.total_usd_per_step == pytest.approx(usd(a.total_usd_per_step) / 2)

    def test_linear_in_steps(self) -> None:
        est = estimate(make_cfg(), steps=250)
        assert est.total_usd_per_run == pytest.approx(250 * usd(est.total_usd_per_step))

    def test_steps_defaults_to_config(self) -> None:
        assert estimate(make_cfg(steps=37)).steps == 37


class TestPriceTable:
    def test_snapshot_values(self) -> None:
        assert PRICES["Qwen/Qwen3-8B"] == ModelPrice(train=0.44, sample=0.60)
        assert PRICES["openai/gpt-oss-20b"] == ModelPrice(train=0.396, sample=0.45)
        assert PRICES["Qwen/Qwen3.5-4B"] == ModelPrice(train=0.737, sample=1.005)
        assert PRICES["moonshotai/Kimi-K2.6"] == ModelPrice(train=4.84, sample=None)

    @pytest.mark.parametrize(
        "model",
        [
            "Qwen/Qwen3.5-9B",
            "Qwen/Qwen3.5-9B-Base",
            "openai/gpt-oss-120b",
            "Qwen/Qwen3.6-27B",
            "deepseek-ai/DeepSeek-V3.1",
            "moonshotai/Kimi-K2.6",
        ],
    )
    def test_unknown_sample_prices_stay_none(self, model: str) -> None:
        assert PRICES[model].sample is None

    def test_unknown_model_raises_with_the_known_list(self) -> None:
        with pytest.raises(KeyError, match="price_overrides"):
            resolve_price("acme/not-a-model")

    def test_list_override_wins(self) -> None:
        price = resolve_price("Qwen/Qwen3-8B", {"Qwen/Qwen3-8B": [1.0, 2.0]})
        assert price == ModelPrice(train=1.0, sample=2.0)

    def test_mapping_override_wins(self) -> None:
        price = resolve_price("Qwen/Qwen3-8B", {"Qwen/Qwen3-8B": {"train": 1.0, "sample": None}})
        assert price == ModelPrice(train=1.0, sample=None)

    def test_override_can_supply_an_unlisted_model(self) -> None:
        price = resolve_price("acme/new-model", {"acme/new-model": [3.0, 4.0]})
        assert price == ModelPrice(train=3.0, sample=4.0)

    def test_malformed_override_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="price_overrides"):
            resolve_price("Qwen/Qwen3-8B", {"Qwen/Qwen3-8B": 0.44})

    def test_incomplete_mapping_override_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="sample"):
            resolve_price("Qwen/Qwen3-8B", {"Qwen/Qwen3-8B": {"train": 1.0}})

    def test_override_flows_through_estimate(self) -> None:
        cfg = make_cfg(price_overrides={"Qwen/Qwen3-8B": [0.88, 1.20]})
        est = estimate(cfg, steps=1)
        # Exactly double the snapshot prices, so exactly double the cost.
        assert est.total_usd_per_step == pytest.approx(2 * 0.066304, abs=1e-9)


class TestUnknownSamplePriceIsSurfaced:
    def test_sample_cost_is_none_not_zero(self) -> None:
        est = estimate(make_cfg(model="Qwen/Qwen3.5-9B-Base"), steps=1)
        assert est.sample_usd_per_step is None
        assert est.total_usd_per_step is None
        assert est.total_usd_per_run is None

    def test_train_cost_is_still_reported(self) -> None:
        est = estimate(make_cfg(model="Qwen/Qwen3.5-9B-Base"), steps=1)
        assert est.train_usd_per_step == pytest.approx(0.0896 * 1.463, abs=1e-9)

    def test_render_says_unknown(self) -> None:
        est = estimate(make_cfg(model="Qwen/Qwen3.5-9B-Base"), steps=1)
        assert "unknown" in render_to_text(est.render())


class TestValidation:
    @pytest.mark.parametrize("steps", [0, -5])
    def test_non_positive_steps_rejected(self, steps: int) -> None:
        with pytest.raises(ValueError, match="steps must be positive"):
            estimate(make_cfg(), steps=steps)

    @pytest.mark.parametrize("rate", [-0.1, 1.5])
    def test_out_of_range_cache_rate_rejected(self, rate: float) -> None:
        with pytest.raises(ValueError, match="prompt_cache_hit_rate"):
            estimate(make_cfg(prompt_cache_hit_rate=rate), steps=1)

    def test_non_positive_lengths_rejected(self) -> None:
        with pytest.raises(ValueError, match="completion_tokens"):
            estimate(make_cfg(completion_tokens=0), steps=1)

    def test_non_positive_group_size_rejected(self) -> None:
        with pytest.raises(ValueError, match="group_size"):
            estimate(make_cfg(group_size=0), steps=1)


class TestFromUsage:
    def test_prices_actual_spend(self) -> None:
        usage = TokenUsage(
            train_tokens=89_600,
            sample_tokens=44_800,
            prompt_cache_hit_tokens=0,
            num_forward_passes=1,
            num_backward_passes=1,
        )
        actual = from_usage(usage, "Qwen/Qwen3-8B")
        assert actual.total_usd_per_step == pytest.approx(0.066304, abs=1e-9)
        assert actual.passes == 2

    def test_cache_hits_reduce_the_bill(self) -> None:
        usage = TokenUsage(train_tokens=0, sample_tokens=44_800, prompt_cache_hit_tokens=15_360)
        actual = from_usage(usage, "Qwen/Qwen3-8B")
        assert actual.sample_tokens_per_step == pytest.approx(29_440.0)
        assert actual.sample_usd_per_step == pytest.approx(0.029440 * 0.60, abs=1e-9)

    def test_matches_estimate_when_the_assumptions_hold(self) -> None:
        # This is the comparison the module exists to enable: same tokens, same dollars.
        est = estimate(make_cfg(), steps=1)
        usage = TokenUsage(
            train_tokens=int(est.train_tokens_per_step),
            sample_tokens=int(est.sample_tokens_per_step),
        )
        assert from_usage(usage, "Qwen/Qwen3-8B").total_usd_per_step == pytest.approx(
            est.total_usd_per_step
        )

    def test_zero_sampling_costs_zero_even_with_an_unknown_sample_price(self) -> None:
        usage = TokenUsage(train_tokens=1_000_000, sample_tokens=0)
        actual = from_usage(usage, "Qwen/Qwen3.5-9B-Base")
        assert actual.sample_usd_per_step == 0.0
        assert actual.total_usd_per_step == pytest.approx(1.463, abs=1e-9)

    def test_sampling_on_an_unpriced_model_reports_unknown(self) -> None:
        usage = TokenUsage(train_tokens=0, sample_tokens=1000)
        assert from_usage(usage, "Qwen/Qwen3.5-9B-Base").sample_usd_per_step is None

    def test_empty_usage_is_free(self) -> None:
        actual = from_usage(TokenUsage(), "Qwen/Qwen3-8B")
        assert actual.total_usd_per_step == pytest.approx(0.0)

    def test_override_is_honoured(self) -> None:
        usage = TokenUsage(train_tokens=1_000_000)
        actual = from_usage(usage, "Qwen/Qwen3-8B", {"Qwen/Qwen3-8B": [1.0, 1.0]})
        assert actual.train_usd_per_step == pytest.approx(1.0)


def render_to_text(table: Table) -> str:
    console = Console(width=140, record=True, force_terminal=False)
    console.print(table)
    return console.export_text()


class TestRender:
    def test_contains_both_rows_and_a_total(self) -> None:
        text = render_to_text(estimate(make_cfg(), steps=10).render())
        assert "sample" in text
        assert "train" in text
        assert "total" in text
        assert "Qwen/Qwen3-8B" in text

    def test_shows_dollars(self) -> None:
        assert "$" in render_to_text(estimate(make_cfg(), steps=1).render())

    def test_custom_title(self) -> None:
        text = render_to_text(estimate(make_cfg(), steps=1).render(title="my sweep"))
        assert "my sweep" in text

    def test_usage_renders(self) -> None:
        text = render_to_text(from_usage(TokenUsage(train_tokens=5), "Qwen/Qwen3-8B").render())
        assert "total" in text
