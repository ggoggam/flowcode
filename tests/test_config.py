"""Hydra composition against the real ``conf/`` tree.

Structured configs have exactly one interesting failure mode: the dataclass and the YAML
drift apart. A field renamed in YAML and not in the schema, or vice versa, either silently
disappears from the composed config or blows up at composition time in a way nobody sees
until a run starts. So every test here composes the *actual* files in ``conf/`` rather
than a fixture — the whole value of the exercise is that it breaks when someone edits the
YAML without editing :mod:`flowcode.config`.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.errors import ConfigCompositionException
from omegaconf import DictConfig, OmegaConf

from flowcode.config import (
    API_KEY_ENV_VAR,
    PROJECT_ID_ENV_VAR,
    CostConfig,
    ModelConfig,
    ReplayConfig,
    RootConfig,
    TrainConfig,
    get_api_key,
    get_project_id,
)
from flowcode.cost import ModelPrice, resolve_price

CONF_DIR = Path(__file__).resolve().parents[1] / "conf"

MODEL_NAMES = sorted(p.stem for p in (CONF_DIR / "model").glob("*.yaml"))
TRAIN_NAMES = sorted(p.stem for p in (CONF_DIR / "train").glob("*.yaml"))
COST_NAMES = sorted(p.stem for p in (CONF_DIR / "cost").glob("*.yaml"))
OBJECTIVE_NAMES = sorted(p.stem for p in (CONF_DIR / "objective").glob("*.yaml"))
ENV_NAMES = sorted(p.stem for p in (CONF_DIR / "env").glob("*.yaml"))


def compose_cfg(overrides: list[str] | None = None) -> DictConfig:
    """Compose ``conf/config.yaml`` the way the CLI will, with a clean Hydra global."""
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        cfg = compose(config_name="config", overrides=overrides or [])
    assert isinstance(cfg, DictConfig)
    return cfg


def yaml_keys(group: str, name: str) -> set[str]:
    """Top-level keys actually present in one group YAML."""
    raw = OmegaConf.load(CONF_DIR / group / f"{name}.yaml")
    assert isinstance(raw, DictConfig)
    return {str(key) for key in raw}


class TestSchemaIsApplied:
    """The composed nodes must carry our dataclass types, or nothing is being validated."""

    def test_root_and_group_types(self) -> None:
        cfg = compose_cfg()
        assert OmegaConf.get_type(cfg) is RootConfig
        assert OmegaConf.get_type(cfg.model) is ModelConfig
        assert OmegaConf.get_type(cfg.train) is TrainConfig
        assert OmegaConf.get_type(cfg.train.replay) is ReplayConfig
        assert OmegaConf.get_type(cfg.cost) is CostConfig

    def test_defaults_list_survives_self(self) -> None:
        # RootConfig's group fields are MISSING precisely so that `- _self_` (which is
        # LAST in conf/config.yaml) does not overwrite the selected groups with schema
        # defaults. If someone gives them default_factory values, this is what breaks.
        cfg = compose_cfg()
        assert cfg.model.name == "Qwen/Qwen3-8B"
        assert cfg.env.name == "mbpp"
        assert cfg.objective.name == "vargrad"
        assert cfg.train.steps == 1000
        assert cfg.cost.prompt_tokens == 300

    def test_nothing_is_left_mandatory_missing(self) -> None:
        # to_object() raises MissingMandatoryValue on any remaining `???`, and returns a
        # real RootConfig (not a dict) precisely because the schema was applied.
        obj = OmegaConf.to_object(compose_cfg())
        assert isinstance(obj, RootConfig)
        assert isinstance(obj.model, ModelConfig)
        assert isinstance(obj.train, TrainConfig)
        assert isinstance(obj.cost, CostConfig)


class TestNoDriftBetweenYamlAndSchema:
    """Every key in every YAML must exist in the schema, and vice versa."""

    @pytest.mark.parametrize("name", MODEL_NAMES)
    def test_model_group_matches(self, name: str) -> None:
        cfg = compose_cfg([f"model={name}"])
        assert yaml_keys("model", name) == set(cfg.model.keys())

    @pytest.mark.parametrize("name", TRAIN_NAMES)
    def test_train_group_matches(self, name: str) -> None:
        cfg = compose_cfg([f"train={name}"])
        assert yaml_keys("train", name) == set(cfg.train.keys())
        replay_yaml = OmegaConf.load(CONF_DIR / "train" / f"{name}.yaml")
        assert isinstance(replay_yaml, DictConfig)
        assert set(replay_yaml.replay.keys()) == set(cfg.train.replay.keys())

    @pytest.mark.parametrize("name", COST_NAMES)
    def test_cost_group_matches(self, name: str) -> None:
        cfg = compose_cfg([f"cost={name}"])
        assert yaml_keys("cost", name) == set(cfg.cost.keys())

    def test_root_keys_match(self) -> None:
        raw = OmegaConf.load(CONF_DIR / "config.yaml")
        assert isinstance(raw, DictConfig)
        own_keys = set(raw.keys()) - {"defaults"}
        group_keys = {"model", "objective", "env", "train", "cost", "backend", "sampler"}
        assert own_keys | group_keys == set(compose_cfg().keys())

    @pytest.mark.parametrize("name", MODEL_NAMES)
    def test_every_model_composes_and_types_cleanly(self, name: str) -> None:
        cfg = compose_cfg([f"model={name}"])
        model = OmegaConf.to_object(cfg.model)
        assert isinstance(model, ModelConfig)
        assert isinstance(model.name, str) and "/" in model.name
        assert isinstance(model.max_context, int)
        assert isinstance(model.lora_rank, int)

    @pytest.mark.parametrize("name", TRAIN_NAMES)
    def test_every_train_composes_and_types_cleanly(self, name: str) -> None:
        train = OmegaConf.to_object(compose_cfg([f"train={name}"]).train)
        assert isinstance(train, TrainConfig)
        assert isinstance(train.replay, ReplayConfig)
        assert isinstance(train.policy_lr, float)
        assert isinstance(train.on_policy_only, bool)

    @pytest.mark.parametrize("name", OBJECTIVE_NAMES)
    def test_every_objective_composes_with_a_target(self, name: str) -> None:
        # objective/env are typed `Any` on purpose (hydra instantiates them), so all we
        # can assert is that the group still carries the contract instantiate() needs.
        cfg = compose_cfg([f"objective={name}"])
        assert cfg.objective._target_.startswith("flowcode.objectives.")
        assert cfg.objective.name == name

    @pytest.mark.parametrize("name", ENV_NAMES)
    def test_every_env_composes_with_a_target(self, name: str) -> None:
        cfg = compose_cfg([f"env={name}"])
        assert cfg.env._target_.startswith("flowcode.envs.")
        assert cfg.env.name == name


class TestOverridesAreValidated:
    def test_scalar_override_applies(self) -> None:
        # Also proves the dataclasses are NOT frozen: a frozen schema makes OmegaConf
        # mark the node read-only and every override below raises ReadonlyConfigError.
        assert compose_cfg(["train.group_size=16"]).train.group_size == 16

    def test_nested_override_applies(self) -> None:
        assert compose_cfg(["train.replay.capacity=42"]).train.replay.capacity == 42

    def test_typo_in_key_is_rejected(self) -> None:
        with pytest.raises(ConfigCompositionException):
            compose_cfg(["train.grop_size=16"])

    def test_wrong_type_is_rejected(self) -> None:
        with pytest.raises(ConfigCompositionException):
            compose_cfg(["train.group_size=not_an_int"])

    def test_typo_in_model_key_is_rejected(self) -> None:
        with pytest.raises(ConfigCompositionException):
            compose_cfg(["model.lora_rnk=8"])

    def test_unknown_group_option_is_rejected(self) -> None:
        with pytest.raises(ConfigCompositionException):
            compose_cfg(["objective=does_not_exist"])

    def test_price_overrides_needs_a_plus_for_a_new_key(self) -> None:
        # Hydra puts the composed config in struct mode, so any key not already present
        # requires `+` regardless of how the field is typed. Documented in CostConfig.
        cfg = compose_cfg(["+cost.price_overrides={acme/brand-new: [0.5, 0.7]}"])
        assert list(cfg.cost.price_overrides["acme/brand-new"]) == [0.5, 0.7]

    def test_composed_price_override_reaches_the_estimator(self) -> None:
        # The realistic path: an override arrives as a ListConfig, not a list, which is
        # why flowcode.cost matches on the Sequence ABC.
        cfg = compose_cfg(["+cost.price_overrides={Qwen/Qwen3-8B: [1.0, 2.0]}"])
        assert resolve_price("Qwen/Qwen3-8B", cfg.cost.price_overrides) == ModelPrice(1.0, 2.0)

    def test_composed_null_sample_price_stays_none(self) -> None:
        cfg = compose_cfg(["+cost.price_overrides={acme/train-only: [1.0, null]}"])
        assert resolve_price("acme/train-only", cfg.cost.price_overrides).sample is None

    def test_composed_mapping_override_reaches_the_estimator(self) -> None:
        cfg = compose_cfg(["+cost.price_overrides={acme/x: {train: 3.0, sample: 4.0}}"])
        assert resolve_price("acme/x", cfg.cost.price_overrides) == ModelPrice(3.0, 4.0)

    def test_smoke_train_group_is_on_policy(self) -> None:
        # The smoke profile exists to be cheap; on_policy_only halves the train tokens,
        # and replay must be off because the two are incompatible.
        cfg = compose_cfg(["train=smoke"])
        assert cfg.train.on_policy_only is True
        assert cfg.train.replay.enabled is False


class TestSecretsBypassHydra:
    def test_api_key_read_from_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(API_KEY_ENV_VAR, "tk-test-123")
        assert get_api_key() == "tk-test-123"

    def test_api_key_is_stripped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(API_KEY_ENV_VAR, "  tk-test-123\n")
        assert get_api_key() == "tk-test-123"

    @pytest.mark.parametrize("value", [None, "", "   "])
    def test_missing_api_key_names_the_env_file(
        self, monkeypatch: pytest.MonkeyPatch, value: str | None
    ) -> None:
        if value is None:
            monkeypatch.delenv(API_KEY_ENV_VAR, raising=False)
        else:
            monkeypatch.setenv(API_KEY_ENV_VAR, value)
        with pytest.raises(RuntimeError) as excinfo:
            get_api_key()
        message = str(excinfo.value)
        assert ".env" in message
        assert API_KEY_ENV_VAR in message

    def test_project_id_is_optional(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(PROJECT_ID_ENV_VAR, raising=False)
        assert get_project_id() is None

    def test_project_id_read_when_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(PROJECT_ID_ENV_VAR, "proj-42")
        assert get_project_id() == "proj-42"

    def test_blank_project_id_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(PROJECT_ID_ENV_VAR, "  ")
        assert get_project_id() is None

    def test_no_secret_ever_lands_in_the_composed_config(self) -> None:
        # A composed config is dumped to outputs/.../.hydra/config.yaml on every run.
        dumped = OmegaConf.to_yaml(compose_cfg())
        assert API_KEY_ENV_VAR.lower() not in dumped.lower()
        assert "api_key" not in dumped

    def test_environment_is_not_consulted_by_composition(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(API_KEY_ENV_VAR, "tk-should-not-appear")
        assert "tk-should-not-appear" not in OmegaConf.to_yaml(compose_cfg())
        assert os.environ[API_KEY_ENV_VAR] == "tk-should-not-appear"


def test_config_dataclasses_are_not_frozen() -> None:
    """Documented constraint, asserted so nobody "fixes" it back.

    Hydra applies command-line overrides by mutating the composed config. OmegaConf marks
    the node of a ``frozen=True`` dataclass read-only, and re-derives that flag from the
    reference type on every merge, so clearing it at store time does not help either.
    Frozen schemas therefore make every override in ``conf/config.yaml``'s own docstring
    fail with ``ReadonlyConfigError``.
    """
    # Tested by mutating instances rather than by reading __dataclass_params__: mutation
    # is the property Hydra actually needs, and a frozen dataclass raises here.
    model = ModelConfig(name="a/b", renderer="r")
    model.lora_rank = 8
    assert model.lora_rank == 8

    replay = ReplayConfig()
    replay.capacity = 5
    assert replay.capacity == 5

    train = TrainConfig()
    train.group_size = 3
    assert train.group_size == 3

    cost = CostConfig()
    cost.prompt_tokens = 7
    assert cost.prompt_tokens == 7

    root = RootConfig()
    root.seed = 11
    assert root.seed == 11
