"""The two Hydra apps: where they find ``conf/``, and that ``flowcode-cost`` is free.

``flowcode-cost`` is the "how much will this cost me" command, so the interesting
assertions are negative ones: it must work with no ``TINKER_API_KEY`` in the environment,
open no connection, leave no output directory behind, and print no deprecation warnings.
Those are checked in a real subprocess, because they are all properties of a *process*
rather than of a function.

The config path gets its own tests because it is the one thing that differs between a
source checkout and an installed wheel, and it fails identically-looking ways in both.
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig
from rich.console import Console

from flowcode.cli import CONFIG_PATH, CONFIG_PATH_CANDIDATES, resolve_config_path
from flowcode.cli import cost as cost_cli
from flowcode.cli import train as train_cli
from flowcode.config import API_KEY_ENV_VAR, RootConfig

REPO_ROOT = Path(__file__).resolve().parents[1]


def compose_cfg(overrides: list[str] | None = None) -> RootConfig:
    """Compose exactly the way the CLI does, from the CLI's own config path.

    Returned as :class:`RootConfig`: what Hydra hands the app is a ``DictConfig`` carrying
    the structured schema, which is a ``RootConfig`` for every purpose but its class.
    """
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_PATH, version_base="1.3"):
        cfg = compose(config_name="config", overrides=overrides or [])
    assert isinstance(cfg, DictConfig)
    return cast(RootConfig, cfg)


def make_layout(root: Path, relative_conf: str) -> Path:
    """Build a synthetic package tree with ``conf/`` at ``relative_conf`` from the cli dir."""
    cli_dir = root / "src" / "flowcode" / "cli"
    cli_dir.mkdir(parents=True)
    conf_dir = (cli_dir / relative_conf).resolve()
    conf_dir.mkdir(parents=True, exist_ok=True)
    (conf_dir / "config.yaml").write_text("seed: 0\n", encoding="utf-8")
    return cli_dir


class TestConfigPathResolution:
    def test_wheel_layout_resolves(self, tmp_path: Path) -> None:
        # <site-packages>/flowcode/conf, next to the package's own modules.
        cli_dir = make_layout(tmp_path, "../conf")
        resolved = Path(resolve_config_path(cli_dir))
        assert resolved == (cli_dir / ".." / "conf").resolve()
        assert (resolved / "config.yaml").is_file()

    def test_source_checkout_layout_resolves(self, tmp_path: Path) -> None:
        # <repo>/conf, three levels up and outside any package.
        cli_dir = make_layout(tmp_path, "../../../conf")
        resolved = Path(resolve_config_path(cli_dir))
        assert resolved == (tmp_path / "conf").resolve()

    def test_the_wheel_layout_wins_when_both_exist(self, tmp_path: Path) -> None:
        cli_dir = make_layout(tmp_path, "../conf")
        (tmp_path / "conf").mkdir()
        (tmp_path / "conf" / "config.yaml").write_text("seed: 1\n", encoding="utf-8")
        assert Path(resolve_config_path(cli_dir)) == (cli_dir / ".." / "conf").resolve()

    def test_neither_layout_names_both_places(self, tmp_path: Path) -> None:
        cli_dir = tmp_path / "src" / "flowcode" / "cli"
        cli_dir.mkdir(parents=True)
        with pytest.raises(FileNotFoundError) as excinfo:
            resolve_config_path(cli_dir)
        message = str(excinfo.value)
        assert "config.yaml" in message
        assert "force-include" in message

    def test_a_directory_without_config_yaml_does_not_count(self, tmp_path: Path) -> None:
        cli_dir = tmp_path / "src" / "flowcode" / "cli"
        cli_dir.mkdir(parents=True)
        (tmp_path / "src" / "flowcode" / "conf").mkdir()
        with pytest.raises(FileNotFoundError):
            resolve_config_path(cli_dir)

    def test_candidates_cover_both_layouts(self) -> None:
        assert CONFIG_PATH_CANDIDATES == ("../conf", "../../../conf")

    def test_the_live_config_path_is_absolute_and_real(self) -> None:
        # Absolute on purpose: for a console script the task function's module is not
        # __main__, so Hydra would resolve a relative config_path as a PACKAGE path and
        # never find a conf/ tree that sits outside the package.
        assert os.path.isabs(CONFIG_PATH)
        assert Path(CONFIG_PATH) == REPO_ROOT / "conf"
        assert (Path(CONFIG_PATH) / "config.yaml").is_file()

    def test_hydra_composes_from_it(self) -> None:
        cfg = compose_cfg()
        assert cfg.model.name == "Qwen/Qwen3-8B"
        assert cfg.train.steps == 1000


class TestCostCommand:
    def render(self, overrides: list[str] | None = None) -> str:
        console = Console(file=io.StringIO(), width=120, no_color=True)
        cost_cli.run(compose_cfg(overrides), console=console)
        stream = console.file
        assert isinstance(stream, io.StringIO)
        return stream.getvalue()

    def test_it_renders_a_priced_table(self) -> None:
        output = self.render()
        assert "Qwen/Qwen3-8B" in output
        assert "sample" in output
        assert "train" in output
        assert "$" in output

    def test_it_reports_the_run_shape(self) -> None:
        output = self.render(["train=smoke"])
        assert "steps=5" in output
        assert "samples/step=8" in output

    def test_on_policy_only_shows_the_single_training_pass(self) -> None:
        output = self.render(["train.on_policy_only=true"])
        assert "train passes/step=1" in output
        assert "forward() oracle is skipped" in output

    def test_a_price_override_reaches_the_table(self) -> None:
        output = self.render(["+cost.price_overrides={Qwen/Qwen3-8B: [100.0, 200.0]}"])
        # 64 samples x 700 tokens x $200/Mtok, with the shipped 0.8 cache hit rate.
        assert "$5.8880" in output

    def test_an_unpriced_model_says_unknown_not_zero(self) -> None:
        output = self.render(["model=qwen3.5-9b-base"])
        assert "unknown" in output

    def test_it_needs_no_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(API_KEY_ENV_VAR, raising=False)
        assert "$" in self.render()


class TestCostArgvInjection:
    def _run_main(self, monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> list[str]:
        seen: list[str] = []

        def fake_entry() -> None:
            seen.extend(sys.argv)

        monkeypatch.setattr(cost_cli, "_entry", fake_entry)
        monkeypatch.setattr(sys, "argv", ["flowcode-cost", *argv])
        cost_cli.main()
        return seen

    def test_the_output_directory_is_suppressed_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A read-only query should not leave a timestamped folder of logs behind.
        argv = self._run_main(monkeypatch, [])
        assert "hydra.run.dir=." in argv
        assert "hydra.output_subdir=null" in argv
        # ...nor the cost.log that Hydra's default job logging drops in the cwd.
        assert "hydra/job_logging=disabled" in argv

    def test_user_overrides_are_preserved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        argv = self._run_main(monkeypatch, ["train=smoke", "model=gpt-oss-20b"])
        assert argv[-2:] == ["train=smoke", "model=gpt-oss-20b"]

    def test_an_explicit_run_dir_is_not_overridden(self, monkeypatch: pytest.MonkeyPatch) -> None:
        argv = self._run_main(monkeypatch, ["hydra.run.dir=/tmp/mine"])
        assert argv.count("hydra.run.dir=.") == 0
        assert "hydra.run.dir=/tmp/mine" in argv


class TestCostAsASubprocess:
    """The properties that only a real process can demonstrate."""

    def _run(self, tmp_path: Path, *overrides: str) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ)
        env.pop(API_KEY_ENV_VAR, None)
        env.pop("TINKER_PROJECT_ID", None)
        return subprocess.run(
            [
                sys.executable,
                # Any Hydra deprecation warning becomes a hard failure here.
                "-W",
                "error::DeprecationWarning",
                "-W",
                "error::UserWarning",
                "-c",
                "from flowcode.cli.cost import main; main()",
                *overrides,
            ],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )

    def test_it_runs_clean_without_an_api_key(self, tmp_path: Path) -> None:
        result = self._run(tmp_path, "train=smoke")
        assert result.returncode == 0, result.stderr
        assert "$" in result.stdout

    def test_it_emits_no_warnings(self, tmp_path: Path) -> None:
        result = self._run(tmp_path)
        assert "Warning" not in result.stderr, result.stderr
        assert "deprecated" not in result.stderr.lower(), result.stderr

    def test_it_leaves_nothing_behind(self, tmp_path: Path) -> None:
        self._run(tmp_path)
        assert not (tmp_path / "outputs").exists()
        assert not (tmp_path / ".hydra").exists()
        assert list(tmp_path.iterdir()) == [], "flowcode-cost wrote files into the cwd"

    def test_a_bad_override_fails_loudly(self, tmp_path: Path) -> None:
        result = self._run(tmp_path, "train.grop_size=4")
        assert result.returncode != 0
        assert "grop_size" in result.stderr


class TestTrainCommand:
    def test_run_hands_the_config_to_the_training_loop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[Any] = []

        async def fake_train(cfg: Any) -> None:
            seen.append(cfg)

        monkeypatch.setattr(train_cli, "train", fake_train)
        cfg = compose_cfg(["train=smoke"])
        train_cli.run(cfg)
        assert seen == [cfg]

    def test_the_incoherent_config_is_refused_through_the_cli(self) -> None:
        # Composition allows it; the loop must not.
        from flowcode.train import validate_train_config

        cfg = compose_cfg(["train.on_policy_only=true", "train.replay.enabled=true"])
        with pytest.raises(ValueError, match="on_policy_only"):
            validate_train_config(cfg)

    def test_both_entry_points_are_hydra_apps(self) -> None:
        # @hydra.main wraps with functools.wraps, so __wrapped__ is the tell.
        assert hasattr(train_cli.main, "__wrapped__")
        assert hasattr(cost_cli._entry, "__wrapped__")

    def test_the_console_scripts_point_at_these_functions(self) -> None:
        pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert 'flowcode-train = "flowcode.cli.train:main"' in pyproject
        assert 'flowcode-cost = "flowcode.cli.cost:main"' in pyproject
