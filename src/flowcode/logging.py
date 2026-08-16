"""Where a step's metrics go: a rich table, a wandb run, or nowhere.

Three backends behind one four-method protocol (:class:`RunLogger`). The training loop
never branches on which one it has, so ``logger=none`` is a genuinely free no-op and the
tests can assert on a recorded list instead of parsing a terminal.

The rich table is deliberately narrow. A GFlowNet run has dozens of metrics and exactly six
that tell you whether it is working:

============  =============================================================================
``loss``      The balance residual. Should fall, but a *rising* loss with a rising log R is
              normal early on — the residual grows as the reward spreads out.
``logR``      Mean and max ``log R`` over the batch. This is the number that matters: it is
              the target distribution, not a proxy for it.
``pass``      Fraction of the batch that passed every test.
``logZ``      The partition-function estimate, learned (``tb``/``db``/``subtb``) or
              in-batch (``vargrad``). Should converge to ``log sum_x R(x)``; a ``logZ``
              that runs away while the loss stalls means the flow LR is too high.
``tokens``    Cumulative billed tokens.
``$``         Running spend. The one number that is not recoverable after the fact.
============  =============================================================================

Everything else still reaches wandb, and the rich logger prints the remainder of the dict
only when a step's metrics change shape (a new key appearing is worth seeing once).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any, Final, Protocol

from rich.console import Console
from rich.table import Table

__all__ = [
    "LOGGER_KINDS",
    "NullLogger",
    "RichLogger",
    "RunLogger",
    "WandbLogger",
    "make_logger",
]

logger = logging.getLogger(__name__)

LOGGER_KINDS: Final[tuple[str, ...]] = ("rich", "wandb", "none")
"""Legal values of ``cfg.logger``."""

_COLUMNS: Final[tuple[tuple[str, tuple[str, ...], str], ...]] = (
    # (heading, candidate metric keys in priority order, format spec)
    ("step", ("step",), "{:>5.0f}"),
    ("loss", ("loss",), "{:>10.4f}"),
    ("logR mean", ("log_reward_mean",), "{:>10.3f}"),
    ("logR max", ("log_reward_max",), "{:>9.3f}"),
    ("pass", ("pass_rate",), "{:>6.2%}"),
    ("logZ", ("log_z_mean", "log_z_estimate_mean"), "{:>8.3f}"),
    ("tokens", ("tokens_total",), "{:>11,.0f}"),
    ("$", ("usd_total",), "{:>8.4f}"),
)


class RunLogger(Protocol):
    """What the training loop needs from a metrics sink."""

    def log(self, step: int, metrics: Mapping[str, float]) -> None:
        """Record one step's metrics."""
        ...

    def log_text(self, message: str) -> None:
        """Record a one-off human-readable line (eval summaries, checkpoint paths)."""
        ...

    def log_object(self, renderable: Any) -> None:
        """Record something only a console can show, e.g. a :class:`rich.table.Table`."""
        ...

    def close(self) -> None:
        """Flush and release the sink. Safe to call twice."""
        ...


class NullLogger:
    """Drops everything. The ``logger=none`` backend, and the tests' default."""

    def log(self, step: int, metrics: Mapping[str, float]) -> None:
        """Ignore the metrics."""
        return None

    def log_text(self, message: str) -> None:
        """Ignore the message."""
        return None

    def log_object(self, renderable: Any) -> None:
        """Ignore the renderable."""
        return None

    def close(self) -> None:
        """Nothing to release."""
        return None


class RichLogger:
    """Prints one table row per step to a :class:`rich.console.Console`.

    The header is printed once, on the first :meth:`log`, and every subsequent row uses the
    same fixed column widths — a fresh ``Table`` per row would re-fit its columns to that
    row's contents and the numbers would wander across the screen.

    Args:
        console: Console to print to. A fresh one is created when omitted; the tests pass
            a ``Console(file=StringIO())``.
    """

    def __init__(self, console: Console | None = None) -> None:
        self.console = Console() if console is None else console
        self._header_printed = False
        self._seen_keys: set[str] = set()

    @staticmethod
    def _cell(metrics: Mapping[str, float], keys: Sequence[str], fmt: str) -> str:
        for key in keys:
            if key in metrics:
                try:
                    return fmt.format(float(metrics[key]))
                except (TypeError, ValueError):  # pragma: no cover - defensive
                    return str(metrics[key])
        return "-"

    def _table(self, show_header: bool) -> Table:
        table = Table(show_header=show_header, header_style="bold", box=None, pad_edge=False)
        for heading, _keys, fmt in _COLUMNS:
            # Width from the format spec so the header and every row line up forever.
            width = max(len(heading), len(fmt.format(0.0)))
            table.add_column(heading, justify="right", width=width, no_wrap=True)
        return table

    def log(self, step: int, metrics: Mapping[str, float]) -> None:
        """Print one row.

        Args:
            step: Optimiser step, printed in the first column.
            metrics: Flat metric dict. Missing columns render as ``-``.
        """
        row_metrics = {"step": float(step), **{k: v for k, v in metrics.items()}}
        table = self._table(show_header=not self._header_printed)
        table.add_row(*(self._cell(row_metrics, keys, fmt) for _h, keys, fmt in _COLUMNS))
        self.console.print(table)
        self._header_printed = True

        new_keys = {k for k in metrics if k not in self._seen_keys}
        if new_keys and self._seen_keys:
            # A key appearing mid-run means something changed shape; say so once.
            self.console.print(f"[dim]new metrics: {', '.join(sorted(new_keys))}[/dim]")
        self._seen_keys |= set(metrics)

    def log_text(self, message: str) -> None:
        """Print a plain line."""
        self.console.print(message)

    def log_object(self, renderable: Any) -> None:
        """Print a rich renderable (a cost table, typically)."""
        self.console.print(renderable)

    def close(self) -> None:
        """Nothing to release; the console is not owned exclusively."""
        return None


class WandbLogger:
    """Mirrors metrics into a Weights & Biases run.

    ``wandb`` is an optional extra (``uv sync --extra wandb``), so it is imported inside
    the constructor: importing this module must not require it, and a missing install has
    to say what to run rather than raise ``ModuleNotFoundError`` from three frames down.

    Args:
        config: The composed run config as a plain dict, recorded as the run's config.
        project: wandb project name.
        name: Optional run name.
        echo: Also print the rich table locally. On by default — a remote dashboard is not
            a substitute for seeing the run in the terminal you launched it from.

    Raises:
        ImportError: If the extra is not installed.
    """

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        project: str = "flowcode",
        name: str | None = None,
        echo: bool = True,
    ) -> None:
        try:
            import wandb
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "logger=wandb needs the optional wandb extra, which is not installed. "
                "Install it with `uv sync --extra wandb`, or run with logger=rich."
            ) from exc
        self._wandb = wandb
        self._run = wandb.init(project=project, name=name, config=dict(config or {}))
        self._echo = RichLogger() if echo else None

    def log(self, step: int, metrics: Mapping[str, float]) -> None:
        """Send the full metric dict upstream, and optionally print it locally."""
        self._wandb.log(dict(metrics), step=step)
        if self._echo is not None:
            self._echo.log(step, metrics)

    def log_text(self, message: str) -> None:
        """Print locally; wandb keeps the stdout capture of the run anyway."""
        if self._echo is not None:
            self._echo.log_text(message)
        else:  # pragma: no cover - only when echo is disabled
            logger.info("%s", message)

    def log_object(self, renderable: Any) -> None:
        """Print locally. Rich renderables have no wandb equivalent worth inventing."""
        if self._echo is not None:
            self._echo.log_object(renderable)

    def close(self) -> None:
        """Finish the wandb run."""
        if self._run is not None:
            self._run.finish()
            self._run = None


def make_logger(
    kind: str,
    *,
    config: Mapping[str, Any] | None = None,
    project: str = "flowcode",
    name: str | None = None,
) -> RunLogger:
    """Build the configured metrics sink.

    Args:
        kind: ``"rich"``, ``"wandb"`` or ``"none"`` — i.e. ``cfg.logger``.
        config: Composed run config, recorded by the wandb backend and ignored by the
            others.
        project: wandb project name.
        name: wandb run name.

    Returns:
        A :class:`RunLogger`.

    Raises:
        ValueError: On an unknown kind.
        ImportError: If ``wandb`` is requested but not installed.
    """
    normalised = kind.strip().lower()
    if normalised == "rich":
        return RichLogger()
    if normalised == "none":
        return NullLogger()
    if normalised == "wandb":
        return WandbLogger(config=config, project=project, name=name)
    raise ValueError(f"unknown logger {kind!r}; expected one of {list(LOGGER_KINDS)}")
