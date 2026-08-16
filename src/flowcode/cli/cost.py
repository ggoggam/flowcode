"""``flowcode-cost`` — what will this run cost me?

    flowcode-cost                                        # the shipped defaults
    flowcode-cost train=smoke                            # price the cheap profile
    flowcode-cost model=gpt-oss-20b train.group_size=16
    flowcode-cost train.on_policy_only=true              # the halved training pass
    flowcode-cost '+cost.price_overrides={Qwen/Qwen3-8B: [0.50, 0.65]}'

Composes exactly the same config ``flowcode-train`` would, then prices it and exits. It
opens no network connection and reads no credential: ``TINKER_API_KEY`` being unset is a
supported way to run this command, because the whole point is to be able to ask the
question before you have decided to spend anything.

It also leaves nothing on disk by default: ``hydra.run.dir=.`` and
``hydra.output_subdir=null`` suppress the timestamped run directory and its config dump,
and ``hydra/job_logging=disabled`` suppresses the ``cost.log`` file Hydra's default job
logging would otherwise drop in the working directory. A read-only query should not
litter. Pass any of the three explicitly on the command line to get that behaviour back.
"""

from __future__ import annotations

import sys
from typing import cast

import hydra
from omegaconf import DictConfig
from rich.console import Console

from flowcode.cli import CONFIG_PATH
from flowcode.config import RootConfig
from flowcode.cost import estimate

__all__ = ["main", "run"]

_LEAVE_NO_TRACE: tuple[str, ...] = (
    "hydra.run.dir=.",
    "hydra.output_subdir=null",
    "hydra/job_logging=disabled",
)
"""Overrides injected unless the caller set the same key themselves."""


def run(cfg: RootConfig, console: Console | None = None) -> None:
    """Print the cost table for a composed config.

    Args:
        cfg: The composed run config.
        console: Where to print; a fresh :class:`rich.console.Console` when omitted.

    Raises:
        KeyError: If the model has no known price and none was supplied via
            ``cost.price_overrides``.
        ValueError: If a config quantity makes the estimate meaningless (non-positive
            lengths, a cache hit rate outside ``[0, 1]``).
    """
    out = Console() if console is None else console
    report = estimate(cfg)
    out.print(report.render())
    passes = "1 (on_policy_only: the forward() oracle is skipped)" if report.passes == 1 else "2"
    out.print(
        f"model={report.model}  steps={report.steps}  "
        f"samples/step={report.samples_per_step} "
        f"({cfg.train.groups_per_step} groups x {cfg.train.group_size})  "
        f"assumed seq_len={report.seq_len} "
        f"({cfg.cost.prompt_tokens} prompt + {cfg.cost.completion_tokens} completion)  "
        f"train passes/step={passes}"
    )
    out.print(
        "[dim]Sequence lengths are assumptions from conf/cost/*.yaml, not measurements. "
        "The training loop reports what a run actually spent at the end.[/dim]"
    )
    if getattr(cfg.backend, "kind", "tinker") != "tinker":
        # The price table prices Tinker's API. On your own hardware the token counts above
        # are still the right throughput figures, but the dollars are not a bill anyone
        # will send — and quietly leaving them on screen invites planning a run around a
        # number that means nothing.
        out.print(
            f"[bold yellow]backend={cfg.backend.kind}: the dollar columns above do not "
            "apply.[/bold yellow] They price Tinker's API; a local run's cost is device "
            "time, which this estimator does not model. Read the token counts as "
            "throughput and ignore the money."
        )


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def _entry(cfg: DictConfig) -> None:
    """The Hydra-decorated body. See :func:`main` for why it is wrapped."""
    run(cast(RootConfig, cfg))


def main() -> None:
    """Console-script entry point (``flowcode-cost``).

    Wraps the decorated function so the "leave no trace" overrides can be injected into
    ``sys.argv`` first — ``@hydra.main`` reads ``sys.argv`` itself and offers no other
    place to set a default override. Anything the user passed for the same key wins,
    because ours is skipped entirely when theirs is present.
    """
    argv = sys.argv[1:]
    injected = [
        override
        for override in _LEAVE_NO_TRACE
        if not any(arg.split("=", 1)[0] == override.split("=", 1)[0] for arg in argv)
    ]
    sys.argv = [sys.argv[0], *injected, *argv]
    _entry()


if __name__ == "__main__":  # pragma: no cover - console script
    main()
