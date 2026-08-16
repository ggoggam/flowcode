"""``flowcode-train`` — compose a config and run the loop.

    flowcode-train                                     # the shipped defaults
    flowcode-train objective=subtb env=humaneval       # swap groups
    flowcode-train train=smoke                         # five cheap steps
    flowcode-train -m objective=tb,subtb,vargrad       # a sweep

Every override is validated during composition against the dataclasses in
:mod:`flowcode.config`, so a typo fails here rather than ten minutes into a paid run.
:func:`flowcode.train.validate_train_config` then rejects combinations that compose fine
but cannot mean what they say — chiefly ``on_policy_only`` together with replay.
"""

from __future__ import annotations

import asyncio
from typing import cast

import hydra
from omegaconf import DictConfig

from flowcode.cli import CONFIG_PATH
from flowcode.config import RootConfig
from flowcode.train import train

__all__ = ["main", "run"]


def run(cfg: RootConfig) -> None:
    """Run one training job.

    Split out of :func:`main` so tests can drive it with a composed config without going
    through ``@hydra.main`` (which owns ``sys.argv`` and creates an output directory).

    Args:
        cfg: The composed run config.
    """
    asyncio.run(train(cfg))


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg: DictConfig) -> None:
    """Console-script entry point (``flowcode-train``).

    Args:
        cfg: Hydra's composed config. Typed ``DictConfig`` because that is what Hydra
            passes; the structured schema is applied to it, so attribute access and
            :func:`omegaconf.OmegaConf.to_object` both see :class:`RootConfig` fields.
            The cast records that — a composed ``DictConfig`` carrying the ``RootConfig``
            schema is a ``RootConfig`` for every purpose except its runtime class.
    """
    run(cast(RootConfig, cfg))


if __name__ == "__main__":  # pragma: no cover - console script
    main()
