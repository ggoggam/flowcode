"""The two Hydra apps, and the one thing they must agree on: where ``conf/`` is.

Two entry points rather than one binary with subcommands: ``@hydra.main`` takes over
``sys.argv`` to parse ``key=value`` overrides, so a subcommand word sitting in front of
those overrides is a word Hydra will try to parse as one.

Locating the config tree
------------------------
The tree lives in two different places depending on how flowcode was installed:

* **installed wheel**: ``<site-packages>/flowcode/conf``, because ``pyproject.toml``
  force-includes ``conf`` as ``flowcode/conf``;
* **source checkout**: ``<repo>/conf``, three levels above this file
  (``<repo>/src/flowcode/cli/``) and *outside* any package.

A relative ``config_path`` cannot cover both, and the reason is worth writing down because
it looks like it should. Hydra resolves a relative ``config_path`` against the declaring
file's directory **only when the task function's module is ``__main__``**; for a console
script the module is ``flowcode.cli.cost``, so
``hydra._internal.utils.compute_search_path_dir`` switches to *package*-relative
resolution, walking ``../`` up the module path and producing a ``pkg://`` search path. In a
source checkout that walk runs out of package before it reaches the repository root and
Hydra reports ``Primary config module 'conf' not found``.

An absolute path has no such ambiguity: ``compute_search_path_dir`` returns it verbatim for
both invocation styles. :func:`resolve_config_path` therefore returns an absolute directory
— the first of the two layouts that actually contains ``config.yaml`` — and
:data:`CONFIG_PATH` is that answer, computed once at import.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

__all__ = ["CONFIG_PATH", "CONFIG_PATH_CANDIDATES", "resolve_config_path"]

CONFIG_PATH_CANDIDATES: Final[tuple[str, ...]] = ("../conf", "../../../conf")
"""Relative to this package, in priority order: installed-wheel layout, then source checkout."""


def resolve_config_path(base: Path | None = None) -> str:
    """Find the ``conf/`` directory for either layout.

    Args:
        base: Directory to resolve the candidates against. Defaults to this package's
            directory. The tests pass synthetic trees to prove both layouts resolve.

    Returns:
        An absolute path to the directory containing ``config.yaml``.

    Raises:
        FileNotFoundError: If neither layout is present, naming both places looked at — an
            installed package missing its configs otherwise surfaces as a mystifying
            ``Cannot find primary config 'config'``.
    """
    root = Path(__file__).resolve().parent if base is None else Path(base).resolve()
    tried: list[str] = []
    for candidate in CONFIG_PATH_CANDIDATES:
        path = (root / candidate).resolve()
        tried.append(str(path))
        if (path / "config.yaml").is_file():
            return str(path)
    raise FileNotFoundError(
        "cannot locate the flowcode config tree; looked for config.yaml in "
        f"{tried}. In a source checkout it lives at <repo>/conf; in a wheel it is "
        "force-included at flowcode/conf (see [tool.hatch.build.targets.wheel.force-include])."
    )


CONFIG_PATH: Final[str] = resolve_config_path()
"""The absolute ``config_path`` both Hydra apps pass to ``@hydra.main``."""
