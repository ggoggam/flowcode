"""Run model-generated Python in a subprocess with limits.

WHAT THIS IS
------------
A safety net against *accidents*. Training a coding model produces a stream of programs
that were never meant to be correct: infinite loops, ``while True: xs.append(0)``,
``open("../../train.py", "w")``, a stray ``import os; os.system("rm -rf ~")`` copied out
of a StackOverflow answer the model half-remembers. This module stops those from taking
down the training run.

WHAT THIS IS NOT
----------------
A security boundary. It does not contain adversarial code and must not be described as
if it does. Concretely, a program that *wants* to escape can still:

* read anything the training user can read — ``~/.ssh``, ``~/.aws``, the source tree
  (the working directory is fresh, but the filesystem is not namespaced);
* write anywhere the user can write, up to ``RLIMIT_FSIZE`` per file;
* open sockets and talk to the network, including exfiltrating whatever it just read;
* spoof this module's structured output by writing to file descriptor 1 directly;
* on macOS, allocate as much memory as it likes (see the caveat below).

Real isolation needs kernel-level enforcement that a ``subprocess`` call cannot provide:
an OCI container with a read-only rootfs and no network namespace, gVisor, seccomp-bpf
syscall filtering, or a disposable VM. All of that is out of scope here and deliberately
not simulated — a half-built jail that is described as a jail is worse than an honest
fence, because it invites running genuinely untrusted code through it.

WHAT IS ENFORCED
----------------
* **Wall-clock timeout with a process-group kill.** The child is started with
  ``start_new_session=True`` and killed with :func:`os.killpg`. A bare ``proc.kill()``
  reaps the child and leaks its grandchildren, which then hold the output pipes open and
  keep burning CPU for the rest of the run.
* **rlimits** applied in ``preexec_fn`` between fork and exec: ``RLIMIT_CPU`` (a
  backstop for the wall clock that survives a process that ignores SIGTERM),
  ``RLIMIT_AS`` (address space), ``RLIMIT_NPROC`` (fork bombs), ``RLIMIT_FSIZE`` (disk
  bombs), ``RLIMIT_CORE = 0`` (no multi-GB core dumps from a segfaulting C extension).
* **A fresh temporary working directory** per execution, deleted afterwards, so relative
  writes land somewhere harmless.
* **A scrubbed environment.** The child gets a dict built from scratch — never
  ``os.environ`` — so ``TINKER_API_KEY``, ``WANDB_API_KEY``, ``AWS_*`` and friends are
  not inherited. A model that prints ``os.environ`` gets a handful of boring strings.
* **Capped output.** stdout/stderr go to files, not pipes, and only the first
  ``max_output_bytes`` of each are read back. An unbounded ``print`` loop cannot OOM the
  *parent*; without this, ``communicate()`` happily buffers gigabytes in the trainer.

macOS CAVEAT
------------
``RLIMIT_AS`` is not usefully enforceable on Darwin: setting it either fails outright or
prevents the interpreter from starting, because the CPython process reserves a large
virtual address space up front regardless of how little it commits. This module detects
Darwin and skips the limit rather than crashing, reports the fact in
:data:`MEMORY_LIMIT_SUPPORTED` and in :attr:`SandboxResult.limits_applied`, and leaves
the wall-clock timeout as the only defence against a memory bomb (the OS will start
swapping and the timeout will fire). Run the trainer on Linux if the memory limit
matters.

``preexec_fn`` is documented as unsafe in a multithreaded parent: it runs Python code
between ``fork`` and ``exec``, where another thread may hold a lock that will never be
released. :func:`run_python` is called from a thread pool by
:mod:`flowcode.envs.code_exec`, so the hook here is kept to a tight loop over a
precomputed list of ``resource.setrlimit`` calls with no imports, no allocation-heavy
work and no logging.
"""

from __future__ import annotations

import contextlib
import os
import platform
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

__all__ = [
    "DEFAULT_MAX_OUTPUT_BYTES",
    "MEMORY_LIMIT_SUPPORTED",
    "SandboxResult",
    "run_python",
    "scrubbed_environ",
]

MEMORY_LIMIT_SUPPORTED: Final[bool] = platform.system() != "Darwin"
"""False on macOS, where ``RLIMIT_AS`` cannot be applied without breaking the child."""

DEFAULT_MAX_OUTPUT_BYTES: Final[int] = 64 * 1024
"""How much of each stream is read back into the parent. The rest stays on disk and dies
with the temp directory."""

DEFAULT_MAX_PROCESSES: Final[int] = 64
"""``RLIMIT_NPROC`` soft limit. The child itself is already forked when this is applied,
so the only effect is that *it* cannot spawn a hundred more."""

_SAFE_PATH: Final[str] = "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
_MIN_FSIZE_BYTES: Final[int] = 16 * 1024 * 1024
_REAP_TIMEOUT_SECONDS: Final[float] = 5.0
_SCRIPT_NAME: Final[str] = "_flowcode_main.py"


@dataclass(frozen=True)
class SandboxResult:
    """Everything the parent learns about one execution.

    Args:
        returncode: Child exit status, or ``None`` if it had to be killed. Negative
            values are ``-signum`` in the usual POSIX convention.
        stdout: First ``max_output_bytes`` of stdout, decoded with ``errors="replace"``.
        stderr: Same for stderr.
        timed_out: True if the wall-clock budget was exhausted and the process group
            was killed.
        duration_seconds: Wall time, for diagnostics only. Never feed this into a
            reward — it is the one number in this dataclass that is not reproducible.
        stdout_truncated: True if stdout was longer than the cap.
        stderr_truncated: True if stderr was longer than the cap.
        limits_applied: Names of the rlimits that were actually set, e.g.
            ``("RLIMIT_CPU", "RLIMIT_NPROC", ...)``. ``RLIMIT_AS`` is absent on macOS.
    """

    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool
    duration_seconds: float
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    limits_applied: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        """True iff the child ran to completion and exited zero."""
        return not self.timed_out and self.returncode == 0

    @property
    def killed_by_signal(self) -> int | None:
        """The signal that killed the child, if one did."""
        if self.returncode is not None and self.returncode < 0:
            return -self.returncode
        return None


def scrubbed_environ(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """Build the child's environment from scratch.

    Nothing is copied from :data:`os.environ`. That is the whole point: an allowlist that
    reads the parent's environment eventually grows an entry someone regrets, whereas a
    dict built from literals cannot leak a credential that this function has never heard
    of.

    ``PYTHONHASHSEED=0`` is set for determinism — without it, ``set`` and ``dict``
    iteration order varies per process, and a completion whose output depends on it would
    score differently on identical inputs.

    Args:
        extra: Additional variables to set. Keys are used as given; callers are trusted
            not to pass secrets.

    Returns:
        A fresh environment dict.
    """
    env = {
        "PATH": _SAFE_PATH,
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        # Silence the chattier ML libraries in case a completion imports one.
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    if extra:
        env.update(extra)
    return env


def _rlimit_plan(
    *, cpu_seconds: int, memory_limit_mb: int | None, max_processes: int, fsize_bytes: int
) -> tuple[list[tuple[int, tuple[int, int]]], tuple[str, ...]]:
    """Precompute the ``setrlimit`` calls so ``preexec_fn`` stays a dumb loop.

    Each entry is clamped to the inherited hard limit: an unprivileged process may lower
    a limit but never raise it, and attempting to raise one raises ``ValueError``, which
    inside ``preexec_fn`` would turn into a failed exec rather than a lenient sandbox.
    """
    plan: list[tuple[int, tuple[int, int]]] = []
    names: list[str] = []

    def add(res: int, name: str, value: int) -> None:
        try:
            _soft, hard = resource.getrlimit(res)
        except (ValueError, OSError):  # pragma: no cover - platform dependent
            return
        target = value if hard == resource.RLIM_INFINITY else min(value, hard)
        plan.append((res, (target, target if hard == resource.RLIM_INFINITY else hard)))
        names.append(name)

    add(resource.RLIMIT_CPU, "RLIMIT_CPU", cpu_seconds)
    if memory_limit_mb is not None and MEMORY_LIMIT_SUPPORTED:
        add(resource.RLIMIT_AS, "RLIMIT_AS", memory_limit_mb * 1024 * 1024)
    if hasattr(resource, "RLIMIT_NPROC"):
        add(resource.RLIMIT_NPROC, "RLIMIT_NPROC", max_processes)
    add(resource.RLIMIT_FSIZE, "RLIMIT_FSIZE", fsize_bytes)
    add(resource.RLIMIT_CORE, "RLIMIT_CORE", 0)
    return plan, tuple(names)


def _read_capped(path: Path, cap: int) -> tuple[str, bool]:
    """Read at most ``cap`` bytes from ``path``; report whether more was there."""
    try:
        with path.open("rb") as fh:
            blob = fh.read(cap + 1)
    except OSError:
        return "", False
    truncated = len(blob) > cap
    return blob[:cap].decode("utf-8", errors="replace"), truncated


def _kill_process_group(proc: subprocess.Popen[bytes]) -> None:
    """SIGKILL the child's whole session, then reap it.

    ``proc.kill()`` alone signals one pid. Anything the child spawned survives, keeps
    running, and — since it inherited the output file descriptors — keeps writing.
    """
    try:
        pgid = os.getpgid(proc.pid)
    except OSError:
        pgid = proc.pid
    try:
        os.killpg(pgid, signal.SIGKILL)
    except OSError:
        # No group to signal (already gone, or never created). Fall back to the pid.
        with contextlib.suppress(OSError):
            proc.kill()
    with contextlib.suppress(subprocess.TimeoutExpired):  # an unkillable child would hang here
        proc.wait(timeout=_REAP_TIMEOUT_SECONDS)


def run_python(
    code: str,
    *,
    timeout_seconds: float = 10.0,
    memory_limit_mb: int | None = 1024,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    max_processes: int = DEFAULT_MAX_PROCESSES,
    extra_env: Mapping[str, str] | None = None,
    files: Mapping[str, str] | None = None,
) -> SandboxResult:
    """Execute ``code`` as a standalone script and return what happened.

    Args:
        code: The program. Written to a file and run with the current interpreter; never
            ``exec``-ed in this process, because ``exec`` shares the trainer's heap,
            imports, signal handlers, file descriptors and exit status with the model.
        timeout_seconds: Wall-clock budget. On expiry the whole process group is killed.
        memory_limit_mb: ``RLIMIT_AS`` in MiB, or ``None`` to skip. Ignored on macOS.
        max_output_bytes: Cap on how much of each stream is read back.
        max_processes: ``RLIMIT_NPROC`` soft limit.
        extra_env: Extra environment variables, merged over the scrubbed base.
        files: Auxiliary files to drop into the working directory, ``{name: contents}``.
            Names must be plain filenames, not paths.

    Returns:
        A :class:`SandboxResult`.

    Raises:
        ValueError: If ``timeout_seconds`` is not positive or a filename is not plain.
    """
    if timeout_seconds <= 0:
        raise ValueError(f"timeout_seconds must be positive, got {timeout_seconds}")
    if max_output_bytes <= 0:
        raise ValueError(f"max_output_bytes must be positive, got {max_output_bytes}")

    workdir = Path(tempfile.mkdtemp(prefix="flowcode-sandbox-"))
    try:
        script = workdir / _SCRIPT_NAME
        script.write_text(code, encoding="utf-8")
        for name, contents in (files or {}).items():
            if os.path.sep in name or name in {"", ".", ".."} or os.path.isabs(name):
                raise ValueError(f"auxiliary file name must be a plain filename, got {name!r}")
            (workdir / name).write_text(contents, encoding="utf-8")

        out_path = workdir / "_flowcode_stdout"
        err_path = workdir / "_flowcode_stderr"

        # RLIMIT_CPU is a whole-seconds backstop: the wall clock should always fire first,
        # so give it a second of slack and let the timeout own the common case.
        cpu_seconds = max(1, int(timeout_seconds) + 1)
        fsize = max(_MIN_FSIZE_BYTES, max_output_bytes * 8)
        plan, limit_names = _rlimit_plan(
            cpu_seconds=cpu_seconds,
            memory_limit_mb=memory_limit_mb,
            max_processes=max_processes,
            fsize_bytes=fsize,
        )

        def _apply_limits() -> None:  # pragma: no cover - runs post-fork, pre-exec
            # Bare try/except rather than contextlib.suppress: this body runs between
            # fork and exec, where every extra import, attribute lookup and allocation is
            # a chance to touch a lock another thread of the parent still holds.
            for res, limits in plan:
                try:  # noqa: SIM105
                    resource.setrlimit(res, limits)
                except (ValueError, OSError):
                    pass

        env = scrubbed_environ({"HOME": str(workdir), "TMPDIR": str(workdir)})
        if extra_env:
            env.update(extra_env)

        started = time.monotonic()
        with out_path.open("wb") as out_fh, err_path.open("wb") as err_fh:
            proc = subprocess.Popen(
                # -B: no __pycache__ writes. -s: ignore the user site directory, so a
                # stray ~/.local/lib package cannot change what a completion sees.
                [sys.executable, "-B", "-s", _SCRIPT_NAME],
                cwd=str(workdir),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=out_fh,
                stderr=err_fh,
                preexec_fn=_apply_limits,
                start_new_session=True,
            )
            timed_out = False
            try:
                proc.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                _kill_process_group(proc)
        duration = time.monotonic() - started

        stdout, out_trunc = _read_capped(out_path, max_output_bytes)
        stderr, err_trunc = _read_capped(err_path, max_output_bytes)
        return SandboxResult(
            returncode=None if timed_out else proc.returncode,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            duration_seconds=duration,
            stdout_truncated=out_trunc,
            stderr_truncated=err_trunc,
            limits_applied=limit_names,
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
