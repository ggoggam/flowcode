"""Tests for the subprocess sandbox.

Every timeout here is sub-second on purpose: this file runs on every commit, and a
sandbox test suite that takes a minute stops being run.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from flowcode.envs.sandbox import (
    MEMORY_LIMIT_SUPPORTED,
    run_python,
    scrubbed_environ,
)


def test_captures_stdout_and_exit_code() -> None:
    result = run_python("print('hello from the sandbox')", timeout_seconds=5.0)
    assert result.ok
    assert result.returncode == 0
    assert "hello from the sandbox" in result.stdout
    assert not result.timed_out


def test_nonzero_exit_and_traceback_on_stderr() -> None:
    result = run_python("raise ValueError('boom')", timeout_seconds=5.0)
    assert not result.ok
    assert result.returncode == 1
    assert "ValueError: boom" in result.stderr


def test_timeout_kills_an_infinite_loop_promptly() -> None:
    started = time.monotonic()
    result = run_python("while True:\n    pass", timeout_seconds=0.4)
    elapsed = time.monotonic() - started

    assert result.timed_out
    assert result.returncode is None
    # The kill must be prompt, not merely eventual: a run that overshoots its budget by
    # seconds per sample destroys the throughput of the training loop.
    assert elapsed < 5.0


def test_timeout_reaps_grandchildren(tmp_path: Path) -> None:
    """A bare ``proc.kill()`` would leave the grandchild running and writing forever."""
    marker = tmp_path / "grandchild.log"
    grandchild = (
        "import time\n"
        "while True:\n"
        f"    with open({str(marker)!r}, 'a') as fh:\n"
        "        fh.write('x')\n"
        "    time.sleep(0.02)\n"
    )
    code = (
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, 'grandchild.py'])\n"
        "time.sleep(30)\n"
    )
    # Raise the process cap so the fork under test is actually allowed to happen; the
    # point of this test is the kill, not RLIMIT_NPROC.
    result = run_python(
        code, timeout_seconds=0.6, max_processes=4096, files={"grandchild.py": grandchild}
    )
    assert result.timed_out

    time.sleep(0.3)
    assert marker.exists() and marker.stat().st_size > 0, (
        "the grandchild never ran, so this test would pass vacuously"
    )
    size_after_kill = marker.stat().st_size
    time.sleep(0.4)
    size_later = marker.stat().st_size if marker.exists() else 0
    assert size_later == size_after_kill, (
        "the grandchild kept writing after the timeout: the process group was not killed"
    )


def test_environment_is_scrubbed_of_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    """A completion that prints ``os.environ`` must not be able to exfiltrate the key."""
    monkeypatch.setenv("TINKER_API_KEY", "sk-tinker-do-not-leak")
    monkeypatch.setenv("WANDB_API_KEY", "wandb-do-not-leak")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-do-not-leak")

    result = run_python(
        "import os\nprint(repr(dict(os.environ)))\nprint(os.environ.get('TINKER_API_KEY'))",
        timeout_seconds=5.0,
    )
    assert result.ok
    assert "do-not-leak" not in result.stdout
    assert "TINKER_API_KEY" not in result.stdout
    assert "None" in result.stdout


def test_scrubbed_environ_is_built_from_scratch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOME_PRIVATE_TOKEN", "value")
    env = scrubbed_environ()
    assert "SOME_PRIVATE_TOKEN" not in env
    assert env["PYTHONHASHSEED"] == "0"
    assert os.environ.get("PATH") != env["PATH"] or env["PATH"].startswith("/usr/local/bin")


def test_output_is_capped_so_a_print_loop_cannot_flood_the_parent() -> None:
    result = run_python(
        "for i in range(200000):\n    print('spam' * 20)",
        timeout_seconds=10.0,
        max_output_bytes=4096,
    )
    assert len(result.stdout) <= 4096
    assert result.stdout_truncated


def test_hash_seed_is_fixed_so_set_ordering_is_reproducible() -> None:
    code = "print(list({'alpha', 'beta', 'gamma', 'delta', 'epsilon'}))"
    first = run_python(code, timeout_seconds=5.0)
    second = run_python(code, timeout_seconds=5.0)
    assert first.ok and second.ok
    assert first.stdout == second.stdout


def test_working_directory_is_fresh_each_run() -> None:
    write = "open('scratch.txt', 'w').write('written')"
    read = "import os\nprint(sorted(os.listdir('.')))\nprint(os.path.exists('scratch.txt'))"
    assert run_python(write, timeout_seconds=5.0).ok
    result = run_python(read, timeout_seconds=5.0)
    assert result.ok
    assert "False" in result.stdout
    assert "scratch.txt" not in result.stdout


def test_auxiliary_files_land_in_the_working_directory() -> None:
    result = run_python(
        "print(open('data.txt').read())",
        timeout_seconds=5.0,
        files={"data.txt": "payload"},
    )
    assert result.ok
    assert "payload" in result.stdout


def test_auxiliary_file_names_must_be_plain() -> None:
    with pytest.raises(ValueError, match="plain filename"):
        run_python("pass", timeout_seconds=1.0, files={"../escape.txt": "nope"})


def test_stdin_is_closed_so_input_cannot_hang() -> None:
    result = run_python(
        "try:\n    input()\nexcept EOFError:\n    print('eof')",
        timeout_seconds=5.0,
    )
    assert result.ok
    assert "eof" in result.stdout


def test_invalid_timeout_rejected() -> None:
    with pytest.raises(ValueError, match="timeout_seconds"):
        run_python("pass", timeout_seconds=0.0)


def test_cpu_and_core_limits_are_applied() -> None:
    result = run_python("print('ok')", timeout_seconds=5.0)
    assert "RLIMIT_CPU" in result.limits_applied
    assert "RLIMIT_CORE" in result.limits_applied


def test_process_limit_blocks_a_fork_bomb() -> None:
    code = (
        "import subprocess, sys\n"
        "spawned = 0\n"
        "try:\n"
        "    for _ in range(200):\n"
        "        subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(5)'])\n"
        "        spawned += 1\n"
        "except BlockingIOError:\n"
        "    pass\n"
        "except OSError:\n"
        "    pass\n"
        "print('spawned', spawned)\n"
    )
    result = run_python(code, timeout_seconds=8.0, max_processes=2)
    # Either the fork was refused outright or the whole group died with the timeout; what
    # must not happen is 200 surviving children.
    assert result.timed_out or "spawned" in result.stdout
    if "spawned" in result.stdout:
        spawned = int(result.stdout.split("spawned")[-1].strip())
        assert spawned < 200


@pytest.mark.skipif(not MEMORY_LIMIT_SUPPORTED, reason="RLIMIT_AS is not enforceable on macOS")
def test_memory_limit_trips_on_a_memory_bomb() -> None:
    result = run_python(
        "chunks = []\nwhile True:\n    chunks.append(bytearray(20 * 1024 * 1024))",
        timeout_seconds=10.0,
        memory_limit_mb=256,
    )
    assert not result.timed_out, "the allocation should have failed long before the timeout"
    assert "MemoryError" in result.stderr
    assert "RLIMIT_AS" in result.limits_applied


@pytest.mark.skipif(MEMORY_LIMIT_SUPPORTED, reason="only meaningful where RLIMIT_AS is unavailable")
def test_memory_limit_degrades_gracefully_on_macos() -> None:
    """On Darwin the limit is skipped, and that must be visible rather than pretended."""
    result = run_python(
        "chunk = bytearray(8 * 1024 * 1024)\nprint(len(chunk))",
        timeout_seconds=5.0,
        memory_limit_mb=1,
    )
    assert result.ok, "skipping RLIMIT_AS must not break the run"
    assert "RLIMIT_AS" not in result.limits_applied
    assert "8388608" in result.stdout
