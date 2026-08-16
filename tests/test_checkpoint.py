"""Checkpoint round-trips, and the preemption drill.

TRC quota is preemptible, so "resume" is not a nice-to-have here — it is the normal way a
run continues. The failure this file exists to prevent is the *quiet* one: a resume that
restores the policy, restarts the LR schedule at step 0, reinitialises ``log Z`` and drops
the replay buffer will train, will log plausible metrics, and will not be the run that was
interrupted.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from flowcode.checkpoint import (
    RUN_STATE_FILENAME,
    RunState,
    load_run_state,
    restore_flow_parameters,
    save_run_state,
)
from flowcode.types import Segment, Trajectory

pytest.importorskip("peft", reason="LocalBackend needs the `local` extra")


def make_trajectory(task_id: str = "t0") -> Trajectory:
    return Trajectory(
        task_id=task_id,
        prompt_tokens=[1, 2, 3],
        completion_tokens=[4, 5],
        sampling_logprobs=[-0.1, -0.2],
        log_reward=-1.0,
        segments=[Segment(0, 2)],
        metadata={"passed": True},
    )


# ------------------------------------------------------------------------ run state


class TestRunStateRoundTrip:
    def test_carries_the_step_index(self, tmp_path: Path) -> None:
        # The LR schedule is a function of this; losing it replays warmup on a policy
        # that is a thousand steps in.
        save_run_state(tmp_path, RunState(step_index=417))
        assert load_run_state(tmp_path).step_index == 417

    def test_carries_the_replay_buffer(self, tmp_path: Path) -> None:
        save_run_state(tmp_path, RunState(replay=[make_trajectory("a"), make_trajectory("b")]))
        restored = load_run_state(tmp_path).replay
        assert [t.task_id for t in restored] == ["a", "b"]
        assert restored[0].completion_tokens == [4, 5]

    def test_carries_the_flow_parameters(self, tmp_path: Path) -> None:
        save_run_state(tmp_path, RunState(flow_parameters=[torch.tensor([1.5, -2.5])]))
        (restored,) = load_run_state(tmp_path).flow_parameters
        assert restored.tolist() == pytest.approx([1.5, -2.5])

    def test_carries_the_rng_states(self, tmp_path: Path) -> None:
        import random

        rng = random.Random(7)
        rng.random()
        save_run_state(
            tmp_path, RunState(rng_state=rng.getstate(), torch_rng_state=torch.get_rng_state())
        )
        state = load_run_state(tmp_path)

        resumed = random.Random()
        resumed.setstate(state.rng_state)
        assert resumed.random() == pytest.approx(rng.random())

    def test_carries_metadata(self, tmp_path: Path) -> None:
        save_run_state(tmp_path, RunState(metadata={"objective": "vargrad"}))
        assert load_run_state(tmp_path).metadata["objective"] == "vargrad"

    def test_writes_a_predictable_filename(self, tmp_path: Path) -> None:
        save_run_state(tmp_path, RunState())
        assert (tmp_path / RUN_STATE_FILENAME).is_file()

    def test_a_missing_run_state_is_loud(self, tmp_path: Path) -> None:
        # Silently returning an empty state would resume at step 0 with a warm policy —
        # a different run wearing the same weights.
        with pytest.raises(FileNotFoundError, match="reinitialised flow parameters"):
            load_run_state(tmp_path)


class TestRestoreFlowParameters:
    def test_copies_in_place(self) -> None:
        live = [torch.nn.Parameter(torch.zeros(3))]
        restore_flow_parameters(live, [torch.tensor([1.0, 2.0, 3.0])])
        assert live[0].detach().tolist() == pytest.approx([1.0, 2.0, 3.0])

    def test_count_mismatch_is_loud(self) -> None:
        with pytest.raises(ValueError, match="misalign per-task log Z"):
            restore_flow_parameters([torch.nn.Parameter(torch.zeros(3))], [])

    def test_shape_mismatch_is_loud(self) -> None:
        # A grown task set would otherwise transpose per-task log Z onto the wrong tasks,
        # which trains, and trains wrong.
        with pytest.raises(ValueError, match="task set changed size"):
            restore_flow_parameters([torch.nn.Parameter(torch.zeros(5))], [torch.zeros(3)])


# --------------------------------------------------------------------- backend state


class TestLocalBackendCheckpoint:
    def test_restores_the_adapter_weights(self, tmp_path: Path) -> None:
        from test_local_backend import tiny_backend

        backend = tiny_backend()
        before = [p.detach().clone() for p in backend.parameters()]
        backend.save_checkpoint(str(tmp_path / "ckpt"))

        # Move the weights, then put them back.
        with torch.no_grad():
            for p in backend.parameters():
                p.add_(1.0)
        assert not all(
            torch.allclose(a, b) for a, b in zip(backend.parameters(), before, strict=True)
        )

        backend.load_checkpoint(str(tmp_path / "ckpt"))
        for restored, original in zip(backend.parameters(), before, strict=True):
            assert torch.allclose(restored, original)

    async def test_restores_the_optimiser_moments(self, tmp_path: Path) -> None:
        # Adam's second moment takes hundreds of steps to settle; a resume without it
        # takes badly-scaled steps on an otherwise converged policy.
        from test_local_backend import make_trajectory as make_local_trajectory
        from test_local_backend import tiny_backend

        backend = tiny_backend()
        await backend.apply_gradient([make_local_trajectory()], [torch.ones(3)])
        await backend.optim_step(lr=1e-2)
        backend.save_checkpoint(str(tmp_path / "ckpt"))

        fresh = tiny_backend()
        assert fresh._optimizer.state_dict()["state"] == {}
        fresh.load_checkpoint(str(tmp_path / "ckpt"))
        assert fresh._optimizer.state_dict()["state"] != {}

    def test_restores_the_policy_version(self, tmp_path: Path) -> None:
        from test_local_backend import tiny_backend

        backend = tiny_backend()
        backend._policy_version = 12
        backend.save_checkpoint(str(tmp_path / "ckpt"))

        fresh = tiny_backend()
        fresh.load_checkpoint(str(tmp_path / "ckpt"))
        assert fresh.policy_version == 12

    def test_a_bare_adapter_is_refused(self, tmp_path: Path) -> None:
        from test_local_backend import tiny_backend

        backend = tiny_backend()
        backend.save_checkpoint(str(tmp_path / "ckpt"))
        (tmp_path / "ckpt" / "backend.pt").unlink()

        fresh = tiny_backend()
        with pytest.raises(FileNotFoundError, match="cold"):
            fresh.load_checkpoint(str(tmp_path / "ckpt"))

    def test_a_missing_checkpoint_is_loud(self, tmp_path: Path) -> None:
        from test_local_backend import tiny_backend

        with pytest.raises(FileNotFoundError, match="no adapter directory"):
            tiny_backend().load_checkpoint(str(tmp_path / "nope"))
