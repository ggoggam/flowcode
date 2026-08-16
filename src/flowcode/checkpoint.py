"""Saving and restoring a run, because preemptible quota takes runs away mid-flight.

Tinker checkpointed the policy for us. Training locally on TRC — where quota is
preemptible and time-boxed — means owning the whole problem, and "the whole problem" is
more than the weights. A run resumed with only its LoRA restored silently loses:

* the **flow parameters**. ``log Z`` / ``log F`` are fitted quantities spanning dozens of
  nats (:mod:`flowcode.objectives.flows`), trained at 100-1000x the policy learning rate
  precisely because they take a while to find their scale. Reinitialising them while the
  policy carries on is worse than restarting — every balance residual for the next few
  hundred steps measures flow error rather than policy error.
* the **replay buffer**. Discarding it makes the batch abruptly on-policy at the moment of
  resume, which is a silent change of algorithm, not a hiccup.
* the **step index**, which the LR schedule is a function of. Resuming at step 0 replays
  the warmup on a policy that is a thousand steps in.

So all of it is written together and restored together, and :func:`load_run_state` is
strict about what it finds rather than filling in blanks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from flowcode.types import Trajectory

__all__ = [
    "RUN_STATE_FILENAME",
    "RunState",
    "load_run_state",
    "restore_flow_parameters",
    "save_run_state",
]

logger = logging.getLogger(__name__)

RUN_STATE_FILENAME = "run_state.pt"
"""Everything that is not the policy, in one file next to the adapter."""


@dataclass
class RunState:
    """The trainer's own state, separate from the backend's.

    Args:
        step_index: Steps completed. The LR schedule is a function of this.
        policy_version: Sampler syncs completed, for staleness accounting on resume.
        flow_parameters: The objective's ``log Z`` / ``log F`` tensors, in
            ``objective.parameters()`` order. That order is stable because
            ``register_tasks`` is called once with the full, deterministically-sorted task
            set — restoring into a differently-registered objective would silently
            transpose per-task entries, so :func:`load_run_state` checks the shapes.
        flow_optimizer: Adam moments for those parameters. Dropping them costs a few
            hundred steps of re-warming a second-moment estimate.
        replay: The buffer's contents.
        rng_state: The trainer's ``random.Random`` state, so task selection resumes where
            it left off rather than replaying the same draws.
        torch_rng_state: Global torch RNG state.
        metadata: Free-form extras (objective name, model name) for sanity-checking a
            checkpoint before trusting it.
    """

    step_index: int = 0
    policy_version: int = 0
    flow_parameters: list[torch.Tensor] = field(default_factory=list)
    flow_optimizer: dict[str, Any] | None = None
    replay: list[Trajectory] = field(default_factory=list)
    rng_state: Any = None
    torch_rng_state: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def save_run_state(directory: str | Path, state: RunState) -> str:
    """Write ``state`` into ``directory``.

    Args:
        directory: Where to write. Created if absent.
        state: What to write.

    Returns:
        The path written.
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    target = path / RUN_STATE_FILENAME
    payload = {
        "step_index": state.step_index,
        "policy_version": state.policy_version,
        "flow_parameters": [t.detach().cpu() for t in state.flow_parameters],
        "flow_optimizer": state.flow_optimizer,
        "replay": state.replay,
        "rng_state": state.rng_state,
        "torch_rng_state": state.torch_rng_state,
        "metadata": state.metadata,
    }
    torch.save(payload, target)
    logger.info(
        "wrote run state to %s (step %d, %d replayed, %d flow tensors)",
        target,
        state.step_index,
        len(state.replay),
        len(state.flow_parameters),
    )
    return str(target)


def load_run_state(directory: str | Path) -> RunState:
    """Read a run state back.

    Args:
        directory: A directory :func:`save_run_state` wrote to.

    Returns:
        The restored state.

    Raises:
        FileNotFoundError: If no run state is there. Deliberately not a warning-and-empty
            default: resuming from a checkpoint that turns out to hold nothing would
            restart training from step 0 with a warm policy, which looks like a working
            run and is not.
    """
    target = Path(directory) / RUN_STATE_FILENAME
    if not target.is_file():
        raise FileNotFoundError(
            f"no {RUN_STATE_FILENAME} in {directory}. A checkpoint without it can only "
            "restore the policy, which would resume at step 0 with reinitialised flow "
            "parameters and an empty replay buffer — a different run wearing the same "
            "weights."
        )
    # weights_only=False: the payload holds Trajectory dataclasses, not just tensors.
    payload = torch.load(target, map_location="cpu", weights_only=False)
    return RunState(
        step_index=int(payload.get("step_index", 0)),
        policy_version=int(payload.get("policy_version", 0)),
        flow_parameters=list(payload.get("flow_parameters", [])),
        flow_optimizer=payload.get("flow_optimizer"),
        replay=list(payload.get("replay", [])),
        rng_state=payload.get("rng_state"),
        torch_rng_state=payload.get("torch_rng_state"),
        metadata=dict(payload.get("metadata", {})),
    )


def restore_flow_parameters(
    parameters: list[torch.nn.Parameter], saved: list[torch.Tensor]
) -> None:
    """Copy saved flow tensors back into live parameters, in order.

    Args:
        parameters: ``objective.parameters()`` as a list.
        saved: What :class:`RunState` carried.

    Raises:
        ValueError: If the counts or any shape disagree. A mismatch means the objective or
            the task set changed between saving and resuming, and copying anyway would
            transpose per-task ``log Z`` entries onto the wrong tasks — which trains, and
            trains wrong.
    """
    if len(parameters) != len(saved):
        raise ValueError(
            f"checkpoint holds {len(saved)} flow tensors but the objective has "
            f"{len(parameters)}. The objective or the registered task set changed since "
            "this checkpoint was written; resuming into it would misalign per-task log Z."
        )
    for i, (live, stored) in enumerate(zip(parameters, saved, strict=True)):
        if live.shape != stored.shape:
            raise ValueError(
                f"flow parameter {i} has shape {tuple(live.shape)} but the checkpoint holds "
                f"{tuple(stored.shape)}; the task set changed size since it was written"
            )
        with torch.no_grad():
            live.copy_(stored.to(live.device, live.dtype))
