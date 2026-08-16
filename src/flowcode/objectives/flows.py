"""Client-side state-flow and partition-function estimators.

Honest statement of the deviation from the papers
-------------------------------------------------
Trajectory Balance, SubTB and Detailed Balance are all written for a setting where the flow
functions share a trunk with the policy: in the LLM formulations (Malkin et al. 2022; Madan
et al. 2023; Hu et al. 2024) ``log Z(x)`` and ``log F(s)`` are *extra heads on the language
model itself*, so they read the model's hidden state and can therefore be arbitrary functions
of the prefix.

We cannot do that on Tinker. The service exposes sampling, a forward pass returning per-token
logprobs, and a backward pass over a fixed set of loss functions; there is no way to attach an
extra head to the hosted model, read its hidden states, or train a head jointly with the LoRA.
So the estimators here are **our own small torch modules, living in the training process, with
their own optimiser** (``train.flow_lr``, conventionally 100-1000x the policy LR because they
are fitting a handful of scalars that can span dozens of nats, not a language model).

The consequence is real and worth stating plainly:

* :class:`ConditionalLogZ` is *exact enough*. ``log Z(x)`` depends only on the prompt, and a
  free parameter per task can represent any per-prompt partition function. Nothing is lost.
* :class:`LogFlowEstimator` is *an approximation*. ``log F(s)`` should depend on the whole
  prefix ``s``, and the only prefix-derived features we have without model internals are the
  ones the trajectory already carries: how far along we are, how much probability mass the
  policy spent getting here, and any partial reward the environment could score. Two different
  prefixes with the same position and cumulative logprob get the same predicted flow. That
  makes DB and SubTB(λ) *biased* in a way TB and VarGrad are not, which is the main reason
  ``objective=vargrad`` is the recommended default and DB is the least recommended.

If you have a flow estimator that can see more of the state, implement :class:`LogFlow` and
pass it as ``flow:`` in the config — :class:`~flowcode.objectives.subtb.SubTB` and
:class:`~flowcode.objectives.db.DetailedBalance` only depend on the protocol. :class:`FlowStates`
deliberately carries the raw completion tokens so a richer estimator can reconstruct prefixes.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from torch import nn

__all__ = [
    "ConditionalLogZ",
    "FlowStates",
    "LogFlow",
    "LogFlowEstimator",
    "LogZ",
    "ScalarLogZ",
]


@dataclass(frozen=True)
class FlowStates:
    """A batch of GFlowNet states whose ``log F`` is wanted, flattened across trajectories.

    Every tensor field has the same leading dimension ``N`` (the total number of intermediate
    states in the minibatch) so an estimator is a single forward pass, not one per trajectory.

    Args:
        task_ids: Task id per state, length ``N``. Used to look up a task embedding.
        traj_index: ``(N,)`` long — index into ``completions`` (and into the original batch).
        end_index: ``(N,)`` long — the state is the completion prefix of this length.
        position: ``(N,)`` float — ``end_index / num_completion_tokens``, in ``(0, 1)``.
        cum_logprob: ``(N,)`` float — ``Σ log P_F`` spent reaching this state. **Detached**:
            it is an input feature, not a path for policy gradient. Letting gradient flow
            through it would have the policy optimise the *flow network's inputs*, which is
            not a term in any of the balance conditions.
        partial_log_reward: ``(N,)`` float — tempered ``β log R(s)`` where the environment
            could score the prefix, 0 elsewhere (read together with ``has_partial``).
        has_partial: ``(N,)`` float — 1 where ``partial_log_reward`` is meaningful, else 0.
        completions: Per-trajectory completion token lists, so a richer estimator can
            reconstruct the prefix as ``completions[traj_index[k]][: end_index[k]]``.
        traj_slices: ``(start, stop)`` row range per trajectory, aligned with the original
            batch order. Empty ranges (``start == stop``) are normal: a trajectory with a
            single segment has no intermediate states at all.
    """

    task_ids: list[str]
    traj_index: torch.Tensor
    end_index: torch.Tensor
    position: torch.Tensor
    cum_logprob: torch.Tensor
    partial_log_reward: torch.Tensor
    has_partial: torch.Tensor
    completions: list[list[int]]
    traj_slices: list[tuple[int, int]]

    def __len__(self) -> int:
        return len(self.task_ids)

    def prefix(self, row: int) -> list[int]:
        """The completion prefix defining state ``row``.

        Args:
            row: Index into the flattened state batch.

        Returns:
            The tokens generated so far, i.e. the state itself.
        """
        return self.completions[int(self.traj_index[row].item())][: int(self.end_index[row].item())]


@runtime_checkable
class LogZ(Protocol):
    """A per-prompt log partition function, ``log Z(x)``."""

    def log_z(self, task_ids: Sequence[str]) -> torch.Tensor:
        """Return ``(len(task_ids),)`` estimates of ``log Z``."""
        ...

    def parameters(self) -> Iterator[nn.Parameter]:
        """The learnable parameters, for the flow optimiser."""
        ...

    def register_tasks(self, task_ids: Sequence[str]) -> None:
        """Size any per-task table before the optimiser is built."""
        ...


@runtime_checkable
class LogFlow(Protocol):
    """A state flow function, ``log F(s)``, for intermediate (non-terminal) states."""

    def log_flow(self, states: FlowStates) -> torch.Tensor:
        """Return ``(len(states),)`` estimates of ``log F``."""
        ...

    def parameters(self) -> Iterator[nn.Parameter]:
        """The learnable parameters, for the flow optimiser."""
        ...

    def register_tasks(self, task_ids: Sequence[str]) -> None:
        """Size any per-task table before the optimiser is built."""
        ...


class _GrowableEmbedding(nn.Module):
    """An embedding table keyed by task id that can grow after construction.

    Hydra instantiates the flow modules from ``conf/objective/*.yaml`` before the environment
    has loaded its task set, so the row count cannot be a constructor argument.
    :meth:`register_tasks` is the intended path and should be called once, before the flow
    optimiser is created. Lookups of unseen ids still work — the table grows and a warning is
    emitted — but growing replaces the ``nn.Parameter`` object, which silently orphans any
    optimiser state already keyed on the old tensor.
    """

    def __init__(self, dim: int, init_std: float = 0.0) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"embedding dim must be positive, got {dim}")
        self.dim = dim
        self.init_std = init_std
        self._index: dict[str, int] = {}
        self.table = nn.Parameter(torch.zeros(0, dim))

    @property
    def num_tasks(self) -> int:
        """How many distinct task ids are currently in the table."""
        return len(self._index)

    def register_tasks(self, task_ids: Sequence[str]) -> None:
        """Add any unseen ids, growing the table in one allocation."""
        unseen = [t for t in dict.fromkeys(task_ids) if t not in self._index]
        if not unseen:
            return
        for task_id in unseen:
            self._index[task_id] = len(self._index)
        rows = torch.zeros(len(unseen), self.dim, dtype=self.table.dtype, device=self.table.device)
        if self.init_std > 0.0:
            rows.normal_(0.0, self.init_std)
        self.table = nn.Parameter(torch.cat([self.table.detach(), rows], dim=0))

    def indices(self, task_ids: Sequence[str]) -> torch.Tensor:
        """Map ids to row indices, lazily registering (with a warning) anything unseen."""
        unseen = [t for t in dict.fromkeys(task_ids) if t not in self._index]
        if unseen:
            warnings.warn(
                f"{len(unseen)} task id(s) were not registered before use (e.g. "
                f"{unseen[0]!r}); growing the table now. Any optimiser already built over "
                "these parameters has been invalidated — call register_tasks(...) with the "
                "full task set before constructing the flow optimiser.",
                RuntimeWarning,
                stacklevel=3,
            )
            self.register_tasks(unseen)
        return torch.tensor(
            [self._index[t] for t in task_ids], dtype=torch.long, device=self.table.device
        )

    def forward(self, task_ids: Sequence[str]) -> torch.Tensor:
        """Look up ``(len(task_ids), dim)`` embeddings."""
        # Resolve indices first: the lookup may grow (and therefore *replace*) ``self.table``,
        # and Python would otherwise have already bound the pre-growth tensor.
        rows = self.indices(task_ids)
        return self.table[rows]


class ScalarLogZ(nn.Module):
    """One learned scalar ``log Z``, shared by every prompt.

    Correct only when the batch is genuinely unconditional (a single task, or a reward whose
    normaliser happens not to vary across prompts). With a mixed task batch a single scalar
    forces one partition function onto prompts with wildly different reward scales, and TB
    degenerates into fitting the average — use :class:`ConditionalLogZ` instead. Kept because
    it is the right thing for the toy convergence tests and for single-task fine-tuning.
    """

    def __init__(self, init_value: float = 0.0) -> None:
        """Initialise the scalar.

        Args:
            init_value: Starting value of ``log Z``. 0 is a reasonable default when rewards
                are already in a sane log range.
        """
        super().__init__()
        self.value = nn.Parameter(torch.tensor(float(init_value)))

    def log_z(self, task_ids: Sequence[str]) -> torch.Tensor:
        """Broadcast the scalar to ``(len(task_ids),)``."""
        return self.value.expand(len(task_ids))

    def register_tasks(self, task_ids: Sequence[str]) -> None:
        """No-op: there is nothing per-task to size."""
        return None


class ConditionalLogZ(nn.Module):
    """``log Z(x)``: a free per-task parameter read out of an embedding table.

    This is the *exact* parameterisation — ``log Z`` is a function of the prompt alone, and
    one unconstrained scalar per prompt can represent any such function. With
    ``embedding_dim > 1`` the extra dimensions plus a shared linear readout let related tasks
    share statistical strength, which helps when the task set is large and each task is only
    visited a few times; ``embedding_dim=1`` (the shipped default) is the plain free-scalar
    case.

    The readout's bias acts as a global ``log Z`` offset and is shared by every task, so the
    overall scale is learned fast and only the per-task residual has to be fitted.
    """

    def __init__(self, embedding_dim: int = 1, init_value: float = 0.0) -> None:
        """Initialise the table and readout.

        Args:
            embedding_dim: Width of the per-task embedding. 1 gives a free scalar per task.
            init_value: Value of ``log Z`` for every task at initialisation, carried by the
                readout bias.

        Raises:
            ValueError: If ``embedding_dim`` is not positive.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.init_value = init_value
        self.tasks = _GrowableEmbedding(embedding_dim)
        self.readout = nn.Linear(embedding_dim, 1)
        with torch.no_grad():
            # Uniform non-zero weights: a zero readout would make the (zero-initialised)
            # embedding rows receive no gradient at all on the first step.
            self.readout.weight.fill_(1.0 / embedding_dim)
            self.readout.bias.fill_(float(init_value))

    def log_z(self, task_ids: Sequence[str]) -> torch.Tensor:
        """Estimate ``log Z`` per task.

        Args:
            task_ids: One id per element of the batch; repeats are fine and share a row.

        Returns:
            Shape ``(len(task_ids),)``.
        """
        if not task_ids:
            return self.readout.bias.new_zeros(0)
        embeddings = self.tasks(task_ids)
        scores: torch.Tensor = self.readout(embeddings)
        return scores.squeeze(-1)

    def register_tasks(self, task_ids: Sequence[str]) -> None:
        """Size the table for the full task set, before the flow optimiser is built."""
        self.tasks.register_tasks(task_ids)

    def forward(self, task_ids: Sequence[str]) -> torch.Tensor:
        """Alias for :meth:`log_z`."""
        return self.log_z(task_ids)


class LogFlowEstimator(nn.Module):
    """``log F(s)`` for intermediate states, from an MLP over the features we actually have.

    Features, per state (see the module docstring for why the list is this short):

    * a learned task embedding — ``log F`` inherits the per-prompt scale of ``log Z(x)``;
    * normalised position ``end_index / T`` in ``(0, 1)``;
    * cumulative ``Σ log P_F`` so far, at two scales (a fixed ``/10`` rescale to keep the MLP
      input O(1) for long completions, and a per-token mean which is length-invariant);
    * the environment's partial ``β log R(s)`` when this boundary carries one, plus a 0/1 mask
      so the network can tell "reward 0" from "no reward available".

    The output layer is zero-initialised, so every state starts at ``log F = 0`` and the first
    gradient step is driven entirely by the balance residuals rather than by MLP noise.
    """

    def __init__(self, hidden_dim: int = 64, task_embedding_dim: int = 32) -> None:
        """Build the estimator.

        Args:
            hidden_dim: Width of the two hidden layers.
            task_embedding_dim: Width of the per-task embedding.

        Raises:
            ValueError: If either dimension is not positive.
        """
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        self.hidden_dim = hidden_dim
        self.task_embedding_dim = task_embedding_dim
        # Small random init: unlike log Z the task embedding here feeds a nonlinearity, and
        # identical rows would make every task's flow identical until the first gradient.
        self.tasks = _GrowableEmbedding(task_embedding_dim, init_std=0.02)
        self.num_extra_features = 5
        self.mlp = nn.Sequential(
            nn.Linear(task_embedding_dim + self.num_extra_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        head = self.mlp[-1]
        assert isinstance(head, nn.Linear)
        with torch.no_grad():
            head.weight.zero_()
            head.bias.zero_()

    def features(self, states: FlowStates) -> torch.Tensor:
        """Assemble the ``(N, task_embedding_dim + 5)`` feature matrix.

        Args:
            states: The states to featurise.

        Returns:
            A detached-input feature matrix (gradient reaches the task embedding only).
        """
        embeddings = self.tasks(states.task_ids)
        tokens_so_far = states.end_index.to(states.cum_logprob.dtype).clamp_min(1.0)
        extra = torch.stack(
            [
                states.position,
                states.cum_logprob / 10.0,
                states.cum_logprob / tokens_so_far,
                states.partial_log_reward,
                states.has_partial,
            ],
            dim=-1,
        )
        return torch.cat([embeddings, extra.detach()], dim=-1)

    def log_flow(self, states: FlowStates) -> torch.Tensor:
        """Predict ``log F(s)`` for every state in the batch.

        Args:
            states: Flattened intermediate states.

        Returns:
            Shape ``(len(states),)``. Returns an empty tensor when there are no intermediate
            states (a single-segment trajectory), which callers must handle.
        """
        if len(states) == 0:
            return self.tasks.table.new_zeros(0)
        predictions: torch.Tensor = self.mlp(self.features(states))
        return predictions.squeeze(-1)

    def register_tasks(self, task_ids: Sequence[str]) -> None:
        """Size the task embedding for the full task set."""
        self.tasks.register_tasks(task_ids)

    def forward(self, states: FlowStates) -> torch.Tensor:
        """Alias for :meth:`log_flow`."""
        return self.log_flow(states)
