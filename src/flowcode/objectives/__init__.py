"""GFlowNet training objectives, in pure torch.

Four losses over the same interface (:class:`~flowcode.objectives.base.Objective`):

============  ==========================================================================
``vargrad``   Trajectory balance with the partition function eliminated in-batch. **The
              recommended default** — no learned ``log Z``, nothing to tune, unbiased.
              Needs ``group_size > 1``.
``tb``        Trajectory balance with a learned per-prompt ``log Z``. Unbiased, sparse
              credit assignment, one fiddly estimator.
``subtb``     SubTB(λ) over sub-trajectory pairs. Much denser credit assignment, at the
              cost of depending on a client-side flow estimator that cannot see the
              prefix (see :mod:`flowcode.objectives.flows`).
``db``        Detailed balance over adjacent states. Densest, and leans hardest on that
              same approximation.
============  ==========================================================================

Nothing here imports ``tinker``. The objectives consume per-token logprobs and
:class:`~flowcode.types.Trajectory` objects and produce a scalar loss, which is what lets the
whole set be unit- and convergence-tested offline at zero API cost — see
``tests/test_convergence.py``, which trains each objective on a toy model whose exact target
distribution is computable by enumeration.
"""

from __future__ import annotations

from flowcode.objectives.base import (
    BaseObjective,
    Granularity,
    Objective,
    boundary_cumulative_logprobs,
    importance_weights,
    reduce_per_trajectory,
    resolve_segments,
    segment_logprob_sums,
    temper_log_reward,
    tempered_log_rewards,
)
from flowcode.objectives.db import DetailedBalance
from flowcode.objectives.flows import (
    ConditionalLogZ,
    FlowStates,
    LogFlow,
    LogFlowEstimator,
    LogZ,
    ScalarLogZ,
)
from flowcode.objectives.subtb import SubTB
from flowcode.objectives.tb import TrajectoryBalance
from flowcode.objectives.vargrad import VarGrad

__all__ = [
    "BaseObjective",
    "ConditionalLogZ",
    "DetailedBalance",
    "FlowStates",
    "Granularity",
    "LogFlow",
    "LogFlowEstimator",
    "LogZ",
    "Objective",
    "ScalarLogZ",
    "SubTB",
    "TrajectoryBalance",
    "VarGrad",
    "boundary_cumulative_logprobs",
    "importance_weights",
    "reduce_per_trajectory",
    "resolve_segments",
    "segment_logprob_sums",
    "temper_log_reward",
    "tempered_log_rewards",
]
