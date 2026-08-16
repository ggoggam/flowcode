"""A bounded off-policy replay buffer over scored trajectories.

Replay is not an optimisation here, it is one of the reasons to use a GFlowNet at all. The
balance conditions the objectives regress on are properties of the policy and flow
functions, not expectations under the sampling distribution, so a trajectory sampled ten
thousand steps ago is still a *valid constraint* — see the "Off-policy correction" section
of :mod:`flowcode.objectives.base`. PPO cannot reuse that data without importance ratios
and clipping; we can reuse it as-is.

What a stored trajectory carries, and why it is enough
------------------------------------------------------
:class:`~flowcode.types.Trajectory` keeps ``prompt_tokens``, ``completion_tokens``,
``sampling_logprobs``, ``log_reward`` and ``segments``. Only the first two matter for
replay: they are exactly what
:meth:`flowcode.tinker_backend.TinkerBackend.compute_logprobs` needs to re-score the
completion **under the current policy**. The stored ``sampling_logprobs`` belong to
whatever policy generated the trajectory and are stale by construction; the training loop
must not feed them to the objective as current-policy logprobs. That is precisely why
``train.on_policy_only`` (which does exactly that) is rejected when replay is enabled.

The prioritization schemes
--------------------------
``prioritize`` picks how :meth:`ReplayBuffer.sample` weights the buffer:

``uniform``
    Every stored trajectory is equally likely. The baseline, and the safe choice.

``recency``
    Weight ``exp(-RECENCY_DECAY * age_rank / N)``, where ``age_rank`` is 0 for the newest
    entry and ``N-1`` for the oldest. The oldest entry stays about ``e^-5 ~ 0.7%`` as
    likely as the newest, so old data thins out but never becomes unreachable.

``reward``
    Weight ``exp(-REWARD_SHARPNESS * reward_rank / (N-1))``, where ``reward_rank`` is 0 for
    the highest ``log_reward`` in the buffer and ``N-1`` for the lowest. Two deliberate
    choices:

    *Rank-based, not a softmax over ``log R`` itself.* ``log R`` here is
    ``beta * log(pass_fraction)`` floored at ``log(reward_floor)``, so it lives on a
    compressed, bounded scale that shifts as the policy improves: early on almost
    everything sits on the floor and a value-softmax is nearly uniform; later the spread
    collapses again as most samples pass. Ranks are invariant to both, so the effective
    selection pressure stays constant over a run instead of drifting with the reward
    distribution.

    *Soft, not top-k.* Keeping only the best trajectories is the obvious thing and the
    wrong one. A GFlowNet is trained to sample *proportionally* to the reward — its selling
    point over PPO is mode coverage — and a buffer that retains only the current best
    solutions feeds back exactly the distribution collapse the objective exists to avoid.
    At the default sharpness the worst entry in the buffer still has ``e^-3 ~ 5%`` of the
    best entry's weight, so every mode keeps mass.

Eviction is FIFO regardless of the scheme. Evicting the lowest-reward entry instead would
make the buffer a hall of fame, with the same collapse problem one level down.

Determinism
-----------
:meth:`ReplayBuffer.sample` takes an explicit :class:`random.Random`, never the global RNG,
and draws without replacement using the exponential-race (Efraimidis-Spirakis) trick: one
uniform per candidate, then take the ``n`` largest ``u ** (1 / w)``. Same buffer, same
seeded rng, same sample — which is what makes a replay-enabled run reproducible.
"""

from __future__ import annotations

import math
import random
from collections import deque
from collections.abc import Iterable, Iterator
from typing import Final

from flowcode.types import Trajectory

__all__ = ["PRIORITIZE_KINDS", "RECENCY_DECAY", "REWARD_SHARPNESS", "ReplayBuffer"]

PRIORITIZE_KINDS: Final[tuple[str, ...]] = ("reward", "uniform", "recency")
"""Legal values of ``train.replay.prioritize``."""

REWARD_SHARPNESS: Final[float] = 3.0
"""Log-odds spread between the best- and worst-ranked trajectory. ``e^-3 ~ 5%``."""

RECENCY_DECAY: Final[float] = 5.0
"""Log-odds spread between the newest and oldest trajectory. ``e^-5 ~ 0.7%``."""


class ReplayBuffer:
    """A fixed-capacity FIFO buffer with weighted, deterministic sampling.

    Args:
        capacity: Maximum trajectories retained. Adding past this evicts the oldest, so
            memory is bounded by ``capacity`` trajectories no matter how long the run is.
        prioritize: One of :data:`PRIORITIZE_KINDS`.

    Raises:
        ValueError: If ``capacity`` is not positive or ``prioritize`` is unknown.
    """

    def __init__(self, capacity: int, prioritize: str = "uniform") -> None:
        if capacity <= 0:
            raise ValueError(f"replay.capacity must be positive, got {capacity}")
        if prioritize not in PRIORITIZE_KINDS:
            raise ValueError(
                f"replay.prioritize must be one of {list(PRIORITIZE_KINDS)}, got {prioritize!r}"
            )
        self.capacity = int(capacity)
        self.prioritize = prioritize
        self._items: deque[Trajectory] = deque(maxlen=self.capacity)
        self._added = 0
        self._evicted = 0

    # ------------------------------------------------------------------ container protocol

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Trajectory]:
        """Iterate oldest to newest."""
        return iter(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)

    @property
    def num_added(self) -> int:
        """Total trajectories ever added, including those since evicted."""
        return self._added

    @property
    def num_evicted(self) -> int:
        """Total trajectories dropped to stay inside ``capacity``."""
        return self._evicted

    # ---------------------------------------------------------------------------- mutation

    def add(self, trajectories: Iterable[Trajectory]) -> None:
        """Append trajectories, evicting the oldest once ``capacity`` is reached.

        Args:
            trajectories: Newly scored trajectories, in generation order.
        """
        for trajectory in trajectories:
            if len(self._items) == self.capacity:
                self._evicted += 1
            self._items.append(trajectory)
            self._added += 1

    def clear(self) -> None:
        """Drop everything. Counters are kept — they describe the run, not the contents."""
        self._items.clear()

    # ---------------------------------------------------------------------------- sampling

    def weights(self) -> list[float]:
        """Current sampling weight of every stored trajectory, oldest first.

        Exposed so the prioritization scheme can be asserted directly instead of inferred
        from a histogram of draws.

        Returns:
            One positive weight per stored trajectory. All ones under ``uniform``.
        """
        n = len(self._items)
        if n == 0:
            return []
        if self.prioritize == "uniform":
            return [1.0] * n
        if self.prioritize == "recency":
            # index 0 is the oldest, so its age rank is n-1.
            return [math.exp(-RECENCY_DECAY * (n - 1 - i) / n) for i in range(n)]

        # reward: rank by log_reward, best first, ties broken by insertion order so the
        # weights are a deterministic function of the buffer contents.
        order = sorted(range(n), key=lambda i: (-self._items[i].log_reward, i))
        denominator = float(n - 1) if n > 1 else 1.0
        weights = [0.0] * n
        for rank, index in enumerate(order):
            weights[index] = math.exp(-REWARD_SHARPNESS * rank / denominator)
        return weights

    def sample(self, n: int, rng: random.Random) -> list[Trajectory]:
        """Draw up to ``n`` trajectories without replacement.

        Args:
            n: How many to draw. Values above ``len(self)`` return the whole buffer (in
                weighted order); zero or less returns an empty list. Asking for more than
                exists is normal early in a run and is not an error.
            rng: A seeded :class:`random.Random`. Required, not defaulted: replay is one of
                the two places a run could become irreproducible, and the other is the
                sampler's own temperature.

        Returns:
            The drawn trajectories, highest sampling key first.
        """
        if n <= 0 or not self._items:
            return []
        items = list(self._items)
        weights = self.weights()
        # Efraimidis-Spirakis: key_i = u_i ** (1 / w_i), take the top n. Equivalent to
        # weighted sampling without replacement, in one pass, with one uniform per item.
        keyed: list[tuple[float, int]] = []
        for index, weight in enumerate(weights):
            u = rng.random()
            # u == 0.0 is possible; log(0) is not, and its key should be the smallest.
            key = -math.inf if u <= 0.0 else math.log(u) / max(weight, 1e-12)
            keyed.append((key, index))
        keyed.sort(key=lambda pair: (-pair[0], pair[1]))
        return [items[index] for _key, index in keyed[: min(n, len(items))]]

    def task_ids(self) -> list[str]:
        """Task ids of everything stored, oldest first. Handy for metrics and tests."""
        return [t.task_id for t in self._items]

    def stats(self) -> dict[str, float]:
        """Buffer metrics for the logger.

        Returns:
            ``size``, ``capacity``, ``num_added``, ``num_evicted``, ``distinct_tasks`` and
            ``log_reward_mean`` (0.0 when empty).
        """
        size = len(self._items)
        mean = sum(t.log_reward for t in self._items) / size if size else 0.0
        return {
            "size": float(size),
            "capacity": float(self.capacity),
            "num_added": float(self._added),
            "num_evicted": float(self._evicted),
            "distinct_tasks": float(len({t.task_id for t in self._items})),
            "log_reward_mean": mean,
        }

    def __repr__(self) -> str:
        return (
            f"ReplayBuffer(size={len(self._items)}, capacity={self.capacity}, "
            f"prioritize={self.prioritize!r})"
        )
