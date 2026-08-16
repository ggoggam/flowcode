"""Samplers: where completions come from.

One protocol (:class:`~flowcode.local_backend.WeightSyncSampler`), several
implementations, selected by ``sampler.mode``. Keeping the choice here rather than in the
trainer is what lets the hardware question stay open — a run moves between a single shared
GPU, a split GPU node and a TPU host by changing a config value, not a code path.

Nothing in this package imports ``vllm`` at module scope. The engine is optional, has no
macOS wheel, and in ``remote`` mode is not even on this machine.
"""

from __future__ import annotations

from typing import Any

from flowcode.samplers.adapters import (
    to_sample_response,
    to_sampled_sequence,
    vllm_sampling_kwargs,
)

__all__ = [
    "SAMPLER_MODES",
    "build_sampler",
    "to_sample_response",
    "to_sampled_sequence",
    "vllm_sampling_kwargs",
]

SAMPLER_MODES = ("colocated", "remote")
"""The deployment topologies. ``colocated`` shares devices with the trainer; ``remote``
reaches an engine already running elsewhere."""


def build_sampler(mode: str, model_name: str, **kwargs: Any) -> Any:
    """Construct the sampler named by ``mode``.

    Imports the implementation lazily: ``colocated`` needs ``vllm`` and ``remote`` needs
    only ``httpx``, so asking for one must not require the other's dependency.

    Args:
        mode: One of :data:`SAMPLER_MODES`.
        model_name: Base model identifier, matching the trainer's.
        **kwargs: Forwarded to the implementation's ``create``. ``remote`` requires
            ``base_url``.

    Returns:
        A sampler satisfying :class:`~flowcode.local_backend.WeightSyncSampler`.

    Raises:
        ValueError: On an unknown mode, or a ``remote`` request without ``base_url``.
    """
    if mode == "colocated":
        from flowcode.samplers.vllm import ColocatedVllmSampler

        return ColocatedVllmSampler.create(model_name, **kwargs)
    if mode == "remote":
        from flowcode.samplers.vllm import RemoteVllmSampler

        base_url = kwargs.pop("base_url", None)
        if not base_url:
            raise ValueError(
                "sampler.mode=remote needs sampler.base_url pointing at the engine's "
                "OpenAI-compatible server, e.g. http://10.0.0.4:8000"
            )
        return RemoteVllmSampler.create(base_url, model_name, **kwargs)
    raise ValueError(f"sampler.mode must be one of {list(SAMPLER_MODES)}, got {mode!r}")
