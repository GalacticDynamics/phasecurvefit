"""Shared scan-over-epoch trainer base, built on `jaxmore.nn`.

All three networks in this package (`OrderingNet`, `TrackNet`, and the joint
`PathAutoencoder`) train with the same structure: partition the Equinox model
into trainable arrays and static structure, scan over epochs, shuffle and batch
inside each epoch, scan over batches, and aggregate the per-batch losses.

`jaxmore.nn.AbstractScanNNTrainer` owns that structure. This module supplies the
one piece it delegates to us -- how to split an Equinox model into its dynamic
and static halves -- so the individual networks only have to say what a training
step *is*.

Notes
-----
The carry is ``(model, opt_state, key)`` with the **full** model. Packing
partitions it; unpacking recombines it. `eqx.partition` and `eqx.combine` are
pytree manipulations performed at trace time, so this costs nothing at runtime;
it just keeps the abstraction honest, and means `make_step` always receives a
model it can actually call.

Gradients must still be taken with respect to the *dynamic* half only -- that is
what makes `freeze_encoder` work. Use `partitioned` to recover the split inside
a step function.

"""

__all__ = ("AbstractEqxScanTrainer", "EqxTrainCarry")

from dataclasses import dataclass
from typing import Any

import equinox as eqx
import optax
from jaxtyping import PRNGKeyArray

from jaxmore.nn import AbstractScanNNTrainer

# Carry threaded through the epoch/batch scans. The model is the *combined*
# model; `pack_carry_state` splits it before it reaches `jax.lax.scan`.
type EqxTrainCarry = tuple[eqx.Module, optax.OptState, PRNGKeyArray]


@dataclass(frozen=True)
class AbstractEqxScanTrainer(AbstractScanNNTrainer):
    """Scan trainer for Equinox models, partitioned by `filter_spec`.

    Subclasses supply `init`; `make_step` and `loss_agg_fn` are constructor
    arguments (see `jaxmore.nn.AbstractScanNNTrainer`).

    Attributes
    ----------
    filter_spec
        How to split the model into trainable and frozen parts. Defaults to
        `eqx.is_array` (train everything). Pass a boolean pytree to freeze a
        subtree.

    """

    filter_spec: Any = eqx.is_array

    def pack_carry_state(
        self, carry: EqxTrainCarry
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Partition the model so the scan carry holds arrays only."""
        model, opt_state, key = carry
        model_dynamic, model_static = eqx.partition(model, self.filter_spec)
        return (model_dynamic, opt_state, key), {"model_static": model_static}

    def unpack_carry_state(
        self, carry: tuple[Any, ...], static: dict[str, Any] | None
    ) -> EqxTrainCarry:
        """Recombine the model so `make_step` receives a callable model.

        Raises
        ------
        ValueError
            If `static` does not carry the ``"model_static"`` produced by
            `pack_carry_state`. The two are a matched pair; a missing key means
            the trainer was wired up wrong, so fail loudly rather than
            recombining against an empty static half.

        """
        model_dynamic, opt_state, key = carry
        if static is None or "model_static" not in static:
            msg = (
                "expected 'model_static' in the static state from "
                f"`pack_carry_state()`, got {static!r}"
            )
            raise ValueError(msg)
        model = eqx.combine(model_dynamic, static["model_static"])
        return (model, opt_state, key)
