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
what makes `freeze_encoder` work. `eqx_step` does that split, so each network
supplies only a loss function.

"""

__all__ = ("EqxScanTrainer", "EqxTrainCarry", "eqx_step")

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax.random as jr
import optax
from jaxtyping import Array, Bool, PRNGKeyArray

from jaxmore.nn import AbstractScanNNTrainer

# Carry threaded through the epoch/batch scans. The model is the *combined*
# model; `pack_carry_state` splits it before it reaches `jax.lax.scan`.
type EqxTrainCarry = tuple[eqx.Module, optax.OptState, PRNGKeyArray]


def eqx_step(
    carry: EqxTrainCarry,
    batch_inputs: tuple[Bool[Array, " B"], tuple[Array, ...]],
    *,
    loss_fn: Callable[..., Array],
    optimizer: optax.GradientTransformation,
    filter_spec: Any,
    **kw: Any,
) -> tuple[Array, EqxTrainCarry]:
    """Run one optimisation step on a batch.

    Calls ``loss_fn(model, *data, mask, key=subkey, **kw)`` and differentiates
    it w.r.t. the `filter_spec`-selected (dynamic) half of the model only.
    `EqxScanTrainer` supplies `filter_spec` (see its `prepare_step_kw`), so the
    step, the carry packing and the optimizer state always agree on which
    leaves are trainable.
    """
    model, opt_state, key = carry
    mask, data = batch_inputs
    model_dynamic, model_static = eqx.partition(model, filter_spec)
    key, subkey = jr.split(key)

    @eqx.filter_value_and_grad
    def _loss(dynamic: eqx.Module, /) -> Array:
        model = eqx.combine(dynamic, model_static)
        return loss_fn(model, *data, mask, key=subkey, **kw)

    loss, grads = _loss(model_dynamic)
    updates, opt_state = optimizer.update(grads, opt_state, model_dynamic)
    model_dynamic = eqx.apply_updates(model_dynamic, updates)
    return loss, (eqx.combine(model_dynamic, model_static), opt_state, key)


@dataclass(frozen=True)
class EqxScanTrainer(AbstractScanNNTrainer):
    """Scan trainer for Equinox models, partitioned by `filter_spec`.

    `make_step` and `loss_agg_fn` are constructor arguments (see
    `jaxmore.nn.AbstractScanNNTrainer`); `make_step` is usually a partial of
    `eqx_step`.

    Attributes
    ----------
    filter_spec
        How to split the model into trainable and frozen parts. Defaults to
        `eqx.is_array` (train everything). Pass a boolean pytree to freeze a
        subtree.

    """

    filter_spec: Any = eqx.is_array

    def init(  # type: ignore[override]
        self,
        model: eqx.Module,
        data: tuple[Array, ...],
        mask: Bool[Array, " N"],
        /,
        *,
        optimizer: optax.GradientTransformation,
        key: PRNGKeyArray,
    ) -> tuple[EqxTrainCarry, tuple[Bool[Array, " N"], tuple[Array, ...]]]:
        """Build the initial carry and the epoch data ``(mask, data)``."""
        opt_state = optimizer.init(eqx.filter(model, self.filter_spec))
        return (model, opt_state, key), (mask, data)

    def prepare_step_kw(
        self, /, *, epoch_idx: Array, num_epochs: int, epoch_key: PRNGKeyArray
    ) -> Mapping[str, Any]:
        """Pass `filter_spec` to `make_step`: one source of truth for the split.

        Subclasses that add per-epoch kwargs must merge in ``super()``'s.
        """
        del epoch_idx, num_epochs, epoch_key
        return {"filter_spec": self.filter_spec}

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
