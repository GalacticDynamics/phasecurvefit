"""Interop with unxt Quantity for phasespace functions.

This module registers plum dispatch overloads for phasespace functions to work
with unxt Quantity-valued component dictionaries. This enables seamless use of
physical units throughout phase-space calculations.

When unxt is installed, these dispatches automatically handle:
- Distance calculations preserving units
- Direction vectors (unitless by nature)
- Velocity norms with proper unit handling
- Cosine similarity normalized correctly

Examples
--------
>>> import jax.numpy as jnp
>>> import unxt as u
>>> from localflowwalk._src.phasespace import euclidean_distance

With Quantity-valued components:

>>> q_a = {"x": u.Q(0.0, "m"), "y": u.Q(0.0, "m")}
>>> q_b = {"x": u.Q(3.0, "m"), "y": u.Q(4.0, "m")}
>>> result = euclidean_distance(q_a, q_b)
>>> float(result.to("m").value)
5.0

"""

__all__: tuple[str, ...] = ()

from collections.abc import Mapping
from typing import Any, TypeAlias

import jax.numpy as jnp
import plum
import quax
from jax import lax
from jaxtyping import Array, ArrayLike, Float, Real

import dataclassish
import quaxed.numpy as qnp
import unxt as u
from unxt import AbstractQuantity as AbcQ
from unxt.quantity import AllowValue

from localflowwalk._src import algorithm, phasespace
from localflowwalk._src.algorithm import Direction, LocalFlowWalkResult, StateMetadata
from localflowwalk._src.autoencoder.normalize import StandardScalerNormalizer
from localflowwalk._src.custom_types import VectorComponents
from localflowwalk._src.query_config import WalkConfig

RQSz0: TypeAlias = Real[AbcQ, " "]  # noqa: UP040
ScalarQComponents: TypeAlias = Mapping[str, RQSz0]  # noqa: UP040
VectorQComponents: TypeAlias = Mapping[str, Real[AbcQ, " N"]]  # noqa: UP040


# ==============================================================================
# JAX primitives dispatch for Quantity handling
# ==============================================================================


def _scan_p_helper(
    usys: u.AbstractUnitSystem,
    q: tuple[AbcQ, ...],
    p: tuple[AbcQ, ...],
    terminate_arr: ArrayLike,
    *args: Any,
    kw: dict[str, Any],
) -> tuple[Array, Array, Array, Array, Array]:
    (
        out_ordered,
        out_mask,
        out_best_idx,
        out_step,
        out_should_stop,
        _,  # metadata placeholder
    ) = lax.scan_p.bind(  # type: ignore[no-untyped-call]
        jnp.array(True),  # noqa: FBT003
        terminate_arr,
        *[u.ustrip(usys, x) for x in q],
        *[u.ustrip(usys, v) for v in p],
        *[u.ustrip(AllowValue, usys, arg) for arg in args],
        **kw,
    )
    return (out_ordered, out_mask, out_best_idx, out_step, out_should_stop)


@quax.register(lax.scan_p)
def scan_p_statemetadata_quantity(
    metadata: StateMetadata,
    terminate_arr: ArrayLike,
    q_x: AbcQ,
    p_x: AbcQ,
    metric_scale: AbcQ,
    max_dist: AbcQ,
    ordered_arr: ArrayLike,
    visited_mask: ArrayLike,
    current_idx: int,
    step: int,
    should_stop: bool,  # noqa: FBT001
    arg0: ArrayLike,  # what is this?
    /,
    **kw: Any,
) -> list[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike, StateMetadata]:
    """Handle ``lax.scan`` when StateMetadata is the leading carry element.

    Quax flattens the bounded_while_loop carry into positional args. In the
    current flattening order we observe the metadata as the first positional
    argument, followed by ``xs`` (typically an empty array for our scan), any
    const arguments, and finally the remaining state elements plus the ``done``
    flag. We peel the state off the tail, strip Quantities, call the primitive
    implementation, and rewrap.
    """
    usys = metadata.get("usys")
    if usys is None:
        msg = "No unit system found in StateMetadata"
        raise RuntimeError(msg)

    (
        out_ordered,
        out_mask,
        out_best_idx,
        out_step,
        out_should_stop,
    ) = _scan_p_helper(
        usys,
        (q_x,),
        (p_x,),
        terminate_arr,
        metric_scale,
        max_dist,
        ordered_arr,
        visited_mask,
        current_idx,
        step,
        should_stop,
        arg0,
        kw=kw,
    )
    return (out_ordered, out_mask, out_best_idx, out_step, out_should_stop, metadata)


@quax.register(lax.scan_p)
def scan_p_statemetadata_quantity(
    metadata: StateMetadata,
    terminate_arr: ArrayLike,
    q_x: AbcQ,
    q_y: AbcQ,
    p_x: AbcQ,
    p_y: AbcQ,
    metric_scale: AbcQ,
    max_dist: AbcQ,
    ordered_arr: ArrayLike,
    visited_mask: ArrayLike,
    current_idx: int,
    step: int,
    should_stop: bool,  # noqa: FBT001
    arg0: ArrayLike,  # what is this?
    /,
    **kw: Any,
) -> list[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike, StateMetadata]:
    """Handle ``lax.scan`` when StateMetadata is the leading carry element.

    Quax flattens the bounded_while_loop carry into positional args. In the
    current flattening order we observe the metadata as the first positional
    argument, followed by ``xs`` (typically an empty array for our scan), any
    const arguments, and finally the remaining state elements plus the ``done``
    flag. We peel the state off the tail, strip Quantities, call the primitive
    implementation, and rewrap.
    """
    usys = metadata.get("usys")
    if usys is None:
        msg = "No unit system found in StateMetadata"
        raise RuntimeError(msg)

    (
        out_ordered,
        out_mask,
        out_best_idx,
        out_step,
        out_should_stop,
    ) = _scan_p_helper(
        usys,
        (q_x, q_y),
        (p_x, p_y),
        terminate_arr,
        metric_scale,
        max_dist,
        ordered_arr,
        visited_mask,
        current_idx,
        step,
        should_stop,
        arg0,
        kw=kw,
    )
    return (out_ordered, out_mask, out_best_idx, out_step, out_should_stop, metadata)


@quax.register(lax.scan_p)
def scan_p_statemetadata_quantity(
    metadata: StateMetadata,
    terminate_arr: ArrayLike,
    q_x: AbcQ,
    q_y: AbcQ,
    q_z: AbcQ,
    p_x: AbcQ,
    p_y: AbcQ,
    p_z: AbcQ,
    metric_scale: AbcQ,
    max_dist: AbcQ,
    ordered_arr: ArrayLike,
    visited_mask: ArrayLike,
    current_idx: int,
    step: int,
    should_stop: bool,  # noqa: FBT001
    arg0: ArrayLike,  # what is this?
    /,
    **kw: Any,
) -> list[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike, StateMetadata]:
    """Handle ``lax.scan`` when StateMetadata is the leading carry element.

    Quax flattens the bounded_while_loop carry into positional args. In the
    current flattening order we observe the metadata as the first positional
    argument, followed by ``xs`` (typically an empty array for our scan), any
    const arguments, and finally the remaining state elements plus the ``done``
    flag. We peel the state off the tail, strip Quantities, call the primitive
    implementation, and rewrap.
    """
    usys = metadata.get("usys")
    if usys is None:
        msg = "No unit system found in StateMetadata"
        raise RuntimeError(msg)

    (
        out_ordered,
        out_mask,
        out_best_idx,
        out_step,
        out_should_stop,
    ) = _scan_p_helper(
        usys,
        (q_x, q_y, q_z),
        (p_x, p_y, p_z),
        terminate_arr,
        metric_scale,
        max_dist,
        ordered_arr,
        visited_mask,
        current_idx,
        step,
        should_stop,
        arg0,
        kw=kw,
    )
    return (out_ordered, out_mask, out_best_idx, out_step, out_should_stop, metadata)


# ------------------------------------------------------
# for KDTree


# TODO: determinet the details of this
@quax.register(lax.scan_p)
def scan_p_qvvvvv(
    pos: AbcQ,
    arg1: ArrayLike,
    arg2: ArrayLike,
    arg3: ArrayLike,
    arg4: ArrayLike,
    arg5: ArrayLike,
    **kw: Any,
) -> list:
    out_v = lax.scan_p.bind(u.ustrip(pos), arg1, arg2, arg3, arg4, arg5, **kw)
    return out_v  # noqa: RET504


@quax.register(lax.scan_p)
def scan_p_statemetadata_quantity(
    metadata: StateMetadata,
    terminate_arr: ArrayLike,
    q_x: AbcQ,
    q_y: AbcQ,
    p_x: AbcQ,
    p_y: AbcQ,
    q_xy: AbcQ,
    ordered_arr: ArrayLike,
    visited_mask: ArrayLike,
    metric_scale: AbcQ,
    max_dist: AbcQ,
    arg0: ArrayLike,
    arg1: ArrayLike,
    current_idx: int,
    step: int,
    should_stop: bool,  # noqa: FBT001
    arg2: ArrayLike,  # what is this?
    /,
    **kw: Any,
) -> list[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike, StateMetadata]:
    """Handle ``lax.scan`` when StateMetadata is the leading carry element.

    Quax flattens the bounded_while_loop carry into positional args. In the
    current flattening order we observe the metadata as the first positional
    argument, followed by ``xs`` (typically an empty array for our scan), any
    const arguments, and finally the remaining state elements plus the ``done``
    flag. We peel the state off the tail, strip Quantities, call the primitive
    implementation, and rewrap.
    """
    usys = metadata.get("usys")
    if usys is None:
        msg = "No unit system found in StateMetadata"
        raise RuntimeError(msg)

    (
        out_ordered,
        out_mask,
        out_best_idx,
        out_step,
        out_should_stop,
    ) = _scan_p_helper(
        usys,
        (q_x, q_y),
        (p_x, p_y),
        terminate_arr,
        q_xy,
        ordered_arr,
        visited_mask,
        metric_scale,
        max_dist,
        arg0,
        arg1,
        current_idx,
        step,
        should_stop,
        arg2,
        kw=kw,
    )
    return (out_ordered, out_mask, out_best_idx, out_step, out_should_stop, metadata)


@quax.register(lax.scan_p)
def scan_p_statemetadata_quantity(
    metadata: StateMetadata,
    terminate_arr: ArrayLike,
    q_x: AbcQ,
    q_y: AbcQ,
    q_z: AbcQ,
    p_x: AbcQ,
    p_y: AbcQ,
    p_z: AbcQ,
    q_xyz: AbcQ,
    ordered_arr: ArrayLike,
    visited_mask: ArrayLike,
    metric_scale: AbcQ,
    max_dist: AbcQ,
    arg0: ArrayLike,
    arg1: ArrayLike,
    current_idx: int,
    step: int,
    should_stop: bool,  # noqa: FBT001
    arg2: ArrayLike,  # what is this?
    /,
    **kw: Any,
) -> list[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike, StateMetadata]:
    """Handle ``lax.scan`` when StateMetadata is the leading carry element.

    Quax flattens the bounded_while_loop carry into positional args. In the
    current flattening order we observe the metadata as the first positional
    argument, followed by ``xs`` (typically an empty array for our scan), any
    const arguments, and finally the remaining state elements plus the ``done``
    flag. We peel the state off the tail, strip Quantities, call the primitive
    implementation, and rewrap.
    """
    usys = metadata.get("usys")
    if usys is None:
        msg = "No unit system found in StateMetadata"
        raise RuntimeError(msg)

    (
        out_ordered,
        out_mask,
        out_best_idx,
        out_step,
        out_should_stop,
    ) = _scan_p_helper(
        usys,
        (q_x, q_y, q_z),
        (p_x, p_y, p_z),
        terminate_arr,
        q_xyz,
        ordered_arr,
        visited_mask,
        metric_scale,
        max_dist,
        arg0,
        arg1,
        current_idx,
        step,
        should_stop,
        arg2,
        kw=kw,
    )
    return (out_ordered, out_mask, out_best_idx, out_step, out_should_stop, metadata)


# ------------------------------------------------------


@quax.register(lax.scatter_p)
def scatter_p_quantity(
    operand: AbcQ, scatter_indices: Array, updates: AbcQ, /, **kw: Any
) -> AbcQ:
    """Handle ``lax.scatter`` when both operand and updates are Quantities.

    Strips units, applies the scatter, then rebuilds the Quantity with the
    original unit.
    """
    unit = operand.unit
    operand_ = u.ustrip(operand)
    updates_ = u.ustrip(unit, updates)
    out = lax.scatter_p.impl(operand_, scatter_indices, updates_, **kw)  # type: ignore[no-untyped-call]
    return operand.__class__(out, unit)


@quax.register(lax.scatter_p)
def scatter_p_array_quantity(
    operand: Array, scatter_indices: Array, updates: AbcQ, /, **kw: Any
) -> Array:
    """Handle ``lax.scatter`` when operand is array and updates are Quantities."""
    unit = updates.unit
    updates_ = u.ustrip(unit, updates)
    return lax.scatter_p.impl(operand, scatter_indices, updates_, **kw)  # type: ignore[no-untyped-call]


@plum.dispatch
def euclidean_distance(q_a: ScalarQComponents, q_b: ScalarQComponents, /) -> RQSz0:
    """Euclidean distance between Quantity-valued component dictionaries.

    Computes the distance between two phase-space positions represented as
    dictionaries with unxt Quantity scalar values.

    Parameters
    ----------
    q_a, q_b : Mapping[str, unxt.AbstractQuantity]
        Position dictionaries with Quantity-valued components. Must have the
        same keys. All values must have compatible length dimensions.

    Returns
    -------
    unxt.Quantity
        The Euclidean distance with the unit of the input components.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> q_a = {"x": u.Q(0.0, "m"), "y": u.Q(0.0, "m")}
    >>> q_b = {"x": u.Q(3.0, "m"), "y": u.Q(4.0, "m")}
    >>> euclidean_distance(q_a, q_b)
    Quantity(Array(5., dtype=float32, weak_type=True), unit='m')

    """
    return quax.quaxify(phasespace.euclidean_distance)(q_a, q_b)


@plum.dispatch
def unit_direction(
    q_a: ScalarQComponents, q_b: ScalarQComponents, /
) -> ScalarQComponents:
    """Compute unit direction vector from q_a to q_b for Quantity-valued components.

    Computes the unit direction vector pointing from position `q_a` to `q_b`,
    where both positions are represented as dictionaries with unxt Quantity
    scalar values.

    Parameters
    ----------
    q_a, q_b : Mapping[str, unxt.AbstractQuantity]
        Position dictionaries with Quantity-valued components. Must have the
        same keys. All values must have compatible length dimensions.

    Returns
    -------
    Mapping[str, unxt.AbstractQuantity]
        A dictionary representing the unit direction vector. The components
        are dimensionless Quantities.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> q_a = {"x": u.Q(0.0, "m"), "y": u.Q(0.0, "m")}
    >>> q_b = {"x": u.Q(3.0, "m"), "y": u.Q(4.0, "m")}
    >>> unit_direction(q_a, q_b)
    {'x': Quantity(Array(0.6, dtype=float32, weak_type=True), unit=''),
     'y': Quantity(Array(0.8, dtype=float32, weak_type=True), unit='')}

    """
    return quax.quaxify(phasespace.unit_direction)(q_a, q_b)


@plum.dispatch
def velocity_norm(velocity: ScalarQComponents, /) -> RQSz0:
    """Compute the norm of a Quantity-valued velocity vector.

    Computes the Euclidean norm of a velocity vector represented as a
    dictionary with unxt Quantity scalar values.

    Parameters
    ----------
    velocity : Mapping[str, unxt.AbstractQuantity]
        Velocity dictionary with Quantity-valued components. All values must
        have compatible velocity dimensions (length/time).

    Returns
    -------
    unxt.Quantity
        The Euclidean norm of the velocity with appropriate units.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> vel = {"x": u.Q(3.0, "m/s"), "y": u.Q(4.0, "m/s")}
    >>> result = velocity_norm(vel)
    >>> result.to("m/s").value
    Array(5., dtype=float32, weak_type=True)

    """
    return quax.quaxify(phasespace.velocity_norm)(velocity)


@plum.dispatch
def unit_velocity(velocity: ScalarQComponents, /) -> ScalarQComponents:
    """Compute unit velocity vector for Quantity-valued components.

    Computes the unit velocity vector from a velocity represented as a
    dictionary with unxt Quantity scalar values.

    Parameters
    ----------
    velocity : Mapping[str, unxt.AbstractQuantity]
        Velocity dictionary with Quantity-valued components. All values must
        have compatible velocity dimensions (length/time).

    Returns
    -------
    Mapping[str, unxt.AbstractQuantity]
        A dictionary representing the unit velocity vector. The components
        are dimensionless Quantities.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> vel = {"x": u.Q(3.0, "m/s"), "y": u.Q(4.0, "m/s")}
    >>> unit_velocity(vel)
    {'x': Quantity(Array(0.6, dtype=float32, ...), unit=''),
     'y': Quantity(Array(0.8, dtype=float32, ...), unit='')}

    """
    return quax.quaxify(phasespace.unit_velocity)(velocity)


@plum.dispatch
def cosine_similarity(vel_a: ScalarQComponents, vel_b: ScalarQComponents, /) -> RQSz0:
    """Compute cosine similarity between Quantity-valued velocity components.

    Computes the cosine similarity (dimensionless) between two vectors
    represented as dictionaries with unxt Quantity scalar values.
    The result is the cosine of the angle between the two vectors.

    Parameters
    ----------
    vel_a, vel_b : Mapping[str, unxt.AbstractQuantity]
        Velocity or direction dictionaries with Quantity-valued components.
        Must have the same keys. All values must have compatible dimensions.

    Returns
    -------
    unxt.Quantity
        The dimensionless cosine similarity between the two vectors.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> vel_a = {"x": u.Q(1.0, "m/s"), "y": u.Q(0.0, "m/s")}
    >>> vel_b = {"x": u.Q(0.0, "m/s"), "y": u.Q(1.0, "m/s")}
    >>> cosine_similarity(vel_a, vel_b)
    Quantity(Array(0., dtype=float32, ...), unit='')

    """
    return quax.quaxify(phasespace.cosine_similarity)(vel_a, vel_b)


_walk_local_flow = algorithm.walk_local_flow.invoke(VectorComponents, VectorComponents)


@plum.dispatch
def walk_local_flow(
    positions: VectorQComponents,
    velocities: VectorQComponents,
    /,
    *,
    start_idx: int,
    metric_scale: RQSz0,
    max_dist: RQSz0 = u.Q(jnp.inf, "m"),  # noqa: B008
    terminate_indices: set[int] | None = None,
    n_max: int | None = None,
    config: WalkConfig = WalkConfig(),  # noqa: B008
    direction: Direction = "forward",
    metadata: StateMetadata = StateMetadata(),  # noqa: B008
    usys: u.AbstractUnitSystem | None = None,
) -> LocalFlowWalkResult:
    """Implement for Quantity-valued phase-space data.

    Parameters
    ----------
    positions : Mapping[str, unxt.AbstractQuantity]
        Position dictionary with Quantity-valued components. All values must
        have compatible length dimensions.
    velocities : Mapping[str, unxt.AbstractQuantity]
        Velocity dictionary with Quantity-valued components. All values must
        have compatible velocity dimensions (length/time).
    start_idx
        Index of the starting observation.
    metric_scale
        Metric-dependent scale parameter.
    max_dist
        Maximum allowed distance for neighbor selection. Observations beyond
        this distance are not considered. Default is infinity.
    terminate_indices : set[int], optional
        Set of observation indices that, when reached, terminate the ordering.
    n_max : int, optional
        Maximum number of observations to include in the ordering. If None,
        all observations are included.
    config : WalkConfig
        Configuration for neighbor queries, containing both the distance metric
        and the query strategy. Use ``WalkConfig(metric=..., strategy=...)`` to
        customize. Defaults to ``WalkConfig()`` which uses
        ``FullPhaseSpaceDistanceMetric`` with ``BruteForce``.
    direction : ['forward', 'backward', 'both'], optional
        Direction to walk the local flow. 'forward' walks along the velocity
        field, 'backward' walks against the velocity field, and 'both' walks in
        both directions.  Default is 'forward'.

    metadata
        Optional metadata to pass through the algorithm state without
        participating in computation. Useful for unit systems or other context.
    usys : unxt.AbstractUnitSystem, optional
        Unit system to use for consistent unit stripping of Quantities. Default
        is SI units.

    Returns
    -------
    LocalFlowWalkResult
        Result container with ordered indices and original data.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> q = {
    ...     "x": u.Q(jnp.array([0.0, 1.0, 2.0]), "m"),
    ...     "y": u.Q(jnp.array([0.0, 0.5, 1.0]), "m"),
    ... }
    >>> p = {
    ...     "x": u.Q(jnp.array([1.0, 1.0, 1.0]), "m/s"),
    ...     "y": u.Q(jnp.array([0.5, 0.5, 0.5]), "m/s"),
    ... }
    >>> result = walk_local_flow(
    ...     q, p, start_idx=0, metric_scale=u.Q(1.0, "m"), usys=u.unitsystems.si
    ... )
    >>> result
    LocalFlowWalkResult(indices=Array([0, 1, 2], dtype=int32),
        positions={'x': Quantity(Array([0., 1., 2.], dtype=float32), unit='m'),
                   'y': Quantity(Array([0. , 0.5, 1. ], dtype=float32), unit='m')},
        velocities={'x': Quantity(Array([1., 1., 1.], dtype=float32), unit='m / s'),
                    'y': Quantity(Array([0.5, 0.5, 0.5], dtype=float32), unit='m / s')})

    """
    # Process the metadata
    if usys is not None:
        metadata = StateMetadata(**(dict(metadata) | {"usys": usys}))

    usys = metadata.get("usys")
    if not isinstance(usys, u.AbstractUnitSystem):
        msg = "`usys` must be an `unxt.AbstractUnitSystem`."  # type: ignore[unreachable]
        raise TypeError(msg)

    if not isinstance(metric_scale, u.AbstractQuantity):
        msg = "`metric_scale` must be an `unxt.AbstractQuantity`."  # type: ignore[unreachable]
        raise TypeError(msg)
    metric_scale = u.ustrip(usys, metric_scale)
    if not isinstance(max_dist, u.AbstractQuantity):
        msg = "`max_dist` must be an `unxt.AbstractQuantity`."  # type: ignore[unreachable]
        raise TypeError(msg)
    max_dist = u.ustrip(usys, max_dist)

    q_values = {k: u.ustrip(usys, v) for k, v in positions.items()}
    p_values = {k: u.ustrip(usys, v) for k, v in velocities.items()}

    # Quaxify the walk_local_flow so Quantities are handled properly
    # by custom dispatches in the quax context. StateMetadata is part of init
    # so the scan_p handler can dispatch on it directly.
    result: LocalFlowWalkResult = quax.quaxify(_walk_local_flow)(
        q_values,
        p_values,
        start_idx=start_idx,
        metric_scale=metric_scale,
        max_dist=max_dist,
        terminate_indices=terminate_indices,
        n_max=n_max,
        config=config,
        metadata=metadata,
        direction=direction,
    )
    return dataclassish.replace(result, positions=positions, velocities=velocities)


@StandardScalerNormalizer.__init__.dispatch
def init(
    self: "StandardScalerNormalizer",
    qs: VectorQComponents,
    ps: VectorQComponents,
    /,
    **kwargs: object,
) -> None:
    self.q_comps = list(qs.keys())
    xs = qnp.stack(list(qs.values()), axis=1)
    self.q_mean = qnp.mean(xs, axis=0)
    self.q_std = qnp.std(xs, axis=0) + qnp.ones_like(self.q_mean) * 1e-8

    self.p_comps = list(ps.keys())
    vs = qnp.stack(list(ps.values()), axis=1)
    self.p_mean = qnp.mean(vs, axis=0)
    self.p_std = qnp.std(vs, axis=0) + qnp.ones_like(self.p_mean) * 1e-8


@StandardScalerNormalizer.transform.dispatch
def transform(
    self: "StandardScalerNormalizer",
    qs: VectorQComponents,
    ps: VectorQComponents,
) -> tuple[Float[Array, "N D"], Float[Array, "N D"]]:
    xs = qnp.stack([qs[k] for k in self.q_comps], axis=1)
    norm_qs = (xs - self.q_mean) / self.q_std
    vs = qnp.stack([ps[k] for k in self.p_comps], axis=1)
    norm_ps = (vs - self.p_mean) / self.p_std
    return norm_qs.ustrip(""), norm_ps.ustrip("")
