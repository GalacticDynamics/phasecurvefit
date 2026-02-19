r"""Scan-over MLP."""

__all__: tuple[str, ...] = ("ScanOverMLP",)

from collections.abc import Callable
from typing import Literal

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.nn as jnn
import jax.random as jr
import jax.tree as jtu
from jaxtyping import Array, PRNGKeyArray

_identity = eqxi.doc_repr(lambda x: x, "lambda x: x")
_relu = eqxi.doc_repr(jnn.relu, "<function relu>")


# TODO: upstream to equinox
class ScanOverMLP(eqx.nn.MLP):
    r"""Multi-layer perceptron with scan-over-layers pattern for fast compilation.

    Similar to ``equinox.nn.MLP``, but uses ``jax.lax.scan`` to iterate over
    identical hidden layers for improved compilation speed.

    The network consists of three components:
    - Input layer: maps in_size -> width_size
    - Hidden layers: scan-over-layers pattern, width_size -> width_size
    (repeated depth - 1 times)
    - Output layer: maps width_size -> out_size

    See: https://docs.kidger.site/equinox/tricks/#improve-compilation-speed-with-scan-over-layers

    """

    layers: tuple[eqx.nn.Linear, object, eqx.nn.Linear]
    activation: Callable
    final_activation: Callable

    use_bias: bool = eqx.field(static=True)
    use_final_bias: bool = eqx.field(static=True)
    in_size: int | Literal["scalar"] = eqx.field(static=True)
    out_size: int | Literal["scalar"] = eqx.field(static=True)

    width_size: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)

    def __init__(
        self,
        in_size: int | Literal["scalar"],
        out_size: int | Literal["scalar"],
        width_size: int,
        depth: int,
        activation: Callable = _relu,
        final_activation: Callable = _identity,
        use_bias: bool = True,  # noqa: FBT001,FBT002
        use_final_bias: bool = True,  # noqa: FBT001,FBT002
        dtype: jax.numpy.dtype | None = None,
        *,
        key: PRNGKeyArray,
    ) -> None:
        if depth < 1:
            msg = "depth must be at least 1"
            raise ValueError(msg)

        del dtype  # Currently unused
        keys = jr.split(key, depth + 1)

        # Input layer
        input_layer = eqx.nn.Linear(in_size, width_size, use_bias=use_bias, key=keys[0])

        # Hidden layers: create depth-1 identical layers using filter_vmap
        def make_hidden_layer(k: PRNGKeyArray) -> eqx.nn.Linear:
            return eqx.nn.Linear(width_size, width_size, use_bias=use_bias, key=k)

        hidden_keys = keys[1:depth]
        if depth > 1:
            hidden_keys = keys[1:depth]
            hidden_layers = eqx.filter_vmap(make_hidden_layer)(hidden_keys)
        else:
            # For depth == 1, construct an empty collection of hidden layers
            # with the correct tree structure and array dtypes/shapes.
            single_hidden = eqx.filter_vmap(make_hidden_layer)(keys[1:2])
            hidden_layers = jtu.map(lambda x: x[:0], single_hidden)

        # Output layer
        output_layer = eqx.nn.Linear(
            width_size, out_size, use_bias=use_final_bias, key=keys[-1]
        )

        # Store as tuple following Equinox MLP pattern
        self.layers = (input_layer, hidden_layers, output_layer)

        self.in_size = in_size
        self.width_size = width_size
        self.out_size = out_size
        self.depth = depth
        # In case `activation` or `final_activation` are learnt, then make a
        # separate copy of their weights for every neuron.
        self.activation = eqx.filter_vmap(
            eqx.filter_vmap(lambda: activation, axis_size=width_size), axis_size=depth
        )()
        if out_size == "scalar":
            self.final_activation = final_activation
        else:
            self.final_activation = eqx.filter_vmap(
                lambda: final_activation, axis_size=out_size
            )()
        self.use_bias = use_bias
        self.use_final_bias = use_final_bias

    def __call__(self, x: Array, *, key: PRNGKeyArray | None = None) -> Array:
        """Forward pass through the MLP.

        Parameters
        ----------
        x : Array
            Input of shape (in_size,).
        key : PRNGKeyArray | None
            Optional JAX random key (not used).

        Returns
        -------
        out : Array
            Output of shape (out_size,) after applying activation to all layers.

        """
        del key

        input_layer, hidden_layers, output_layer = self.layers

        # Input layer + activation
        x = input_layer(x)
        # Extract the first activation (index 0) from the vmapped tree
        layer_activation = jtu.map(
            lambda act: act[0] if eqx.is_array(act) else act, self.activation
        )
        x = eqx.filter_vmap(lambda a, b: a(b))(layer_activation, x)

        # Scan over hidden layers
        dynamic, static = eqx.partition(hidden_layers, eqx.is_array)

        def scan_fn(
            carry: tuple[Array, int | Array], layer_params: eqx.nn.Linear
        ) -> tuple[tuple[Array, int | Array], None]:
            x, layer_idx = carry  # Unpack carry
            # Evaluate the layer
            layer = eqx.combine(layer_params, static)
            x = layer(x)
            # Extract and apply layer_idx-th activation from the vmapped tree
            layer_activation = jtu.map(
                lambda act: act[layer_idx + 1] if eqx.is_array(act) else act,
                self.activation,
            )
            x = eqx.filter_vmap(lambda a, b: a(b))(layer_activation, x)
            # Return the output and incremented index
            return (x, layer_idx + 1), None

        (x, _), _ = jax.lax.scan(scan_fn, (x, 0), dynamic)

        # Output layer + final activation
        x = output_layer(x)
        if self.out_size == "scalar":
            x = self.final_activation(x)
        else:
            x = eqx.filter_vmap(lambda a, b: a(b))(self.final_activation, x)

        return x
