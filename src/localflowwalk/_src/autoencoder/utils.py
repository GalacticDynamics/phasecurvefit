"""Utility functions for models."""

__all__: tuple[str, ...] = (
    "masked_mean",
    "shuffle_and_batch",
)

import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Bool, Float, PRNGKeyArray, Shaped

from localflowwalk._src.custom_types import FSz0


def masked_mean(arr: Float[Array, " N"], mask: Bool[Array, " N"]) -> FSz0:
    r"""Compute the mean of an array over only the masked elements.

    Parameters
    ----------
    arr : Array, shape (N,)
        Input array.
    mask : Array, shape (N,)
        Binary mask where True = include in mean, False = exclude.

    Returns
    -------
    mean : Array
        Scalar mean value over masked elements.

    """
    n_real = jnp.sum(mask)
    return jnp.sum(arr * mask) / n_real


def shuffle_and_batch(
    mask: Bool[Array, " N"],
    /,
    *args: Shaped[Array, "N ?*shape"],
    key: PRNGKeyArray,
    batch_size: int,
    pad_value: float = 0,
) -> tuple[Bool[Array, "Nb B"], tuple[Float[Array, "Nb B ?*shape"], ...]]:
    r"""Shuffle arrays and batch them with padding mask.

    Separates data into usable (True) and ignorable (False) based on the mask.
    Shuffles within each group independently, then batches with usable data first.

    Parameters
    ----------
    mask : Array, shape (N,)
        Binary mask where True = usable data, False = ignorable data.
    *args : Array
        Variable number of arrays with matching first dimension to shuffle and
        batch. All must have shape (N, ...).
    key : PRNGKeyArray
        JAX random key for deterministic shuffling.
    batch_size : int
        Desired batch size.
    pad_value : float, optional
        Value to use for padding the arrays. Default is 0.

    Returns
    -------
    combined_mask : Array, shape (n_batches, batch_size)
        Binary mask where True = real usable data, False = padding or ignorable data.
    batched_args : tuple of Array
        Shuffled and batched arrays. Each has shape (n_batches, batch_size, ...).
        Usable data appears first, then ignorable data, with padding at the end.

    """
    N = len(mask)

    # Step 1: Sort so True comes first, False comes second
    sort_perm = jnp.argsort(~mask)
    sorted_mask = mask[sort_perm]
    sorted_args = tuple(arr[sort_perm] for arr in args)

    # Step 2: Create shuffle permutation that keeps True and False groups separate
    # Generate random values for shuffling
    rand_vals_true = jr.uniform(key, shape=(N,))
    rand_vals_false = jr.uniform(jr.fold_in(key, 1), shape=(N,))

    # Combine random values: True positions get small values, False get large values
    # This ensures that when we sort, True values come first, False values come second,
    # but within each group they are shuffled
    combined_rand = jnp.where(
        sorted_mask,
        rand_vals_true,  # True positions: small random values
        1.0 + rand_vals_false,  # False positions: large random values
    )

    # Permutation that shuffles within groups
    shuffle_perm = jnp.argsort(combined_rand)

    # Apply permutation to mask and args
    shuffled_mask = sorted_mask[shuffle_perm]
    shuffled_args = tuple(arr[shuffle_perm] for arr in sorted_args)

    # Calculate padding needed for constant batch shape
    n_batches = (N + batch_size - 1) // batch_size  # Ceiling division
    total_padded = n_batches * batch_size
    pad_amount = total_padded - N

    # Helper function to pad and batch arrays with custom pad value
    def pad_and_batch_with_value(
        arr: Float[Array, "N ..."], pad_value: float
    ) -> Float[Array, "Nb Bs ..."]:
        # Pad first dimension
        pad_width = [(0, pad_amount)] + [(0, 0)] * (arr.ndim - 1)
        padded = jnp.pad(arr, pad_width, constant_values=pad_value)
        # Reshape to (n_batches, batch_size, ...)
        return padded.reshape((n_batches, batch_size, *arr.shape[1:]))

    # Pad and batch the args
    batched_args = tuple(
        pad_and_batch_with_value(arr, pad_value=pad_value) for arr in shuffled_args
    )

    # Create padding mask: True for real data, False for padding
    padding_mask = jnp.ones((n_batches, batch_size), dtype=bool)
    if pad_amount > 0:
        padding_mask = padding_mask.at[-1, -pad_amount:].set(False)

    # Batch the usable/ignorable mask
    batched_data_mask = pad_and_batch_with_value(shuffled_mask, pad_value=False)

    # Combine masks: True only where data is real AND usable
    combined_mask = padding_mask & batched_data_mask

    return combined_mask, batched_args
