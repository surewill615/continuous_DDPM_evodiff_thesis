import torch


def expand_dims(v, dims):
    """
    Expand a 1D or 0D tensor to `dims` dimensions.

    Example:
        v.shape = (B,)   dims = 2 -> v.shape = (B, 1)
        v.shape = (B,)   dims = 4 -> v.shape = (B, 1, 1, 1)
    """
    return v[(...,) + (None,) * (dims - 1)]