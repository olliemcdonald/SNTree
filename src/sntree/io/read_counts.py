"""Compact, checked storage for nonnegative read counts."""

import numpy as np


READ_COUNT_DTYPE = np.int16
MAX_READ_COUNT = np.iinfo(READ_COUNT_DTYPE).max


def as_read_counts(values, name="Read counts"):
    """Narrow counts without silently wrapping overflow or missing AD values."""
    values = np.asarray(values)
    if values.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain numeric integer counts")
    if values.size:
        low, high = values.min(), values.max()
        if not (0 <= low and high <= MAX_READ_COUNT):
            raise ValueError(
                f"{name} must be between 0 and {MAX_READ_COUNT} for int16 "
                f"storage; observed range [{low}, {high}]. "
                "Check for missing/negative counts or excessive read depth."
            )
        if values.dtype.kind == "f" and np.any(values != np.floor(values)):
            raise ValueError(f"{name} must contain integer counts")
    return values.astype(READ_COUNT_DTYPE, copy=False)
