import numpy as np
from numba import njit, prange
from math import lgamma, log, exp


# -------------------------------------------------------------------------
# xlogy and xlog1py (Numba-JIT safe versions of SciPy's stable primitives)
# -------------------------------------------------------------------------

@njit
def xlogy(a, b):
    """
    Compute a*log(b) stably:
      - returns 0 if a==0
      - returns approx -inf if b<=0
    """
    if a == 0.0:
        return 0.0
    if b <= 0.0:
        return -np.inf   # effectively -inf without producing NaNs
    return a * log(b)


@njit
def xlog1py(a, b):
    """
    Compute a*log(1+b) stably.
    Handles cases where a==0 or 1+b<=0.
    """
    if a == 0.0:
        return 0.0
    t = 1.0 + b
    if t <= 0.0:
        return -np.inf
    return a * log(t)


# -------------------------------------------------------------------------
# Binomial logpmf for k,n,p
# Supports:
#   1) vector k,n and scalar p
#   2) vector k,n and vector p
# -------------------------------------------------------------------------

@njit(parallel=False)
def _logpmf_binom_scalar(k, n, p):
    """
    Compute binomial logpmf for vector k,n with scalar p.
    Output: vector L where L[i] = log Binom(n[i], k[i]) + k[i]*log(p) + ...
    """
    out = np.empty(k.shape, dtype=np.float64)
    for i in prange(k.size):
        # log C(ni, ki)
        comb = (
            lgamma(n[i] + 1.0)
            - (lgamma(k[i] + 1.0) + lgamma(n[i] - k[i] + 1.0))
        )

        out[i] = comb + xlogy(k[i], p) + xlog1py(n[i] - k[i], -p)
    return out


@njit(parallel=False)
def _logpmf_binom_vec(k, n, p):
    """
    Fully elementwise binomial logpmf for vector k, n, p:
      out[i] = Binom(n[i], k[i]) logpmf with p[i].
    """
    out = np.empty(k.shape, dtype=np.float64)
    for i in prange(k.size):

        comb = (
            lgamma(n[i] + 1.0)
            - (lgamma(k[i] + 1.0) + lgamma(n[i] - k[i] + 1.0))
        )

        out[i] = comb + xlogy(k[i], p[i]) + xlog1py(n[i] - k[i], -p[i])
    return out


# Wrappers
def broadcast_binom_args(k, n, p):
    """
    Enforce explicit NumPy broadcasting of k,n,p so that numba kernels
    receive *elementwise matched 1D float arrays*.
    """
    k_arr = np.asarray(k, dtype=np.float64)
    n_arr = np.asarray(n, dtype=np.float64)
    p_arr = np.asarray(p, dtype=np.float64)

    # Let NumPy broadcast naturally
    kb, nb, pb = np.broadcast_arrays(k_arr, n_arr, p_arr)

    # Return flattened 1D versions + original broadcasted shape
    return kb.ravel(), nb.ravel(), pb.ravel(), kb.shape

def logpmf_binom(k, n, p):
    kf, nf, pf, out_shape = broadcast_binom_args(k, n, p)
    out = _logpmf_binom_vec(kf, nf, pf)
    return out.reshape(out_shape)


# -------------------------------------------------------------------------
# LogSumExp utilities (Numba-safe)
# -------------------------------------------------------------------------

@njit
def _logsumexp2_scalar(a, b):
    m = a if a > b else b
    return m + log(exp(a - m) + exp(b - m))


@njit(parallel=False)
def _logsumexp2_matrix(a, b):
    B, L = a.shape
    out = np.empty((B, L), np.float64)
    for i in prange(B):
        for j in range(L):
            A = a[i,j]
            Bv = b[i,j]

            if A == -np.inf and Bv == -np.inf:
                out[i,j] = -np.inf
                continue

            m = A if A > Bv else Bv
            out[i,j] = m + log(exp(A-m) + exp(Bv-m))

    return out

def logsumexp2_matrix(a, b):
    """
    Wrapper ensuring a and b are 2D with identical shape,
    matching SciPy broadcasting for two-term logsumexp mixture.
    """
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)

    # Ensure broadcasting for 1D inputs
    # Case: (B,) and (B,) --> make them (B,1)
    if a_arr.ndim == 1 and b_arr.ndim == 1:
        a_arr = a_arr[:, None]
        b_arr = b_arr[:, None]

    # Case: a is (B,L), b is (B,) or vice versa
    a_arr, b_arr = np.broadcast_arrays(a_arr, b_arr)

    # Ensure now BOTH are 2D
    if a_arr.ndim != 2:
        raise ValueError(f"logsumexp2_matrix expected 2D arrays, got shape {a_arr.shape}")

    return _logsumexp2_matrix(a_arr, b_arr)


@njit(parallel=False)
def _logsumexp_axis2(mat):
    B, A_p, A_c = mat.shape
    out = np.empty((B, A_p), np.float64)
    for i in prange(B):
        for j in range(A_p):
            m = -np.inf
            all_inf = True

            for k in prange(A_c):
                if mat[i,j,k] != -np.inf:
                    all_inf = False
                if mat[i,j,k] > m:
                    m = mat[i,j,k]

            if all_inf:
                out[i,j] = -np.inf
                continue

            s = 0.0
            for k in range(A_c):
                s += exp(mat[i,j,k] - m)

            out[i,j] = m + log(s)
    return out


def logsumexp_axis2(mat):
    mat_arr = np.asarray(mat, dtype=np.float64)
    return _logsumexp_axis2(mat_arr)

