import numpy as np
def cholesky_decomposition(A, *, jitter=0.0, check_sym=True):
    """
    Returns the lower-triangular Cholesky factor L such that A ≈ L @ L.T.

    Parameters
    ----------
    A : array_like (n, n)
        Symmetric positive definite matrix.
    jitter : float
        If > 0, adds jitter * I to A (useful if A is nearly PSD / numerical issues).
    check_sym : bool
        If True, checks symmetry (within tolerance) before decomposing.

    Returns
    -------
    L : ndarray (n, n)
        Lower-triangular Cholesky factor.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")

    if check_sym and not np.allclose(A, A.T, rtol=1e-10, atol=1e-12):
        raise ValueError("A must be symmetric (within numerical tolerance)")

    if jitter > 0.0:
        A = A + jitter * np.eye(A.shape[0])

    # NumPy returns lower-triangular by default
    L = np.linalg.cholesky(A)
    return L


# --- Example ---
if __name__ == "__main__":
    A = np.array([[4, 2, 2],
                  [2, 5, 1],
                  [2, 1, 3]], dtype=float)

    L = cholesky_decomposition(A)
    print("L =\n", L)
    print("Check (L @ L.T) =\n", L @ L.T)
    print("Max abs error:", np.max(np.abs(A - L @ L.T)))
