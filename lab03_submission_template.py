"""Lab 03 portfolio submission template.

Submit this single file to Gradescope once you've filled in the two
functions marked TODO below. Do not rename any of the functions - the
autograder looks them up by name.

What you need to do:
  1. Implement `lu_factorisation`. This must be your own implementation -
     do not import `scipy.linalg` or `sympy` or call a library
     decomposition routine; this exercise is assessed on your own
     implementation of the algorithm.
  2. Leave `row_add`/`gaussian_elimination` and `determinant` as given -
     they're provided code from the lab sheet, not something you write.
  3. Implement `runtime_comparison`. It must time your `lu_factorisation`
     against `gaussian_elimination` across a range of matrix sizes, produce
     a log-log plot of the two, and save it to "plot.png" in the working
     directory - the autograder looks for that exact filename.
"""

import numpy as np
from scipy.sparse import diags


def generate_safe_system(n):
    """
    Generate a linear system A x = b where A is strictly diagonally dominant,
    ensuring LU factorization without pivoting will work.

    Parameters:
        n (int): Size of the system (n x n)

    Returns:
        A (ndarray): n x n strictly diagonally dominant matrix
        b (ndarray): RHS vector
        x_true (ndarray): The true solution vector
    """

    k = [np.ones(n - 1), -2 * np.ones(n), np.ones(n - 1)]
    offset = [-1, 0, 1]
    A = diags(k, offset).toarray()

    # Solution is always all ones
    x_true = np.ones((n, 1))

    # Compute b = A @ x_true
    b = A @ x_true

    return A, b, x_true


def lu_factorisation(A):
    """
    Compute the LU factorisation of a square matrix A.

    The function decomposes a square matrix ``A`` into the product of a lower
    triangular matrix ``L`` and an upper triangular matrix ``U`` such that:

    .. math::
        A = L U

    where ``L`` has unit diagonal elements and ``U`` is upper triangular.

    Parameters
    ----------
    A : numpy.ndarray
        A 2D NumPy array of shape ``(n, n)`` representing the square matrix to
        factorise.

    Returns
    -------
    L : numpy.ndarray
        A lower triangular matrix with shape ``(n, n)`` and unit diagonal.
    U : numpy.ndarray
        An upper triangular matrix with shape ``(n, n)``.
    """
    n, m = A.shape
    if n != m:
        raise ValueError(f"Matrix A is not square {A.shape=}")

    U = A.astype(float)
    L = np.eye(n)
    for i in range(n - 1):
        for j in range(i + 1, n):
            L[j, i] = U[j, i] / U[i, i]
            U[j, :] = U[j, :] - L[j, i] * U[i, :]

    return L, U


def determinant(A):
    n = A.shape[0]
    L, U = lu_factorisation(A)

    det_L = 1.0
    det_U = 1.0

    for i in range(n):
        det_L *= L[i, i]
        det_U *= U[i, i]

    return det_L * det_U


def row_add(A, b, p, k, q):
    """
    Perform an in-place row addition operation on a linear system.

    This function applies the elementary row operation:

    ``row_p ← row_p + k * row_q``

    where `row_p` and `row_q` are rows in the coefficient matrix `A` and the
    right-hand side vector `b`. It updates the entries of `A` and `b`
    **in place**, directly modifying the original data.

    Parameters
    ----------
    A : numpy.ndarray
        A 2D NumPy array of shape ``(n, n)`` representing the coefficient matrix
        of the linear system.
    b : numpy.ndarray
        A 2D NumPy array of shape ``(n, 1)`` representing the right-hand side
        vector of the system.
    p : int
        The index of the row to be updated (destination row). Must satisfy
        ``0 <= p < n``.
    k : float
        The scalar multiplier applied to `row_q` before adding it to `row_p`.
    q : int
        The index of the source row to be scaled and added. Must satisfy
        ``0 <= q < n``.
    """
    n = A.shape[0]

    # Perform the row operation
    for j in range(n):
        A[p, j] = A[p, j] + k * A[q, j]

    # Update the corresponding value in b
    b[p, 0] = b[p, 0] + k * b[q, 0]


def gaussian_elimination(A, b):
    """
    Perform Gaussian elimination to reduce a linear system to upper triangular
    form.

    This function performs **forward elimination** to transform the coefficient
    matrix `A` into an upper triangular matrix, while applying the same
    operations to the right-hand side vector `b`. This is the first step in
    solving a linear system of equations of the form ``Ax = b`` using Gaussian
    elimination.

    Parameters
    ----------
    A : numpy.ndarray
        A 2D NumPy array of shape ``(n, n)`` representing the coefficient matrix
        of the system.
    b : numpy.ndarray
        A 2D NumPy array of shape ``(n, 1)`` representing the right-hand side
        vector.

    Returns
    -------
    None
        This function modifies `A` and `b` **in place** and does not return
        anything.
    """
    # find shape of system
    n = A.shape[0]

    # perform forward elimination
    for i in range(n - 1):
        # eliminate column i
        for j in range(i + 1, n):
            # row j
            factor = A[j, i] / A[i, i]
            row_add(A, b, j, -factor, i)


def runtime_comparison():
    """
    Time lu_factorisation against gaussian_elimination across a range of
    matrix sizes, produce a log-log plot comparing the two, and save it
    to "plot.png" (the autograder looks for this exact filename).
    """
    import time
    from matplotlib import pyplot as plt

    sizes = [2**j for j in range(1, 8)]
    lu_times = []
    ge_times = []

    for n in sizes:
        A, b, x = generate_safe_system(n)

        start = time.perf_counter()
        lu_factorisation(A)
        lu_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        gaussian_elimination(A, b)
        ge_times.append(time.perf_counter() - start)

    plt.loglog(sizes, lu_times, label="LU factorisation")
    plt.loglog(sizes, ge_times, label="Gaussian elimination")
    plt.xlabel("matrix size n")
    plt.ylabel("time (s)")
    plt.legend()
    plt.savefig("plot.png")


if __name__ == "__main__":
    A_large, b_large, x_large = generate_safe_system(100)
    print(determinant(A_large))
    runtime_comparison()
