"""
Submit this file, plus a separate .md or .txt file with your written
explanation,Fill in the TODOs below - do not rename either
function, the autograder looks up `generate_table` by name.

What you need to do:
  1. Implement `gram_schmidt_qr`: the classical Gram-Schmidt QR
     factorisation, as covered in the lectures.
  2. Implement `generate_table`: build the eps/error1/error2/error3 table
     for eps = 10**-6, ..., 10**-16 using `gram_schmidt_qr`, and save it to
     "table.csv" in the working directory - the autograder looks for that
     exact filename, with a header row and columns named exactly
     eps, error1, error2, error3.
"""

import numpy as np
import pandas as pd


def A_func(eps):
    return np.array([[1, 1 + eps], [1 + eps, 1]], dtype=float)


def gram_schmidt_qr(A):
    """
    Compute the QR factorisation of a square matrix using the classical
    Gram-Schmidt process.

    Parameters
    ----------
    A : numpy.ndarray
        A square 2D NumPy array of shape ``(n, n)`` representing the input
        matrix.

    Returns
    -------
    Q : numpy.ndarray
        Orthonormal matrix of shape ``(n, n)`` where the columns form an
        orthonormal basis for the column space of A.
    R : numpy.ndarray
        Upper triangular matrix of shape ``(n, n)``.
    """
    n, m = A.shape
    if n != m:
        raise ValueError(f"the matrix A is not square, {A.shape=}")

    Q = np.empty_like(A)
    R = np.zeros_like(A)

    for j in range(n):
        v = A[:, j].copy()
        for i in range(j):
            # Classical GS: project the ORIGINAL column a_j onto q_i
            # (modified GS would use the partially updated v instead).
            R[i, j] = Q[:, i] @ A[:, j]
            v -= R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        if R[j, j] == 0:
            raise ValueError("A is singular to working precision")
        Q[:, j] = v / R[j, j]

    return Q, R


def _exact_Q_2x2(A):
    """Stable reference Q for a 2x2 matrix (see module docstring)."""
    a1 = A[:, 0]
    q1 = a1 / np.linalg.norm(a1)
    perp = np.array([-q1[1], q1[0]])  # det([q1, perp]) = +1
    # det(Q) must equal sign(det A) because det R = r11 * r22 > 0.
    q2 = perp if np.linalg.det(A) > 0 else -perp
    return np.column_stack([q1, q2])


def generate_table():
    """
    Build the eps/error1/error2/error3 table for eps = 10**-6, ..., 10**-16
    using gram_schmidt_qr, and save it to "table.csv" (the autograder looks
    for this exact filename).
    """
    rows = []
    for eps in [10.0**-j for j in range(6, 17)]:
        A = A_func(eps)
        Q, R = gram_schmidt_qr(A)
        I = np.eye(A.shape[0])

        # error1: backward error, error2: loss of orthogonality,
        # error3: forward error in Q (vs a stable 2x2 reference)
        error1 = np.linalg.norm(A - Q @ R, 2)
        error2 = np.linalg.norm(Q.T @ Q - I, 2)
        error3 = np.linalg.norm(Q - _exact_Q_2x2(A), 2)

        rows.append(
            {"eps": eps, "error1": error1, "error2": error2, "error3": error3}
        )

    df = pd.DataFrame(rows)
    df.to_csv("table.csv", index=False)
    return df


if __name__ == "__main__":
    generate_table()
