#!/usr/bin/python
from math import sqrt

import numpy as np

from cagd.vec import vec2


# solves the system of linear equations Ax = res
# where A is a tridiagonal matrix with diag2 representing the main diagonal
# diag1 and diag3 represent the lower and upper diagonal respectively
# all four parameters are vectors of size n
# the first element of diag1 and the last element of diag3 are ignored
# therefore diag1[i], diag2[i] and diag3[i] are located on the same row of A
def solve_tridiagonal_equation(a, b, c, d):
    N = len(a)
    cp = np.zeros(N, dtype='float64')  # store tranformed c or c'
    dp = np.zeros(N, dtype='float64')  # store transformed d or d'
    X = np.zeros(N, dtype='float64')  # store unknown coefficients

    # Perform Forward Sweep
    # Equation 1 indexed as 0 in python
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    # Equation 2, ..., N (indexed 1 - N-1 in Python)
    for i in np.arange(1, (N), 1):
        dnum = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / dnum
        dp[i] = (d[i] - a[i] * dp[i - 1]) / dnum

    # Perform Back Substitution
    X[(N - 1)] = dp[N - 1]  # Obtain last xn

    for i in np.arange((N - 2), -1, -1):  # use x[i+1] to obtain x[i]
        X[i] = (dp[i]) - (cp[i]) * (X[i + 1])

    return X


# solves the system of linear equations Ax = res
# where A is an almost tridiagonal matrix with diag2 representing the main diagonal
# diag1 and diag3 represent the lower and upper diagonal respectively
# all four parameters are vectors of size n
# the first element of diag1 and the last element of diag3 represent the top right and bottom left elements of A
# diag1[i], diag2[i] and diag3[i] are located on the same row of A
def solve_almost_tridiagonal_equation_old(diag1, diag2, diag3, res):
    assert (len(diag1) == len(diag2) == len(diag3) == len(res))

    n = len(res)

    v = [0.0 for _ in range(n)]
    y = [0.0 for _ in range(n)]
    s = [0.0 for _ in range(n)]
    z = [0.0 for _ in range(n)]

    v[0] = 0
    y[0] = 0
    s[0] = 1

    for k in range(0, n - 1):
        z[k] = 1 / (diag2[k] + diag1[k] * v[k])
        v[k + 1] = -z[k] * diag3[k]

    for k in range(1, n):
        y[k] = z[k - 1] * (res[k - 1] - diag1[k - 1] * y[k - 1])
        s[k] = -diag1[k - 1] * s[k - 1] * z[k - 1]

    t = [0.0 for _ in range(n)]
    w = [0.0 for _ in range(n)]

    t[n - 1] = 1
    w[n - 1] = 0

    for k in range(n - 2, -1, -1):
        t[k] = v[k] * t[k + 1] + s[k]
        w[k] = v[k] * w[k + 1] + y[k]

    x = [0.0 for _ in range(n)]
    x[n - 1] = (res[n - 1] - diag3[n - 1] * w[0] - diag1[n - 1] * w[n - 2]) / (
            diag3[n - 1] * t[0] + diag1[n - 1] * t[n - 2] + diag2[n - 1])

    for k in range(n - 2, -1, -1):
        x[k] = t[k] * x[n - 1] + w[k]

    return x


# solves the system of linear equations Ax = res
# where A is an almost tridiagonal matrix with diag2 representing the main diagonal
# diag1 and diag3 represent the lower and upper diagonal respectively
# all four parameters are vectors of size n
# the first element of diag1 and the last element of diag3 represent the top right and bottom left elements of A
# diag1[i], diag2[i] and diag3[i] are located on the same row of A
def solve_almost_tridiagonal_equation(a, b, c, d):
    n = len(d)
    v = [0 for _ in range(n)]
    y = [0 for _ in range(n)]
    s = [0 for _ in range(n)]
    z = [0 for _ in range(n)]
    s = [0 for _ in range(n)]

    v[0] = 0
    y[0] = 0
    s[0] = 1

    for k in range(1, n):
        z[k] = (1 / (b[k - 1] + a[k - 1] * v[k - 1]))
        v[k] = ((-1 * z[k]) * c[k - 1])
        y[k] = z[k] * (d[k - 1] - (a[k - 1] * y[k - 1]))
        s[k] = ((-1 * a[k - 1]) * s[k - 1] * z[k])

    t = [0 for _ in range(n + 1)]
    w = [0 for _ in range(n + 1)]
    t[n] = 1
    w[n] = 0

    for k in range(n - 1, 0, -1):
        t[k] = (v[k] * t[k + 1]) + s[k]
        w[k] = (v[k] * w[k + 1]) + y[k]

    x = [0 for _ in range(n)]
    x[n - 1] = (d[n - 1] - c[n - 1] * w[1] - a[n - 1] * w[n - 1]) / (
            c[n - 1] * t[1] + a[n - 1] * t[n - 1] + b[n - 1])

    for k in range(n - 1, 0, -1):
        x[k - 1] = (t[k] * x[n - 1]) + w[k]

    return x

if __name__ == "__main__":
    a = [-16, 11, 12, 13, 14]
    b = [1, 2, 3, 4, 5]
    c = [7, 8, 9, 10, 15]
    d = [16, 17, 18, 19, 20]

    print(solve_almost_tridiagonal_equation(a, b, c, d))
