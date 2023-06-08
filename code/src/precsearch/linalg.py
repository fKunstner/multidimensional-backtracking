from dataclasses import dataclass

import numpy as np


def is_positive(p: np.ndarray):
    return np.all(p > 0)


def orth_2d(v):
    return np.array([-v[1], v[0]])


def find_intersection_lines(x, dx, y, dy):
    """Returns the intersection of two lines defined by x + a dx, y + b dy

    Looking for
        x + a dx = y + b dy
    or equivalently
        a dx - b dy = y - x
    In matrix form
        [dx0, -dy0] [a] = [y0-x0]
        [dx1, -dy1] [b] = [y1-x1]
        [...,  ...]     = [ ... ]
        [dxd, -dyd]     = [yd-xd]
    """
    ab = np.linalg.solve(np.concatenate([dx, -dy], axis=2), y - x)
    a, b = ab[0], ab[1]
    return x + a * dx, y + b * dy


def find_intersection():
    p1, p2 = find_intersection_with_axes(P, orthogonal(dp))
    a1, a2 = find_intersection_with_axes(A, orthogonal(da))

    if p1[1] > a1[1] and p2[0] > a2[0]:
        return np.array([0.5 * a2[0], 0.5 * a1[1]])

    M = np.array(
        [
            [dp[0], dp[1]],
            [da[0], da[1]],
        ]
    )
    target = np.array([np.inner(dp, P), np.inner(da, A)])
    newP = np.linalg.solve(M, target)
    return newP


@dataclass
class Simplex:
    """

    The base of the simplex is the point b such that the set can be written as
        {x : <x - b, g(b)> <= 0}

    Where f is the normalized log-determinant,
        f(x) = (1/d) * sum(log(x))
    and g is its gradient
    """

    base: np.ndarray

    @staticmethod
    def validate(p: np.ndarray):
        if not is_positive(p):
            raise ValueError("Using a point p that has negative entries. ", p)

    @staticmethod
    def vertices():
        pass

    @staticmethod
    def from_base(base: np.ndarray):
        """Creates a simplex from its base."""
        return Simplex(base)

    def intersection(self, other: "Simplex"):
        pass
