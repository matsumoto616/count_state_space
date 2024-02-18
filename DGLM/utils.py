from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve


@dataclass
class MeanAndCov:
    mean: np.ndarray
    cov: np.ndarray


@dataclass
class AlphaAndBeta:
    alpha: float
    beta: float


@dataclass
class ZeroAndPlus_MeanAndCov:
    zero: MeanAndCov | None
    plus: MeanAndCov | None


@dataclass
class ZeroAndPlus_AlphaAndBeta:
    zero: AlphaAndBeta
    plus: AlphaAndBeta


def dotdot(a, b, c):
    return np.dot(np.dot(a, b), c)


def dotdotinv(a, b, c):
    """a * b * c^{-1}, where c is symmetric positive"""
    return solve(c, np.dot(a, b).T, assume_a="pos", overwrite_b=True).T
