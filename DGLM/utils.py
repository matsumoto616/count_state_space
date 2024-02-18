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


def make_diag_stack_matrix(matrix_list):
    """
    行列のリストから対角方向に結合した行列を作成する
    """
    dim_i = sum([m.shape[0] for m in matrix_list])
    dim_j = sum([m.shape[1] for m in matrix_list])
    block_diag = np.zeros((dim_i, dim_j))

    pos_i = pos_j = 0
    for m in matrix_list:
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                block_diag[pos_i + i, pos_j + j] = m[i, j]
        pos_i += m.shape[0]
        pos_j += m.shape[1]

    return block_diag


def make_hstack_matrix(matrix_list):
    """
    行列のリストから横方向に結合した行列を作成する
    """
    return np.concatenate(matrix_list, 1)


def stack_matrix(M0, N):
    """
    ndarray Mを0軸にN個重ねた ndarrayを作成する
    """
    M = np.zeros((N, M0.shape[0], M0.shape[1]))
    for n in range(N):
        M[n] = M0

    return M
