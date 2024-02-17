from itertools import permutations

import numpy as np
from scipy.optimize import root
from scipy.special import digamma, polygamma

from DGLM.utils import AlphaAndBeta, MeanAndCov, dotdot, dotdotinv


class BernoulliLogisticDGLM:
    def __init__(self) -> None:
        pass

    def theta_predict_step(
        self, theta_filt: MeanAndCov, G: np.ndarray, W: np.ndarray
    ) -> MeanAndCov:
        m, C = theta_filt.mean, theta_filt.cov
        a = G @ m
        R = dotdot(G.T, C, G) + W
        return MeanAndCov(a, R)

    def lambda_predict_step(self, theta_pred: MeanAndCov, F: np.ndarray) -> MeanAndCov:
        a, R = theta_pred.mean, theta_pred.cov
        f = F.T @ a
        q = dotdot(F.T, R, F)
        return MeanAndCov(f, q)

    def calc_alpha_beta(self, lambda_pred: MeanAndCov) -> AlphaAndBeta:
        f, q = lambda_pred.mean[0, 0], lambda_pred.cov[0, 0]

        fun1 = lambda x: digamma(x[0]) - digamma(x[1])
        fun2 = lambda x: polygamma(1, x[0]) + polygamma(1, x[1])
        fun = lambda x: [fun1(x) - f, fun2(x) - q]

        errors = {}
        for alpha0, beta0 in permutations(np.exp(range(-5, 5)), 2):
            _f = fun1([alpha0, beta0])
            _q = fun2([alpha0, beta0])
            error = np.mean(np.abs([_f - f, _q - q]))
            errors[(alpha0, beta0)] = error

        x0 = min(errors, key=errors.get)
        res = root(fun, x0=x0)
        return AlphaAndBeta(res.x[0], res.x[1])

    def z_predict(self, alpha_beta: AlphaAndBeta) -> MeanAndCov:
        alpha, beta = alpha_beta.alpha, alpha_beta.beta
        z_mean = alpha / (alpha + beta)
        z_cov = alpha * beta / (alpha + beta) ** 2
        return MeanAndCov(z_mean, z_cov)

    def lambda_filter_step(self, alpha_beta: AlphaAndBeta, z_obs: float) -> MeanAndCov:
        alpha, beta = alpha_beta.alpha, alpha_beta.beta
        g = digamma(alpha + z_obs) - digamma(beta + 1 - z_obs)
        p = polygamma(1, alpha + z_obs) + polygamma(1, beta + 1 - z_obs)
        return MeanAndCov(g, p)

    def theta_filter_step(
        self,
        theta_pred: MeanAndCov,
        lambda_pred: MeanAndCov,
        lambda_filt: MeanAndCov,
        F: np.ndarray,
    ) -> MeanAndCov:
        a, R = theta_pred.mean, theta_pred.cov
        f, q = lambda_pred.mean, lambda_pred.cov
        g, p = lambda_filt.mean, lambda_filt.cov

        m = a + R @ F * (g - f) / q
        C = R - R @ F @ F.T @ R.T * (1 - p / q) / q
        return MeanAndCov(m, C)
