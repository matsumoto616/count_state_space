from itertools import permutations

import numpy as np
from scipy.optimize import root, root_scalar
from scipy.special import digamma, polygamma
from scipy.stats import nbinom

from DGLM.utils import AlphaAndBeta, MeanAndCov, dotdot, dotdotinv


class BernoulliLogisticDGLM:
    def __init__(self) -> None:
        pass

    def theta_predict_step(
        self, theta_filt: MeanAndCov, G: np.ndarray, W: np.ndarray
    ) -> MeanAndCov:
        m, C = theta_filt.mean, theta_filt.cov
        a = G @ m
        R = dotdot(G, C, G.T) + W
        return MeanAndCov(a, R)

    def lambda_predict_step(self, theta_pred: MeanAndCov, F: np.ndarray) -> MeanAndCov:
        a, R = theta_pred.mean, theta_pred.cov
        f = F.T @ a
        q = dotdot(F.T, R, F)
        return MeanAndCov(f, q)

    def calc_alpha_beta(self, lambda_pred: MeanAndCov) -> AlphaAndBeta:
        f, q = lambda_pred.mean[0, 0], lambda_pred.cov[0, 0]

        fun1 = lambda x: digamma(np.exp(x[0])) - digamma(np.exp(x[1]))
        fun2 = lambda x: polygamma(1, np.exp(x[0])) + polygamma(1, np.exp(x[1]))
        fun = lambda x: [fun1(x) - f, fun2(x) - q]

        jac11 = lambda x: polygamma(1, np.exp(x[0])) * np.exp(x[0])
        jac12 = lambda x: -1 * polygamma(1, np.exp(x[1])) * np.exp(x[1])
        jac21 = lambda x: polygamma(2, np.exp(x[0])) * np.exp(x[0])
        jac22 = lambda x: polygamma(2, np.exp(x[1])) * np.exp(x[1])
        jac = lambda x: [[jac11(x), jac12(x)], [jac21(x), jac22(x)]]

        errors = {}
        for log_alpha0, log_beta0 in permutations(range(-5, 5), 2):
            _f = fun1([log_alpha0, log_beta0])
            _q = fun2([log_alpha0, log_beta0])
            error = np.mean(np.abs([_f - f, _q - q]))
            errors[(log_alpha0, log_beta0)] = error

        x0 = min(errors, key=errors.get)
        res = root(fun, x0=x0, jac=jac)
        return AlphaAndBeta(np.exp(res.x[0]), np.exp(res.x[1]))
    
    def z_predict(self, alpha_beta: AlphaAndBeta) -> MeanAndCov:
        alpha, beta = alpha_beta.alpha, alpha_beta.beta
        z_mean = alpha / (alpha + beta)
        z_cov = alpha * beta / (alpha + beta) ** 2
        return MeanAndCov(z_mean, z_cov)

    def lambda_filter_step(self, alpha_beta: AlphaAndBeta, z_obs: int) -> MeanAndCov:
        alpha, beta = alpha_beta.alpha, alpha_beta.beta
        g = digamma(alpha + z_obs) - digamma(beta + 1 - z_obs)
        p = polygamma(1, alpha + z_obs) + polygamma(1, beta + 1 - z_obs)
        return MeanAndCov(np.array([[g]]), np.array([[p]]))

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


class PoissonLoglinearDGLM:
    def __init__(self) -> None:
        pass

    def theta_predict_step(
        self, theta_filt: MeanAndCov, G: np.ndarray, W: np.ndarray
    ) -> MeanAndCov:
        m, C = theta_filt.mean, theta_filt.cov
        a = G @ m
        R = dotdot(G, C, G.T) + W
        return MeanAndCov(a, R)

    def lambda_predict_step(self, theta_pred: MeanAndCov, F: np.ndarray) -> MeanAndCov:
        a, R = theta_pred.mean, theta_pred.cov
        f = F.T @ a
        q = dotdot(F.T, R, F)
        return MeanAndCov(f, q)

    def calc_alpha_beta(self, lambda_pred: MeanAndCov) -> AlphaAndBeta:
        f, q = lambda_pred.mean[0, 0], lambda_pred.cov[0, 0]

        fun1 = lambda x: digamma(np.exp(x[0])) - np.log(np.exp(x[1])) - f
        fun2 = lambda x: polygamma(1, np.exp(x)) - q
        f1prime = lambda x: -1
        f1prime2 = lambda x: 0
        f2prime1 = lambda x: polygamma(2, np.exp(x)) * np.exp(x)
        f2prime2 = lambda x: polygamma(3, np.exp(x)) * np.exp(2*x) + polygamma(2, np.exp(x)) * np.exp(x)       

        errors = {}
        for log_alpha0 in range(-10, 10):
            error = np.abs(fun2(log_alpha0))
            errors[log_alpha0] = error
        log_alpha0 = min(errors, key=errors.get)
        res = root_scalar(fun2, x0=log_alpha0, fprime=f2prime1, fprime2=f2prime2)
        log_alpha = res.root

        errors = {}
        for log_beta0 in range(-10, 10):
            error = np.abs(fun1([log_alpha, log_beta0]))
            errors[log_beta0] = error
        log_beta0 = min(errors, key=errors.get)
        res = root_scalar(lambda x: fun1([log_alpha, x]), x0=log_beta0, fprime=f1prime, fprime2=f1prime2)
        log_beta = res.root

        return AlphaAndBeta(np.exp(log_alpha), np.exp(log_beta))

    def y_predict(self, alpha_beta: AlphaAndBeta) -> MeanAndCov:
        alpha, beta = alpha_beta.alpha, alpha_beta.beta
        y_mean, y_cov = nbinom.stats(alpha, beta/(1+beta))
        return MeanAndCov(y_mean, y_cov)

    def lambda_filter_step(self, alpha_beta: AlphaAndBeta, y_obs: int) -> MeanAndCov:
        alpha, beta = alpha_beta.alpha, alpha_beta.beta
        g = digamma(alpha + y_obs) - np.log(beta + 1)
        p = polygamma(1, alpha + y_obs)
        return MeanAndCov(np.array([[g]]), np.array([[p]]))

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

