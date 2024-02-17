import pytest
import numpy as np
from scipy.special import digamma, polygamma
from itertools import permutations
from DGLM.dglm import BernoulliLogisticDGLM
from DGLM.utils import MeanAndCov, AlphaAndBeta

class TestBernoulliLogisticDGLM:
    def test_can_run_theta_predict_step(self):
        d = 2
        x0 = MeanAndCov(np.ones(d), np.eye(d))
        G = np.eye(d)
        W = np.eye(d)

        dglm = BernoulliLogisticDGLM()
        theta_pred = dglm.theta_predict_step(x0, G, W)

        assert np.all(theta_pred.mean == np.ones(d))
        assert np.all(theta_pred.cov == np.eye(d)*2)

    def test_can_run_lambda_predict_step(self):
        d = 2
        theta_pred = MeanAndCov(np.ones(d), np.eye(d))
        F = np.eye(d)

        dglm = BernoulliLogisticDGLM()
        lambda_pred = dglm.lambda_predict_step(theta_pred, F)

        assert np.all(lambda_pred.mean == np.ones(d))
        assert np.all(lambda_pred.cov == np.eye(d))

    def test_calc_alpha_beta(self):
        trues = np.linspace(0.1, 10, 20)

        for alpha_true, beta_true, in permutations(trues, 2):
            f_true = digamma(alpha_true) - digamma(beta_true)
            q_true = polygamma(1, alpha_true) + polygamma(1, beta_true)

            y_pred = MeanAndCov(
                np.array([[f_true]]),
                np.array([[q_true]])
            )
            dglm = BernoulliLogisticDGLM()
            alpha_beta = dglm.calc_alpha_beta(y_pred)
            alpha, beta = alpha_beta.alpha, alpha_beta.beta
            f = digamma(alpha) - digamma(beta)
            q = polygamma(1, alpha) + polygamma(1, beta)

            assert np.isclose(alpha_true, alpha_beta.alpha, atol=1e-3)
            assert np.isclose(beta_true, alpha_beta.beta, atol=1e-3)
            # assert np.isclose(f_true, f, atol=1e-3)
            # assert np.isclose(q_true, q, atol=1e-3)

    def test_can_run_z_predict(self):
        alpha_beta = AlphaAndBeta(1, 1)        
        dglm = BernoulliLogisticDGLM()
        z_pred = dglm.z_predict(alpha_beta)

        assert z_pred.mean == 0.5
        assert z_pred.cov == 0.25

    def test_can_run_lambda_filter_step(self):
        alpha_beta = AlphaAndBeta(1, 1)
        for z in [0, 1]:   
            dglm = BernoulliLogisticDGLM()
            lambda_filt = dglm.lambda_filter_step(alpha_beta, z)

    def test_can_run_theta_filter_step(self):
        pass

    def test_theta_update(self):
        dglm = BernoulliLogisticDGLM()

        d = 2
        theta0 = MeanAndCov(np.ones((d, 1)), np.eye(d))
        F = np.ones((2, 1))
        G = np.eye(d)
        W = np.eye(d)
        for z in [0, 1]:
            theta_pred = dglm.theta_predict_step(theta0, G, W)
            lambda_pred = dglm.lambda_predict_step(theta_pred, F)
            alpha_beta = dglm.calc_alpha_beta(lambda_pred)
            lambda_filt = dglm.lambda_filter_step(alpha_beta, z)
            theta_filt = dglm.theta_filter_step(theta_pred, lambda_pred, lambda_filt, F)