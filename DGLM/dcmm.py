import numpy as np
from DGLM.dglm import PoissonLoglinearDGLM, BernoulliLogisticDGLM
from DGLM.utils import ZeroAndPlus_MeanAndCov, ZeroAndPlus_AlphaAndBeta

class BinaryPoissonDCMM:
    def __init__(self) -> None:
        self.bl_dglm = BernoulliLogisticDGLM()
        self.pl_dglm = PoissonLoglinearDGLM()

    def theta_predict_step(
            self,
            theta_filt: ZeroAndPlus_MeanAndCov,
            G_zero: np.ndarray,
            W_zero: np.ndarray,
            G_plus: np.ndarray,
            W_plus: np.ndarray,
        ) -> ZeroAndPlus_MeanAndCov:
        theta_zero_filt, theta_plus_filt = theta_filt.zero, theta_filt.plus
        theta_zero_pred = self.bl_dglm.theta_predict_step(theta_zero_filt, G_zero, W_zero)
        theta_plus_pred = self.pl_dglm.theta_predict_step(theta_plus_filt, G_plus, W_plus)

        return ZeroAndPlus_MeanAndCov(theta_zero_pred, theta_plus_pred)

    def lambda_predict_step(
            self, 
            theta_pred: ZeroAndPlus_MeanAndCov, 
            F_zero: np.ndarray,
            F_plus: np.ndarray
        ) -> ZeroAndPlus_MeanAndCov:
        theta_zero_pred, theta_plus_pred = theta_pred.zero, theta_pred.plus
        lambda_zero_pred = self.bl_dglm.lambda_predict_step(theta_zero_pred, F_zero)
        lambda_plus_pred = self.pl_dglm.lambda_predict_step(theta_plus_pred, F_plus)
        
        return ZeroAndPlus_MeanAndCov(lambda_zero_pred, lambda_plus_pred)

    def calc_alpha_beta(self, lambda_pred: ZeroAndPlus_MeanAndCov) -> ZeroAndPlus_AlphaAndBeta:
        lambda_zero_pred, lambda_plus_pred = lambda_pred.zero, lambda_pred.plus
        alpha_beta_zero = self.bl_dglm.calc_alpha_beta(lambda_zero_pred)
        alpha_beta_plus = self.pl_dglm.calc_alpha_beta(lambda_plus_pred)

        return ZeroAndPlus_AlphaAndBeta(alpha_beta_zero, alpha_beta_plus)
    
    def y_predict(self, alpha_beta: ZeroAndPlus_AlphaAndBeta) -> ZeroAndPlus_MeanAndCov:
        alpha_beta_zero, alpha_beta_plus = alpha_beta.zero, alpha_beta.plus
        z_predict = self.bl_dglm.z_predict(alpha_beta_zero)
        x_predict = self.pl_dglm.y_predict(alpha_beta_plus)

        return ZeroAndPlus_MeanAndCov(z_predict, x_predict)

    def lambda_filter_step(self, alpha_beta: ZeroAndPlus_AlphaAndBeta, y_obs: int) -> ZeroAndPlus_MeanAndCov:
        alpha_beta_zero, alpha_beta_plus = alpha_beta.zero, alpha_beta.plus
        z_obs = int(bool(y_obs))
        lambda_filt_zero = self.bl_dglm.lambda_filter_step(alpha_beta_zero, z_obs)
        if z_obs:
            x_obs = y_obs - z_obs
            lambda_filt_plus = self.pl_dglm.lambda_filter_step(alpha_beta_plus, x_obs)
            return ZeroAndPlus_MeanAndCov(lambda_filt_zero, lambda_filt_plus)
        else:
            return ZeroAndPlus_MeanAndCov(lambda_filt_zero, None)

    def theta_filter_step(
        self,
        theta_pred: ZeroAndPlus_MeanAndCov,
        lambda_pred: ZeroAndPlus_MeanAndCov,
        lambda_filt: ZeroAndPlus_MeanAndCov,
        F_zero: np.ndarray,
        F_plus: np.ndarray,
    ) -> ZeroAndPlus_MeanAndCov:
        theta_zero_pred, theta_plus_pred = theta_pred.zero, theta_pred.plus
        lambda_zero_pred, lambda_plus_pred = lambda_pred.zero, lambda_pred.plus
        lambda_zero_filt, lambda_plus_filt = lambda_filt.zero, lambda_filt.plus
        theta_zero_filt = self.bl_dglm.theta_filter_step(theta_zero_pred, lambda_zero_pred, lambda_zero_filt, F_zero)
        if lambda_plus_filt is not None:
            theta_plus_filt = self.pl_dglm.theta_filter_step(theta_plus_pred, lambda_plus_pred, lambda_plus_filt, F_plus)
            return ZeroAndPlus_MeanAndCov(theta_zero_filt, theta_plus_filt)
        else:
            return ZeroAndPlus_MeanAndCov(theta_zero_filt, theta_plus_pred)