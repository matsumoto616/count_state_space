import numpy as np
from utils import MeanAndCov, dotdot, dotdotinv

class BernoulliLogisticDGLM():
    def __init__(self) -> None:
        pass
        
    def predict_step(self, filt: MeanAndCov, F: np.ndarray, Q:np.ndarray) -> MeanAndCov:
        mu_pred = F @ filt.mean
        V_pred = dotdot(F.T, Q, F)
        return MeanAndCov(mu_pred, V_pred)