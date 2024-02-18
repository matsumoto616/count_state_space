# %%
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tqdm

from DGLM.data_maker import make_weekly_and_yearly_data
from DGLM.dcmm import BinaryPoissonDCMM
from DGLM.utils import MeanAndCov, ZeroAndPlus_MeanAndCov

# %%
start_date = "2020-01-01"
end_date = "2021-12-31"
df_data = make_weekly_and_yearly_data(start_date, end_date)

# %%
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_data.index, y=df_data["y_raw"]))

# %%
dcmm = BinaryPoissonDCMM()

#%%
ts = df_data.index
ys = df_data["y_raw"]
y_preds = []
theta_zero_filt = MeanAndCov(np.ones((1, 1)), np.eye(1))
theta_plus_filt = MeanAndCov(np.ones((1, 1)), np.eye(1))
theta_filt = ZeroAndPlus_MeanAndCov(theta_zero_filt, theta_plus_filt)
F_zero = np.eye(1)
F_plus = np.eye(1)
G_zero = np.eye(1)
G_plus = np.eye(1)
W_zero = np.eye(1) * 0.1
W_plus = np.eye(1) * 0.1
for t in tqdm.tqdm(range(len(ys))):
    theta_pred = dcmm.theta_predict_step(theta_filt, G_zero, W_zero, G_plus, W_plus)
    lambda_pred = dcmm.lambda_predict_step(theta_pred, F_zero, F_plus)
    alpha_beta = dcmm.calc_alpha_beta(lambda_pred)
    y_pred = dcmm.y_predict(alpha_beta)
    y_preds.append(y_pred)
    lambda_filt = dcmm.lambda_filter_step(alpha_beta, ys[t])
    theta_filt = dcmm.theta_filter_step(theta_pred, lambda_pred, lambda_filt, F_zero, F_plus)

# %%
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=ts,
        y=ys,
        name="data"
    )
)
fig.add_trace(
    go.Scatter(
        x=ts,
        y=[y_pred.plus.mean for y_pred in y_preds],
        name="mean"
    )
)
# fig.add_trace(
#     go.Scatter(
#         x=ts,
#         y=[np.sqrt(y_pred.cov) for y_pred in y_preds],
#         name="std"
#     )
# )
# %%
