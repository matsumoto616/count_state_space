# %%
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tqdm

from data_maker import make_weekly_and_yearly_data, make_weekly_data
from DGLM.dcmm import BinaryPoissonDCMM
from DGLM.utils import (
    MeanAndCov,
    ZeroAndPlus_MeanAndCov,
    ZeroAndPlus_AlphaAndBeta,
    make_diag_stack_matrix,
    make_hstack_matrix,
    stack_matrix,
)

# %%
start_date = "2020-01-01"
end_date = "2021-12-31"
df_data = make_weekly_and_yearly_data(start_date, end_date)
# df_data = make_weekly_data(start_date, end_date)

# %%
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_data.index, y=df_data["y_raw"]))

# %%
dcmm = BinaryPoissonDCMM()

# %%
ts = df_data.index
ys = df_data["y_raw"]
y_preds = []
y_filts = []

# 状態方程式の行列
G0_base_trend = np.array([[1]])
G0_weekly_seasonal = np.array(
    [
        [-1, -1, -1, -1, -1, -1],
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
    ]
)
G0 = make_diag_stack_matrix([G0_base_trend, G0_weekly_seasonal])

# 観測方程式の行列
F0_base_trend = np.array([[1]])
F0_weekly_seasonal = np.array([[1, 0, 0, 0, 0, 0]])
F0 = make_hstack_matrix([F0_base_trend, F0_weekly_seasonal]).T  # 論文の定義

# ノイズの行列
W0_base_trend = np.array([[0.1]])
W0_weekly_seasonal = np.array(
    [
        [0.1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]
)
W0 = make_diag_stack_matrix([W0_base_trend, W0_weekly_seasonal])

F_zero = F0
F_plus = F0
G_zero = G0
G_plus = G0
W_zero = W0
W_plus = W0 * 0.01

# %%
theta_zero_filt = MeanAndCov(np.ones((G_zero.shape[0], 1)), np.eye(G_zero.shape[0]))
theta_plus_filt = MeanAndCov(np.ones((G_plus.shape[0], 1)), np.eye(G_plus.shape[0]))
theta_filt = ZeroAndPlus_MeanAndCov(theta_zero_filt, theta_plus_filt)

for t in tqdm.tqdm(range(len(ys))):
    theta_pred = dcmm.theta_predict_step(theta_filt, G_zero, W_zero, G_plus, W_plus)
    lambda_pred = dcmm.lambda_predict_step(theta_pred, F_zero, F_plus)
    alpha_beta_pred = dcmm.calc_alpha_beta(lambda_pred)
    y_pred = dcmm.y_predict(alpha_beta_pred)
    y_preds.append(y_pred)
    lambda_filt = dcmm.lambda_filter_step(alpha_beta_pred, ys[t])
    if lambda_filt.plus is None:
        lambda_filt_rev = ZeroAndPlus_MeanAndCov(lambda_filt.zero, lambda_pred.plus)
    else:
        lambda_filt_rev = lambda_filt
    alpha_beta_filt = dcmm.calc_alpha_beta(lambda_filt_rev)
    y_filt = dcmm.y_predict(alpha_beta_filt)
    y_filts.append(y_filt)
    theta_filt = dcmm.theta_filter_step(
        theta_pred, lambda_pred, lambda_filt, F_zero, F_plus
    )

# %%
fig = go.Figure()
fig.add_trace(go.Scatter(x=ts, y=ys, name="data"))
fig.add_trace(
    go.Scatter(
        x=ts,
        y=[y_pred.plus.mean * y_pred.zero.mean for y_pred in y_preds],
        name="mean_pred",
    )
)
fig.add_trace(
    go.Scatter(
        x=ts,
        y=[y_filt.plus.mean * y_filt.zero.mean for y_filt in y_filts],
        name="mean_filt",
    )
)
fig.add_trace(
    go.Scatter(x=ts, y=[np.sqrt(y_filt.plus.cov) for y_filt in y_filts], name="std")
)

# fig.update_layout(
#     yaxis_range=[0, max(ys)]
# )
# %%
