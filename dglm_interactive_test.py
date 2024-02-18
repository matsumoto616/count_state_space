# %%
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tqdm

from data_maker import make_weekly_and_yearly_data, make_weekly_data
from DGLM.dglm import BernoulliLogisticDGLM, PoissonLoglinearDGLM
from DGLM.utils import (
    MeanAndCov,
    make_diag_stack_matrix,
    make_hstack_matrix,
    stack_matrix,
)

# %%
start_date = "2020-01-01"
end_date = "2021-12-31"
# df_data = make_weekly_and_yearly_data(start_date, end_date)
df_data = make_weekly_data(start_date, end_date)

# %%
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_data.index, y=df_data["y_raw"]))

# %%
dglm = BernoulliLogisticDGLM()

# %%
zs = df_data["z"]
z_preds = []
alpha_betas = []
theta_filt = MeanAndCov(np.ones((1, 1)), np.eye(1))
F = np.eye(1)
G = np.eye(1)
W = np.eye(1) * 0.1
for t in tqdm.tqdm(range(len(df_data))):
    theta_pred = dglm.theta_predict_step(theta_filt, G, W)
    lambda_pred = dglm.lambda_predict_step(theta_pred, F)
    alpha_beta = dglm.calc_alpha_beta(lambda_pred)
    alpha_betas.append(alpha_beta)
    z_pred = dglm.z_predict(alpha_beta)
    z_preds.append(z_pred)
    lambda_filt = dglm.lambda_filter_step(alpha_beta, zs[t])
    theta_filt = dglm.theta_filter_step(theta_pred, lambda_pred, lambda_filt, F)

h = 100
z_fores = []
for t in tqdm.tqdm(range(h)):
    theta_pred = dglm.theta_predict_step(theta_filt, G, W)
    lambda_pred = dglm.lambda_predict_step(theta_pred, F)
    alpha_beta = dglm.calc_alpha_beta(lambda_pred)
    z_fore = dglm.z_predict(alpha_beta)
    z_fores.append(z_fore)

# %%
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_data.index, y=df_data["z"], name="data"))
fig.add_trace(
    go.Scatter(x=df_data.index, y=[z_pred.mean for z_pred in z_preds], name="mean")
)
fig.add_trace(
    go.Scatter(
        x=df_data.index, y=[np.sqrt(z_pred.cov) for z_pred in z_preds], name="std"
    )
)

# %%
fig = go.Figure()
fig.add_trace(
    go.Scatter(x=df_data.index, y=[alpha_beta.alpha for alpha_beta in alpha_betas])
)
fig.add_trace(
    go.Scatter(x=df_data.index, y=[alpha_beta.beta for alpha_beta in alpha_betas])
)

# %%
dglm = PoissonLoglinearDGLM()

# %%
ts = df_data.index
ys = df_data["yt_weekly"]
y_preds = []
alpha_betas = []
theta_filt = MeanAndCov(np.ones((1, 1)), np.eye(1))
F = np.eye(1)
G = np.eye(1)
W = np.eye(1) * 0.1
for t in tqdm.tqdm(range(len(ys))):
    theta_pred = dglm.theta_predict_step(theta_filt, G, W)
    lambda_pred = dglm.lambda_predict_step(theta_pred, F)
    alpha_beta = dglm.calc_alpha_beta(lambda_pred)
    alpha_betas.append(alpha_beta)
    y_pred = dglm.y_predict(alpha_beta)
    y_preds.append(y_pred)
    lambda_filt = dglm.lambda_filter_step(alpha_beta, ys[t])
    theta_filt = dglm.theta_filter_step(theta_pred, lambda_pred, lambda_filt, F)

h = 100
y_fores = []
for t in tqdm.tqdm(range(h)):
    theta_pred = dglm.theta_predict_step(theta_filt, G, W)
    theta_filt = theta_pred
    lambda_pred = dglm.lambda_predict_step(theta_pred, F)
    alpha_beta = dglm.calc_alpha_beta(lambda_pred)
    y_fore = dglm.y_predict(alpha_beta)
    y_fores.append(y_fore)

# %%
fig = go.Figure()
fig.add_trace(go.Scatter(x=ts, y=ys, name="data"))
fig.add_trace(go.Scatter(x=ts, y=[y_pred.mean for y_pred in y_preds], name="mean"))
fig.add_trace(
    go.Scatter(x=ts, y=[np.sqrt(y_pred.cov) for y_pred in y_preds], name="std")
)

# %%
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
W0_base_trend = np.array([[0.001]])
W0_weekly_seasonal = np.array(
    [
        [0.001, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]
)
W0 = make_diag_stack_matrix([W0_base_trend, W0_weekly_seasonal])

# %%
dglm = PoissonLoglinearDGLM()

# %%
ts = df_data.index
ys = df_data["y_raw"]
y_preds = []
y_filts = []
theta_filts = []
alpha_beta_preds = []
alpha_beta_filts = []
lambda_preds = []
lambda_filts = []
F = F0
G = G0
W = W0
theta_filt = MeanAndCov(np.ones((G.shape[0], 1)), np.eye(G.shape[0]))
for t in tqdm.tqdm(range(len(ys))):
    theta_pred = dglm.theta_predict_step(theta_filt, G, W)
    lambda_pred = dglm.lambda_predict_step(theta_pred, F)
    lambda_preds.append(lambda_pred)
    alpha_beta_pred = dglm.calc_alpha_beta(lambda_pred)
    alpha_beta_preds.append(alpha_beta_pred)
    y_pred = dglm.y_predict(alpha_beta_pred)
    y_preds.append(y_pred)
    lambda_filt = dglm.lambda_filter_step(alpha_beta_pred, ys[t])
    lambda_filts.append(lambda_filt)
    alpha_beta_filt = dglm.calc_alpha_beta(lambda_filt)
    alpha_beta_filts.append(alpha_beta_filt)
    y_filt = dglm.y_predict(alpha_beta_filt)
    y_filts.append(y_filt)
    theta_filt = dglm.theta_filter_step(theta_pred, lambda_pred, lambda_filt, F)
    theta_filts.append(theta_filt)

# %%
fig = go.Figure()
fig.add_trace(go.Scatter(x=ts, y=ys, name="data"))
fig.add_trace(go.Scatter(x=ts, y=[y_filt.mean for y_filt in y_filts], name="mean_filt"))
fig.add_trace(go.Scatter(x=ts, y=[y_pred.mean for y_pred in y_preds], name="mean_pred"))
fig.add_trace(
    go.Scatter(x=ts, y=[np.sqrt(y_filt.cov) for y_filt in y_filts], name="std")
)
# %%
fig = go.Figure()
fig.add_trace(go.Scatter(x=ts, y=ys, name="data"))
fig.add_trace(
    go.Scatter(
        x=ts, y=[theta_filt.mean[0, 0] for theta_filt in theta_filts], name="mean"
    )
)
# %%
dglm = BernoulliLogisticDGLM()

# %%
ts = df_data.index
zs = df_data["yt_weekly"].apply(lambda x: int(bool(x)))
z_preds = []
z_filts = []
theta_filts = []
F = F0
G = G0
W = W0
theta_filt = MeanAndCov(np.ones((G.shape[0], 1)), np.eye(G.shape[0]))
for t in tqdm.tqdm(range(len(zs))):
    theta_pred = dglm.theta_predict_step(theta_filt, G, W)
    lambda_pred = dglm.lambda_predict_step(theta_pred, F)
    alpha_beta_pred = dglm.calc_alpha_beta(lambda_pred)
    z_pred = dglm.z_predict(alpha_beta_pred)
    z_preds.append(z_pred)
    lambda_filt = dglm.lambda_filter_step(alpha_beta_pred, zs[t])
    alpha_beta_filt = dglm.calc_alpha_beta(lambda_filt)
    z_filt = dglm.z_predict(alpha_beta_filt)
    z_filts.append(z_filt)
    theta_filt = dglm.theta_filter_step(theta_pred, lambda_pred, lambda_filt, F)
    theta_filts.append(theta_filt)

# %%
fig = go.Figure()
fig.add_trace(go.Scatter(x=ts, y=zs, name="data"))
fig.add_trace(go.Scatter(x=ts, y=[z_filt.mean for z_filt in z_filts], name="mean_filt"))
fig.add_trace(go.Scatter(x=ts, y=[z_pred.mean for z_pred in z_preds], name="mean_pred"))
fig.add_trace(
    go.Scatter(x=ts, y=[np.sqrt(z_filt.cov) for z_filt in z_filts], name="std")
)
# %%
fig = go.Figure()
fig.add_trace(go.Scatter(x=ts, y=zs, name="data"))
fig.add_trace(
    go.Scatter(
        x=ts, y=[theta_filt.mean[1, 0] for theta_filt in theta_filts], name="mean"
    )
)
# %%
