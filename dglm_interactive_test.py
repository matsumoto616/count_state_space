# %%
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tqdm

from DGLM.data_maker import make_weekly_and_yearly_data
from DGLM.dglm import BernoulliLogisticDGLM, PoissonLoglinearDGLM
from DGLM.utils import MeanAndCov

# %%
start_date = "2020-01-01"
end_date = "2021-12-31"
df_data = make_weekly_and_yearly_data(start_date, end_date)

# %%
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_data.index, y=df_data["y_raw"]))

# %%
dglm = BernoulliLogisticDGLM()

#%%
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
fig.add_trace(
    go.Scatter(
        x=df_data.index,
        y=df_data["z"],
        name="data"
    )
)
fig.add_trace(
    go.Scatter(
        x=df_data.index,
        y=[z_pred.mean for z_pred in z_preds],
        name="mean"
    )
)
fig.add_trace(
    go.Scatter(
        x=df_data.index,
        y=[np.sqrt(z_pred.cov) for z_pred in z_preds],
        name="std"
    )
)

# %%
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=df_data.index,
        y=[alpha_beta.alpha for alpha_beta in alpha_betas]
    )
)
fig.add_trace(
    go.Scatter(
        x=df_data.index,
        y=[alpha_beta.beta for alpha_beta in alpha_betas]
    )
)

#%%
dglm = PoissonLoglinearDGLM()

#%%
ts = df_data.index
ys = df_data["y_raw"]
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
        y=[y_pred.mean for y_pred in y_preds],
        name="mean"
    )
)
fig.add_trace(
    go.Scatter(
        x=ts,
        y=[np.sqrt(y_pred.cov) for y_pred in y_preds],
        name="std"
    )
)
# %%
