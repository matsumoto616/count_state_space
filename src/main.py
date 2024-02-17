#%%
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from data_maker import make_weekly_and_yearly_data

#%%
start_date = "2020-01-01"
end_date = "2021-12-31"
df_data = make_weekly_and_yearly_data(start_date, end_date)

# %%
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=df_data.index, y=df_data["y_raw"]
    )
)
# %%
from dglm import BernoulliLogisticDGLM
from utils import MeanAndCov

# %%
d = 2
x0 = MeanAndCov(np.ones(d), np.eye(d))

# %%
dglm = BernoulliLogisticDGLM()

# %%
F = np.eye(d)
Q = np.eye(d) * 0.1
dglm.predict_step(x0, F, Q)
# %%
