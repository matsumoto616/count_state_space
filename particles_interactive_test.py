#%% 
import particles
import numpy as np
import plotly.graph_objects as go
from particles import state_space_models as ssm
from particles import distributions as dists
from particles.collectors import Moments, Collector
from data_maker import make_weekly_and_yearly_data, make_weekly_data

# %%
start_date = "2020-01-01"
end_date = "2021-12-31"
# df_data = make_weekly_and_yearly_data(start_date, end_date)
df_data = make_weekly_data(start_date, end_date)

# %%
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_data.index, y=df_data["y_raw"]))

# %%
class CountStateSpace(ssm.StateSpaceModel):
    """ Theta-Logistic state-space model (used in Ecology).
    """
    default_params = {'sigmaX_trend': 0.5, 'sigmaX_seasonal': 0.5}

    def PX0(self):
        X0_dists_trend = [dists.Normal(loc=0, scale=1)]
        X0_dists_seasonal = [
            dists.Normal(loc=0, scale=1)
            for _ in range(6)
        ]
        X0_dists = X0_dists_trend + X0_dists_seasonal
        return dists.IndepProd(*X0_dists)

    def PX(self, t, xp):  #  Distribution of X_t given X_{t-1} = xp (p=past)
        X_dists_trend = [dists.Normal(loc=xp[:, 0], scale=self.sigmaX_trend)]
        X_dists_seasonal = [
            dists.Normal(loc=-1*sum(xp[:, i] for i in range(1,7)), scale=self.sigmaX_seasonal),
            dists.Dirac(loc=xp[:, 2]),
            dists.Dirac(loc=xp[:, 3]),
            dists.Dirac(loc=xp[:, 4]),
            dists.Dirac(loc=xp[:, 5]),
            dists.Dirac(loc=xp[:, 6])
        ]
        X_dists = X_dists_trend + X_dists_seasonal
        return dists.IndepProd(*X_dists)

    def PY(self, t, xp, x):  # Distribution of Y_t given X_t=x, and X_{t-1}=xp
        return dists.Poisson(rate=np.exp(x[:, 0]+x[:, 1]))
    
# %%
# my_ssm = CountStateSpace()  # use default values for all parameters
# x, y = my_ssm.simulate(100)

# %%
# fig = go.Figure(
#     go.Scatter(
#         x=list(range(100)),
#         y=[_y[0] for _y in y],
#         mode="markers"
#     )
# )
# fig

# %%
y = df_data["y_raw"]
my_ssm = CountStateSpace()
fk_boot = ssm.Bootstrap(ssm=my_ssm, data=y)
# my_alg = particles.SMC(fk=fk_boot, N=1000)
# my_alg.run()

# %%
def f(W, X):  # expected signature for the moment function
    return {
        "trend_mean": np.average(X[:, 0], weights=W),
        "seasonal_mean": np.average(X[:, 1], weights=W),
        "y_mean": np.average(np.exp(X[:, 0]+X[:, 1]), weights=W)
    }  # for instance

alg_with_mom = particles.SMC(fk=fk_boot, N=100000, collect=[Moments(mom_func=f)])
alg_with_mom.run()

#%%
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=df_data.index,
        y=y,
        name="data"
    )
)
fig.add_trace(
    go.Scatter(
        x=df_data.index,
        y=[m['seasonal_mean'] for m in alg_with_mom.summaries.moments],
        name="filt_mean"
    )
)
# %%
