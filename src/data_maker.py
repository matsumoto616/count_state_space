import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from scipy.stats import yeojohnson

def make_weekly_and_yearly_data(start_date, end_date, seed=0):
    rng = np.random.default_rng(seed=seed)

    weekday_true = [0, 100, 0, 10, 50, 100, 0]
    weekday_range = [0, 20, 0, 5, 10, 20, 0]
    weekday_peak_prob = [0.1, 0, 0, 0, 0, 0, 0.1]
    weekday_peak_value = [20, 0, 0, 0, 0, 0, 20]
    weekday_missing_prob = [0, 0.1, 0, 0, 0, 0.1, 0]
    
    yearly_trend = [max(int(10*np.sin(2*np.pi*i/ 365)), 0) for i in range(365)]

    df_data = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date))
    df_data["weekday"] = df_data.index.map(lambda x: x.weekday())
    df_data["yt_weekly"] = df_data["weekday"].map(lambda x: weekday_true[x])
    df_data["y_weekly"] = df_data["weekday"].map(
        lambda x: make_random_data(
            x,
            weekday_true,
            weekday_range,
            weekday_peak_prob,
            weekday_peak_value,
            weekday_missing_prob,
            rng
        )
    )

    df_data["y_yearly"] = [yearly_trend[i%365] for i in range(len(df_data))]
    df_data["z"] = df_data["y_yearly"].map(lambda x: int(bool(x))) * df_data["weekday"].map(lambda x: int(bool(weekday_range[x])))
    df_data["y_raw"] = df_data["y_weekly"] * df_data["y_yearly"]
    df_data["y_raw"] = df_data["y_raw"] * df_data["z"]

    return df_data


def make_random_data(
        weekday,
        weekday_true,
        weekday_range,
        weekday_peak_prob,
        weekday_peak_value,
        weekday_missing_prob,
        rng
    ):
    
    value = weekday_true[weekday]
    value += rng.integers(-weekday_range[weekday], weekday_range[weekday], endpoint=True)
    if rng.random() < weekday_peak_prob[weekday]:
        value += weekday_peak_value[weekday]
    if rng.random() < weekday_missing_prob[weekday]:
        value = 0

    return value