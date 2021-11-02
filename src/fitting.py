import pandas as pd
import xarray as xr
from lmfit.models import (
    LinearModel,
    LorentzianModel,
    GaussianModel,
    PolynomialModel,
    Model,
)
import matplotlib.pyplot as plt


def datetime_to_int(dt):
    return int(dt.strftime("%H%M"))


def gaussian(x, amp, cen, wid):
    return amp * exp(-((x - cen) ** 2) / wid)


def line(x, a1=1, a2=1, a3=1, b=1):
    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]
    return a1 * x1 + a2 * x2 + a3 * x3 + b


if __name__ == "__main__":
    # dframe = pd.read_csv('peak.csv')
    da = xr.open_dataarray("data/sims.nc")
    df = da.sel(rh=10).to_dataframe(name="dis")
    df = df.reset_index()
    df["hour_minute"] = df["times"].apply(lambda x: datetime_to_int(x))
    print(df.tail())

    # model = Model(line)
    # print(f"parameter names: {model.param_names}")
    # print(f"independent variables: {model.independent_vars}")
    # params = model.make_params(a1=0.3, a2=3, a3=1.25, b=0)

    # model = LorentzianModel()
    model = GaussianModel()

    # model = SineModel()
    # model = PolynomialModel()
    # model = LinearModel()
    # params = model.guess(df["dis"], x=df.int_time)
    params = model.guess(df["dis"], x=df.hour_minute)

    result = model.fit(df["dis"], params, x=df.hour_minute)
    print(result.fit_report())
    plt.figure()
    ax = result.plot_fit()
    # plt.plot(df.index.hour, result.best_fit, "-", label="best fit")
    plt.legend()
    plt.savefig("figs/fit.jpg")
