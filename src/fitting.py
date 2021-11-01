import pandas as pd
import xarray as xr
from lmfit.models import LinearModel, LorentzianModel, SineModel, PolynomialModel, Model
import matplotlib.pyplot as plt


def datetime_to_int(dt):
    return int(dt.strftime("%m%d%H"))


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
    print(df.head())
    print(df.tail())
    model = Model(line)
    print(f"parameter names: {model.param_names}")
    print(f"independent variables: {model.independent_vars}")
    params = model.make_params(a1=0.3, a2=3, a3=1.25, b=0)

    # model = LorentzianModel()
    # model = SineModel()
    # model = PolynomialModel()
    # model = LinearModel()
    # params = model.guess(df["dis"], x=df.int_time)
    # params = model.guess(df["dis"], x=[df.temp, df.rh, df.v])

    result = model.fit(df["dis"], params, x=[df.temp, df.rh, df.v])
    print(result.fit_report())
    plt.figure()
    ax = result.plot_fit()
    # plt.plot(df.index.hour, result.best_fit, "-", label="best fit")
    plt.legend()
    plt.savefig("figs/fit.jpg")
