import ccxt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, ConstantKernel as C
from scipy.optimize import minimize_scalar

exchange = ccxt.binance()

symbol = 'BTC/USDT'
timeframe = '1w'
limit = 200

ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
data = np.array(ohlcv)

X = np.arange(len(data)).reshape(-1, 1)
y = np.log(data[:, 4])

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)    
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

kernel = (
    C(1.0, (1e-3, 1e3)) *
    RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    + DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-3, 1e3))
    + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e1))
)

gpr = GaussianProcessRegressor(
    kernel=kernel,
    alpha=1e-7,
    n_restarts_optimizer=30,
    normalize_y=True
)

gpr.fit(X_scaled, y_scaled)

X_pred = np.linspace(0, len(data) + 20, 300).reshape(-1, 1)
X_pred_scaled = scaler_X.transform(X_pred)

y_pred_scaled, sigma_scaled = gpr.predict(X_pred_scaled, return_std=True)

y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

price_range = scaler_y.data_max_ - scaler_y.data_min_
sigma = sigma_scaled * price_range

plt.figure(figsize=(12, 6))


plt.plot(X_pred, y_pred, "b", label="GPR Prediction")
plt.fill_between(
    X_pred.ravel(),
    y_pred - 1.96 * sigma,
    y_pred + 1.96 * sigma,
    alpha=0.2,
    color="blue",
    label="95% Confidence Interval"
)
plt.scatter(X, y, color="black", label="Observed Prices", s=20)
plt.title(f"Gaussian Process Regression on {symbol} ({timeframe})")
plt.xlabel("Time (index)")
plt.ylabel("Price (USDT)")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend()
plt.show()
