import ccxt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, ConstantKernel as C
exchange = ccxt.binance()

symbol = 'SOL/USDT'
timeframe = '4h'
limit = 200 

ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
data = np.array(ohlcv)

X = np.arange(len(data)).reshape(-1, 1)
y = data[:, 4]

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=5.0) + DotProduct() + WhiteKernel()
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.0)

gpr.fit(X_scaled, y_scaled)

X_pred = np.linspace(0, len(data) + 20, 300).reshape(-1, 1)
X_pred_scaled = scaler_X.transform(X_pred)

y_pred_scaled, sigma_scaled = gpr.predict(X_pred_scaled, return_std=True)

y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

price_range = scaler_y.data_max_ - scaler_y.data_min_
sigma = sigma_scaled * price_range

plt.figure(figsize=(12, 6))

plt.scatter(X, y, color="black", label="Observed Prices", s=20)

plt.plot(X_pred, y_pred, "b", label="GPR Prediction")
plt.fill_between(
    X_pred.ravel(),
    y_pred - 1.96 * sigma,
    y_pred + 1.96 * sigma,
    alpha=0.2,
    color="blue",
    label="95% Confidence Interval"
)

plt.title(f"Gaussian Process Regression on {symbol} ({timeframe})")
plt.xlabel("Time (index)")
plt.ylabel("Price (USDT)")
plt.legend()
plt.show()