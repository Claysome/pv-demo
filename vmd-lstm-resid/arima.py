import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from itertools import product
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tqdm import tqdm


def train_resid_arima(train_loader):
    resid_values = []
    for inputs, targets in train_loader:
        resid_values.extend(targets.cpu().numpy())
    resid_values = np.array(resid_values)
    
    p_values = np.arange(0, 3)
    d_values = np.arange(0, 3)
    q_values = np.arange(0, 3)
    best_score, best_cfg = float("inf"), None
    for p, d, q in product(p_values, d_values, q_values):
        order = (p, d, q)
        try:
            model = ARIMA(resid_values, order=order)
            model_fit = model.fit()
            yhat = model_fit.predict()
            mse = mean_squared_error(resid_values, yhat)
            if mse < best_score:
                best_score, best_cfg = mse, order
            print(f'ARIMA{order} MSE={mse}')
        except:
            continue
    print(f'Best ARIMA{best_cfg} MSE={best_score}')
    model = ARIMA(resid_values, order=best_cfg)
    model_fit = model.fit()
    model_fit.save('vmd-lstm-resid/models/arima.pkl')
    print('Model Saved')


def train_resid_sarima(train_loader):
    resid_values = []
    feature_values = []
    for inputs, targets in train_loader:
        resid_values.extend(targets.cpu().numpy())
        inputs_flatten = inputs.cpu().numpy().reshape(inputs.size(0), -1)
        feature_values.extend(inputs_flatten)
    resid_values = np.array(resid_values)
    feature_values = np.array(feature_values)

    df = pd.DataFrame(feature_values)
    df['resid'] = resid_values

    p_values = np.arange(0, 3)
    d_values = np.arange(0, 3)
    q_values = np.arange(0, 3)
    best_score, best_cfg = float("inf"), None
    for p, d, q in tqdm(product(p_values, d_values, q_values)):
        order = (p, d, q)
        try:
            model = SARIMAX(df['resid'], exog=df.drop('resid', axis=1), order=order)
            model_fit = model.fit(disp=False)
            yhat = model_fit.predict(exog=df.drop('resid', axis=1))
            mse = mean_squared_error(df['resid'], yhat)
            if mse < best_score:
                best_score, best_cfg = mse, order
            print(f'SARIMA{order} MSE={mse}')
            model_fit.save('vmd-lstm-resid/models/sarima.pkl')
        except:
            pass
    print(f'Best SARIMA{best_cfg} MSE={best_score}')
    model = SARIMAX(df['resid'], exog=df.drop('resid', axis=1), order=best_cfg)
    model_fit = model.fit(disp=False)
    model_fit.save('vmd-lstm-resid/models/sarima.pkl')
    print('Model Saved')