import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from core.feature_eng import FeatureEng
from core.preprocess import Preprocess
from core.rf import RF
from core.bp import BP
from core.mlp import MLP
from core.data_loader import DataLoader


data = DataLoader('data/pv.csv').get_data()

X_train, X_test, y_train, y_test = Preprocess.train_test_split(data, 'Active_Power')

X_train = Preprocess.scale(X_train)
X_test = Preprocess.scale(X_test)

X_train = FeatureEng.get_time_feature(X_train)
X_train = FeatureEng.get_statistical_feature(X_train)
X_test = FeatureEng.get_time_feature(X_test)
X_test = FeatureEng.get_statistical_feature(X_test)
X_train = X_train.drop(columns=['timestamp'])
X_test = X_test.drop(columns=['timestamp'])
X_train = X_train.dropna()
y_train = y_train.loc[X_train.index]
X_test = X_test.dropna()
y_test = y_test.loc[X_test.index]

model1 = RF(X_train)
model2 = BP(X_train)
model3 = MLP(X_train)
model1.set_model()
model2.set_model()
model3.set_model()

model1.train(X_train, y_train)
y_pred1 = model1.predict(X_test)
y_pred1 = np.maximum(y_pred1, 0)
r2_1, rmse_1, mae_1 = model1.evaluate(y_test, y_pred1)

model2.train(X_train, y_train)
y_pred2 = model2.predict(X_test)
y_pred2 = np.maximum(y_pred2, 0)
r2_2, rmse_2, mae_2 = model2.evaluate(y_test, y_pred2)

model3.train(X_train, y_train)
y_pred3 = model3.predict(X_test)
y_pred3 = np.maximum(y_pred3, 0)
r2_3, rmse_3, mae_3 = model3.evaluate(y_test, y_pred3)

result = pd.DataFrame({'Actual': y_test, 'RF': y_pred1, 'BP': y_pred2, 'MLP': y_pred3})
result.to_csv('models/result.csv')

res = pd.DataFrame({'RF': [r2_1, rmse_1, mae_1],
                    'BP': [r2_2, rmse_2, mae_2],
                    'MLP': [r2_3, rmse_3, mae_3]},
                   index=['R2', 'RMSE', 'MAE'])
print(res.T)

