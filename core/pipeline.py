import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np

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

model = RF(X_train)
# model = BP(X_train)
# model = MLP(X_train)
model.set_model()
model.train(X_train, y_train)
y_pred = model.predict(X_test)
y_pred = np.maximum(y_pred, 0)
model.evaluate(y_test, y_pred)
model.plot(y_test, y_pred)