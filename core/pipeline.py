import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.feature_eng import FeatureEng
from core.preprocess import Preprocess
from core.rf import RF
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


rf = RF(X_train)
rf.set_model()
rf.train(X_train, y_train)
y_pred = rf.predict(X_test)
rf.evaluate(y_test, y_pred)

# plot
import matplotlib.pyplot as plt
plt.plot(y_test.values, label='actual')
plt.plot(y_pred, label='prediction')
plt.legend()
plt.show()