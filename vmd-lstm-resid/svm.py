import joblib
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV


def train_svm(loader):
    X, y = [], []
    for inputs, targets in loader:
        inputs_flatten = inputs.cpu().numpy().reshape(inputs.size(0), -1)
        X.extend(inputs_flatten)
        y.extend(targets.cpu().numpy())
    X = np.array(X)
    y = np.array(y)

    # param_grid = {
    #     'C': [0.1, 1, 10, 100],
    #     'epsilon': [0.01, 0.1, 0.2],
    #     'kernel': ['rbf', 'linear', 'poly'],
    #     'gamma': ['scale', 'auto'],
    #     'max_iter': [1000, 5000, 10000]
    # }
    # clf = svm.SVR()
    # grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
    # grid_search.fit(X, y)
    # print("最佳参数组合:", grid_search.best_params_)
    # 最佳参数组合: {'C': 100, 'epsilon': 0.2, 'gamma': 'scale', 'kernel': 'rbf', 'max_iter': 10000}
    # best_clf = grid_search.best_estimator_
    best_clf = svm.SVR(C=100, epsilon=0.2, gamma='scale', kernel='rbf', max_iter=10000)
    joblib.dump(best_clf, 'vmd-lstm-resid/models/svm.pkl')
    print('Model Saved')


def eval_svm(loader):
    clf = joblib.load('vmd-lstm-resid/models/svm.pkl')
    outputs_values, targets_values = [], []
    for inputs, targets in loader:
        inputs = inputs.view(inputs.size(0), -1)
        outputs = clf.predict(inputs.cpu().numpy())
        outputs_values.extend(outputs)
        targets_values.extend(targets.cpu().numpy())
    return outputs_values, targets_values