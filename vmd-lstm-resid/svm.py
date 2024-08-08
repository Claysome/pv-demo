import joblib
import numpy as np
import pandas as pd
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
    best_clf.fit(X, y)
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
    return np.array(outputs_values), np.array(targets_values)


def train_svm_for_each_timepoint(loader):
    # 逐时间点训练模型，全天96个时间点
    for i in range(96):
        exec(f"X_{i} = []")
        exec(f"y_{i} = []")
    
    org_inputs, org_targets = [], []
    for inputs, targets in loader:
        inputs_flatten = inputs.cpu().numpy().reshape(inputs.size(0), -1)
        org_inputs.extend(inputs_flatten)
        org_targets.extend(targets.cpu().numpy())

    for i in range(len(org_inputs)):
        for j in range(96):
            if (i + 96) % 96 == j:
                exec(f"X_{j}.append(org_inputs[i])")
                exec(f"y_{j}.append(org_targets[i])")
    for i in range(96):
        exec(f"X_{i} = np.array(X_{i})")
        exec(f"y_{i} = np.array(y_{i})")
        exec(f"clf_{i} = svm.SVR(C=100, epsilon=0.2, gamma='scale', kernel='rbf', max_iter=10000)")
        exec(f"clf_{i}.fit(X_{i}, y_{i})")
        exec(f"joblib.dump(clf_{i}, 'vmd-lstm-resid/models/svm_{i}.pkl')")
        print(f"Model_{i} Saved")


def eval_svm_for_each_timepoint(loader):
    outputs_values, targets_values = [], []
    for i in range(96):
        exec(f"clf_{i} = joblib.load('vmd-lstm-resid/models/svm_{i}.pkl')")
    org_inputs, org_targets = [], []
    for inputs, targets in loader:
        inputs_flatten = inputs.cpu().numpy().reshape(inputs.size(0), -1)
        org_inputs.extend(inputs_flatten)
        org_targets.extend(targets.cpu().numpy())

    for i in range(len(org_inputs)):
        for j in range(96):
            if i % 96 == j:
                exec(f"outputs = clf_{j}.predict(org_inputs[i].reshape(1, -1))")
                exec(f"outputs_values.extend(outputs)")
                targets_values.extend(org_targets[i])

    # filter = pd.DataFrame({"timestamp": loader.dataset.data.index[-len(outputs_values):], "values": outputs_values.flatten()})
    # filter.set_index("timestamp", inplace=True)
    # filter["hour"] = filter.index.hour
    # filter["values"] = filter.apply(lambda x: 0 if x["hour"] <= 6 or x["hour"] >= 20 else x["values"], axis=1)
    # outputs_values = filter["values"].values
    return np.array(outputs_values), np.array(targets_values)