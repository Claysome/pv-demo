# -*- coding: utf-8 -*-


import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import pandas as pd
from sklearn.metrics import confusion_matrix


lstm = pd.read_csv('ramp-stl-ec/res/lstm.csv')
pca_lstm = pd.read_csv('ramp-stl-ec/res/pca_lstm.csv')
pca_stl_lstm = pd.read_csv('ramp-stl-ec/res/pca_stl_lstm.csv')

pca_lstm['pred'] = [float(pred.strip('[]')) for pred in pca_lstm['pred']]
pca_stl_lstm['pred'] = [float(pred.strip('[]')) for pred in pca_stl_lstm['pred']]


postive = 50 * 0.2

date_range = pd.date_range('2020-01-01', '2020-12-31', freq='15min')

lstm['timestamp'] = date_range[:len(lstm)]
pca_lstm['timestamp'] = date_range[:len(pca_lstm)]
pca_stl_lstm['timestamp'] = date_range[:len(pca_stl_lstm)]

lstm.set_index('timestamp', inplace=True)
pca_lstm.set_index('timestamp', inplace=True)
pca_stl_lstm.set_index('timestamp', inplace=True)

lstm = lstm.resample('h').sum()
pca_lstm = pca_lstm.resample('h').sum()
pca_stl_lstm = pca_stl_lstm.resample('h').sum()

def detect_ramp_event(value1, value2, postive=80*0.15):
    diff = abs(value2 - value1)
    return 1 if diff > postive else 0


for data in [lstm, pca_lstm, pca_stl_lstm]:
    # 生成真实的爬坡事件标签
    true_labels = [detect_ramp_event(data['true'].iloc[i], data['true'].iloc[i + 1]) for i in range(len(data['true']) - 1)]

    # 生成预测的爬坡事件标签（以 lstm 的预测为例，你可以根据需要更换为其他预测结果）
    pred_labels = [detect_ramp_event(data['pred'].iloc[i], data['pred'].iloc[i + 1]) for i in range(len(data['pred']) - 1)]

    # 生成混淆矩阵
    conf_matrix = confusion_matrix(true_labels, pred_labels)
    TP = conf_matrix[1, 1]
    TN = conf_matrix[0, 0]
    FP = conf_matrix[0, 1]
    FN = conf_matrix[1, 0]
    FA = TP / (TP + FP)
    RC = TP / (TP + FN)
    CSI = TP / (TP + FN + FP)

    print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}, FA: {FA:4f}, RC: {RC:4f}, CSI: {CSI:4f}")

