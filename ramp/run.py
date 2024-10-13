# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd

from preprocess import Preprocess
from tasks import PredictionTasks
from dataset import WindData
from lstm import LSTMModel


data_path = f"data/pv_power.csv"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

preprossor = Preprocess(data_path)
data_org, data_trend, data_seasonal, data_residual = preprossor.run()

data_org_train, data_org_test = PredictionTasks.split_train_test(data_org)
data_trend_train, data_trend_test = PredictionTasks.split_train_test(data_trend)
data_seasonal_train, data_seasonal_test = PredictionTasks.split_train_test(data_seasonal)
data_residual_train, data_residual_test = PredictionTasks.split_train_test(data_residual)

# print(data_org_train.head(30))

################################################## TRAIN ORG SIGNAL #######################################################
# # train org signal
# train_org_dataset = WindData(data_org_train)
# train_org_loader = DataLoader(train_org_dataset, batch_size=32, shuffle=True)
params = {
    "input_size": data_org_train.shape[1] - 1,
    "hidden_size1": 100,
    "hidden_size2": 200,
    "num_layers": 1,
    "output_size": 1,
}
model_org = LSTMModel(**params).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model_org.parameters(), lr=0.005)
num_epochs = 10
# PredictionTasks.train_lstm(model_org, train_org_loader, criterion, optimizer, num_epochs, device, "org_lstm")

# test org signal
test_org_dataset = WindData(data_org_test)
test_org_loader = DataLoader(test_org_dataset, batch_size=32, shuffle=False)
model_org.load_state_dict(torch.load('ramp/models/org_lstm.pth'))
outputs_values, targets_values = PredictionTasks.eval_lstm(model_org, test_org_loader, device)


################################################## TRAIN STL SIGNAL #######################################################
# train stl signal
train_t_dataset = WindData(data_trend_train)
train_s_dataset = WindData(data_seasonal_train)
train_r_dataset = WindData(data_residual_train)
train_t_loader = DataLoader(train_t_dataset, batch_size=32, shuffle=True)
train_s_loader = DataLoader(train_s_dataset, batch_size=32, shuffle=True)
train_r_loader = DataLoader(train_r_dataset, batch_size=32, shuffle=True)
params = {
    "input_size": data_org_train.shape[1] - 1,
    "hidden_size1": 100,
    "hidden_size2": 200,
    "num_layers": 1,
    "output_size": 1,
}
model = LSTMModel(**params).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)
num_epochs = 10
# PredictionTasks.train_lstm(model, train_t_loader, criterion, optimizer, num_epochs, device, "trend_lstm")
# PredictionTasks.train_lstm(model, train_s_loader, criterion, optimizer, num_epochs, device, "seasonal_lstm")
# PredictionTasks.train_lstm(model, train_r_loader, criterion, optimizer, num_epochs, device, "residual_lstm")

# test stl signal
test_t_dataset = WindData(data_trend_test)
test_s_dataset = WindData(data_seasonal_test)
test_r_dataset = WindData(data_residual_test)
test_t_loader = DataLoader(test_t_dataset, batch_size=32, shuffle=False)
test_s_loader = DataLoader(test_s_dataset, batch_size=32, shuffle=False)
test_r_loader = DataLoader(test_r_dataset, batch_size=32, shuffle=False)
model_t = LSTMModel(**params).to(device)
model_s = LSTMModel(**params).to(device)
model_r = LSTMModel(**params).to(device)
model_t.load_state_dict(torch.load('ramp/models/trend_lstm.pth'))
model_s.load_state_dict(torch.load('ramp/models/seasonal_lstm.pth'))
model_r.load_state_dict(torch.load('ramp/models/residual_lstm.pth'))
outputs_values_t, targets_values_t = PredictionTasks.eval_lstm(model_t, test_t_loader, device)
outputs_values_s, targets_values_s = PredictionTasks.eval_lstm(model_s, test_s_loader, device)
outputs_values_r, targets_values_r = PredictionTasks.eval_lstm(model_r, test_r_loader, device)

# sum stl signal
outputs_values_stl = outputs_values_t + outputs_values_s + outputs_values_r
outputs_values_stl = [0 if x < 0 else x for x in outputs_values_stl]
targets_values_stl = targets_values_t + targets_values_s + targets_values_r

################################################## EVENT DETECTION #######################################################

detection = pd.DataFrame({"true": data_org_test["TARGET"][-len(outputs_values):]})
detection["pred_org"] = outputs_values
detection["pred_trend"] = outputs_values_t
detection["pred_seasonal"] = outputs_values_s
detection["pred_residual"] = outputs_values_r
detection["ramp_true"] = False
detection["ramp_org"] = False
detection["ramp_trend"] = False
detection["ramp_seasonal"] = False
detection["ramp_residual"] = False

# detection.to_csv("ramp/res/pred.csv", index=True)


# import matplotlib.pyplot as plt
# plt.figure(figsize=(12, 6))
# plt.plot(outputs_values_t, label='Predictions')
# plt.plot(targets_values_t, label='True Values')
# plt.legend()
# plt.show()
# from sklearn.metrics import r2_score, mean_squared_error
# print(r2_score(targets_values, outputs_values))
# print(r2_score(targets_values_stl, outputs_values_stl))
# print(mean_squared_error(targets_values, outputs_values))
# print(mean_squared_error(targets_values_stl, outputs_values_stl))
# res_df = pd.DataFrame({"pred": outputs_values, "true": targets_values}).to_csv("ramp/res/org_lstm.csv", index=False)