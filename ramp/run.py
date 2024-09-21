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

# train org signal
train_org_dataset = WindData(data_org_train)
train_org_loader = DataLoader(train_org_dataset, batch_size=32, shuffle=True)
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
PredictionTasks.train_lstm(model_org, train_org_loader, criterion, optimizer, num_epochs, device, "org_lstm")

# test org signal
test_org_dataset = WindData(data_org_test)
test_org_loader = DataLoader(test_org_dataset, batch_size=32, shuffle=False)
model_org.load_state_dict(torch.load('ramp/models/org_lstm.pth'))
outputs_values, targets_values = PredictionTasks.eval_lstm(model_org, test_org_loader, device)

import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(outputs_values, label='Predictions')
plt.plot(targets_values, label='True Values')
plt.legend()
plt.show()
from sklearn.metrics import r2_score, mean_squared_error
print(r2_score(targets_values, outputs_values))
# print(mean_squared_error(targets_values, outputs_values, squared=False))
# res_df = pd.DataFrame({"pred": outputs_values, "true": targets_values}).to_csv("ramp/res/org_lstm.csv", index=False)