import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd

from dataset import PVData
from preprocess import Preprocess
from lstm import (
    LSTMModel,
    train_lstm,
    train_pca_lstm,
    train_pca_stl_lstm,
    eval_lstm,
    eval_pca_stl_lstm
)

if __name__ == '__main__':
    file_path = 'data/cleaned_pv.csv'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
####################################################LSTM###################################################
    # sig, resid, test = Preprocess(file_path).get_data()
    # train_dataset = PVData(sig)
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # input_size = sig.shape[1] - 1
    # hidden_size = 80
    # num_layers = 1
    # output_size = 1

    # model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
    # criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.005)

    # # 训练模型1
    # num_epochs = 50
    # train_lstm(model, train_loader, criterion, optimizer, num_epochs, device)

    # test_dataset = PVData(test)
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    # model.load_state_dict(torch.load('ramp-stl-ec/models/lstm.pth'))
    # outputs_values, targets_values = eval_lstm(model, test_loader, device)


####################################################PCA-LSTM###################################################
    # sig, resid, test = Preprocess(file_path).get_data(pca=True)
    # train_dataset = PVData(sig)
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # input_size = sig.shape[1] - 1
    # hidden_size = 80
    # num_layers = 1
    # output_size = 1

    # model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
    # criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.005)

    # # 训练模型1
    # num_epochs = 50
    # train_pca_lstm(model, train_loader, criterion, optimizer, num_epochs, device)

    # test_dataset = PVData(test)
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    # model.load_state_dict(torch.load('ramp-stl-ec/models/pca_lstm.pth'))
    # outputs_values, targets_values = eval_lstm(model, test_loader, device)


####################################################PCA-STL-LSTM###################################################
    stl_labels = ['trend', 'seasonal', 'resid']
    label = 'resid'
    sig, resid, test = Preprocess(file_path).get_data(pca=True, stl=True, label=label)
    # print(f"Training model for {label}")
    # train_dataset = PVData(sig)
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # input_size = sig.shape[1] - 1
    # hidden_size = 80
    # num_layers = 1
    # output_size = 1

    # model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
    # criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.005)

    # # 训练模型1
    # num_epochs = 20
    # train_pca_stl_lstm(model, train_loader, criterion, optimizer, num_epochs, device, label)

    # sig, resid, test = Preprocess(file_path).get_data(pca=True, stl=False)
    # input_size = sig.shape[1] - 1
    # hidden_size = 80
    # num_layers = 1
    # output_size = 1

    # model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
    # test_dataset = PVData(test)
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    # outputs_values, targets_values = eval_pca_stl_lstm(model, test_loader, device, stl_labels)


#################################################PLOT###################################################
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(12, 6))
    # plt.plot(outputs_values, label='Predictions')
    # plt.plot(targets_values, label='True Values')
    # plt.legend()
    # plt.show()
    # from sklearn.metrics import r2_score, mean_squared_error
    # print(r2_score(targets_values, outputs_values))
    # print(mean_squared_error(targets_values, outputs_values, squared=False))
    # res_df = pd.DataFrame({"pred": outputs_values, "true": targets_values.flatten()}).to_csv("ramp-stl-ec/res/pca_stl_lstm.csv", index=False)