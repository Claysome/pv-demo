import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from preprocess import Preprocess
from dataset import PVData
from svm import train_svm, eval_svm, train_svm_for_each_timepoint, eval_svm_for_each_timepoint
from arima import train_resid_arima, train_resid_sarima
from lstm import (
    LSTMModel, 
    train_lstm, eval_lstm, 
    train_lstm_by_group, eval_lstm_by_group, 
    train_resid_lstm, eval_resid_lstm, 
    eval_resid_lstm_arima, eval_resid_lstm_sarima
)


if __name__ == '__main__':
    target_groups = [1, 2, 3, 4, 5, 6]
    file_path = 'data/cleaned_pv.csv'

####################################################SVM###################################################
    imf, resid, test = Preprocess(file_path).get_rep_data()
    # train_dataset = PVData(imf)
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

    # # 训练模型
    # # train_svm(train_loader)
    # train_svm_for_each_timepoint(train_loader)

    test_dataset = PVData(test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    # outputs_values, targets_values = eval_svm(test_loader)
    outputs_values, targets_values = eval_svm_for_each_timepoint(test_loader)
    outputs_values = outputs_values[:300]
    targets_values = targets_values[36:]
    targets_values = targets_values[:len(outputs_values)]

####################################################LSTM###################################################

    # imf, resid, test = Preprocess(file_path).get_rep_data()
    # train_dataset = PVData(imf)
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # input_size = imf.shape[1] - 1
    # hidden_size1 = 100
    # hidden_size2 = 200
    # num_layers = 1
    # output_size = 1

    # model = LSTMModel(input_size, hidden_size1, hidden_size2, num_layers, output_size).to(device)
    # criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    # # 训练模型
    # num_epochs = 15
    # # train_lstm(model, train_loader, criterion, optimizer, num_epochs, device)

    # test_dataset = PVData(test)
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    # model.load_state_dict(torch.load(f'vmd-lstm-resid/models/lstm.pth'))
    # outputs_values, targets_values = eval_lstm(model, test_loader, criterion)


#################################################VMD-LSTM###################################################

    # for target_group in target_groups:
    #     imf, resid, test = Preprocess(file_path).get_rep_data(target_group)
    #     train_dataset = PVData(imf)
    #     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #     input_size = imf.shape[1] - 1
    #     hidden_size1 = 100
    #     hidden_size2 = 200
    #     num_layers = 1
    #     output_size = 1

    #     model = LSTMModel(input_size, hidden_size1, hidden_size2, num_layers, output_size).to(device)
    #     criterion = nn.MSELoss()
    #     optimizer = optim.Adam(model.parameters(), lr=0.001)

        # # 训练模型
        # num_epochs = 15
        # train_lstm_by_group(model, train_loader, criterion, optimizer, num_epochs, device, target_group)

    # imf, resid, test = Preprocess(file_path).get_rep_data()
    # test_dataset = PVData(test)
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # input_size = imf.shape[1] - 1
    # hidden_size1 = 100
    # hidden_size2 = 200
    # num_layers = 1
    # output_size = 1

    # model = LSTMModel(input_size, hidden_size1, hidden_size2, num_layers, output_size).to(device)
    # criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    # outputs_values, targets_values = eval_lstm_by_group(model, test_loader, criterion, target_groups)


#################################################VMD-LSTM-RESID###################################################

    # imf, resid, test = Preprocess(file_path).get_rep_data()
    # resid_dataset = PVData(resid)
    # resid_loader = DataLoader(resid_dataset, batch_size=32, shuffle=False)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # input_size = imf.shape[1] - 1
    # hidden_size1 = 100
    # hidden_size2 = 200
    # num_layers = 1
    # output_size = 1

    # model = LSTMModel(input_size, hidden_size1, hidden_size2, num_layers, output_size).to(device)
    # criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    # outputs_values, targets_values = eval_lstm_by_group(model, resid_loader, criterion, target_groups)
    # resid_values = outputs_values.flatten() - targets_values.flatten()
    # resid = resid.iloc[-len(resid_values):]
    # resid["Active_Power"] = resid_values
    # resid_dataset = PVData(resid)
    # resid_loader = DataLoader(resid_dataset, batch_size=32, shuffle=False)

    # num_epochs = 10
    # train_resid_lstm(model, resid_loader, criterion, optimizer, num_epochs, device)
    # train_resid_arima(resid_loader)
    # train_resid_sarima(resid_loader)

    # test_dataset = PVData(test)
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    # outputs_values, targets_values = eval_resid_lstm_sarima(model, test_loader, target_groups)


#################################################PLOT#########################################################
    import matplotlib.pyplot as plt
    plt.plot(targets_values, label='True')
    plt.plot(outputs_values, label='Predicted')
    plt.legend()
    plt.show()
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
    print(r2_score(targets_values, outputs_values))
    # print(mean_squared_error(targets_values, outputs_values, squared=False))