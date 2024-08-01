import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib


class LSTMModel(nn.Module):

    def __init__(self, input_size, hidden_size1, hidden_size2, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size1, num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out = self.fc(out[:, -1, :])
        return out


def train_lstm_by_group(model, train_loader, criterion, optimizer, num_epochs, device, target_group):
    for epoch in tqdm(range(num_epochs)):
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')
    print('Finished Training')
    torch.save(model.state_dict(), f'vmd-lstm-resid/models/lstm_{target_group}.pth')
    print('Model Saved')


def eval_lstm_by_group(model, test_loader, criterion, target_groups):
    targets_values = []
    outputs_values = []
    for target in target_groups:
        model.load_state_dict(torch.load(f'vmd-lstm-resid/models/lstm_{target}.pth'))
        model.eval()
        outputs_group = []
        print(f"Evaluating model for target group {target}...")
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                outputs_group.extend(outputs.cpu().numpy())
        outputs_values.append(outputs_group)
    outputs_values = np.sum(outputs_values, axis=0)
    outputs_values = np.maximum(outputs_values, 0).reshape(-1, 1)
    filter = pd.DataFrame({"timestamp": test_loader.dataset.data.index[-len(outputs_values):], "values": outputs_values.flatten()})
    filter.set_index("timestamp", inplace=True)
    filter["hour"] = filter.index.hour
    filter["values"] = filter.apply(lambda x: 0 if x["hour"] <= 6 or x["hour"] >= 20 else x["values"], axis=1)
    outputs_values = filter["values"].values
        
    for inputs, targets in test_loader:
        targets_values.extend(targets.cpu().numpy())
    targets_values = np.array(targets_values)
    return outputs_values, targets_values


def train_lstm(model, train_loader, criterion, optimizer, num_epochs, device):
    for epoch in tqdm(range(num_epochs)):
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')
    print('Finished Training')
    torch.save(model.state_dict(), f'vmd-lstm-resid/models/lstm.pth')
    print('Model Saved')


def eval_lstm(model, test_loader, criterion):
    model.eval()
    # total_loss = 0
    outputs_values, targets_values = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            # loss = criterion(outputs, targets)
            outputs_values.extend(outputs.cpu().numpy())
            targets_values.extend(targets.cpu().numpy())
    #         total_loss += loss.item()
    # print(f'Test Loss: {total_loss/len(test_loader):.4f}')
    filter = pd.DataFrame({"timestamp": test_loader.dataset.data.index[-len(outputs_values):], "values": outputs_values})
    filter.set_index("timestamp", inplace=True)
    filter["hour"] = filter.index.hour
    filter["values"] = filter.apply(lambda x: 0 if x["hour"] <= 6 or x["hour"] >= 20 else x["values"], axis=1)
    outputs_values = filter["values"].values
    return np.array(outputs_values), np.array(targets_values)


def train_resid_lstm(model, train_loader, criterion, optimizer, num_epochs, device):
    for epoch in tqdm(range(num_epochs)):
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')
    print('Finished Training')
    torch.save(model.state_dict(), f'vmd-lstm-resid/models/lstm_resid.pth')
    print('Model Saved')


def eval_resid_lstm(model, test_loader, criterion, target_groups):
    targets_values = []
    outputs_values = []
    resid_values = []
    for target in target_groups:
        model.load_state_dict(torch.load(f'vmd-lstm-resid/models/lstm_{target}.pth'))
        model.eval()
        outputs_group = []
        print(f"Evaluating model for target group {target}...")
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                outputs_group.extend(outputs.cpu().numpy())
        outputs_values.append(outputs_group)
    outputs_values = np.sum(outputs_values, axis=0)
    outputs_values = np.maximum(outputs_values, 0).reshape(-1, 1)

    model.load_state_dict(torch.load(f'vmd-lstm-resid/models/lstm_resid.pth'))
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            resid_values.extend(outputs.cpu().numpy())
    
    outputs_values = outputs_values + np.array(resid_values)
    outputs_values = np.maximum(outputs_values, 0).reshape(-1, 1)
    filter = pd.DataFrame({"timestamp": test_loader.dataset.data.index[-len(outputs_values):], "values": outputs_values.flatten()})
    filter.set_index("timestamp", inplace=True)
    filter["hour"] = filter.index.hour
    filter["values"] = filter.apply(lambda x: 0 if x["hour"] <= 6 or x["hour"] >= 20 else x["values"], axis=1)
    outputs_values = filter["values"].values
        
    for inputs, targets in test_loader:
        targets_values.extend(targets.cpu().numpy())
    targets_values = np.array(targets_values)
    return outputs_values, targets_values


def eval_resid_lstm_arima(model, test_loader, target_groups):
    targets_values = []
    outputs_values = []
    for target in target_groups:
        model.load_state_dict(torch.load(f'vmd-lstm-resid/models/lstm_{target}.pth'))
        model.eval()
        outputs_group = []
        print(f"Evaluating model for target group {target}...")
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                outputs_group.extend(outputs.cpu().numpy())
        outputs_values.append(outputs_group)
    outputs_values = np.sum(outputs_values, axis=0)
    outputs_values = np.maximum(outputs_values, 0).reshape(-1, 1)

    arima = joblib.load('vmd-lstm-resid/models/arima.pkl')
    resid_values = arima.forecast(steps=len(outputs_values))
    outputs_values = outputs_values + np.array(resid_values.reshape(-1, 1))
    filter = pd.DataFrame({"timestamp": test_loader.dataset.data.index[-len(outputs_values):], "values": outputs_values.flatten()})
    filter.set_index("timestamp", inplace=True)
    filter["hour"] = filter.index.hour
    filter["values"] = filter.apply(lambda x: 0 if x["hour"] <= 6 or x["hour"] >= 20 else x["values"], axis=1)
    outputs_values = filter["values"].values
        
    for inputs, targets in test_loader:
        targets_values.extend(targets.cpu().numpy())
    targets_values = np.array(targets_values)
    return outputs_values, targets_values


def eval_resid_lstm_sarima(model, test_loader, target_groups):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    targets_values = []
    outputs_values = []
    resid_values = []
    for target in target_groups:
        model.load_state_dict(torch.load(f'vmd-lstm-resid/models/lstm_{target}.pth'))
        model.eval()
        outputs_group = []
        print(f"Evaluating model for target group {target}...")
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                outputs_group.extend(outputs.cpu().numpy())
        outputs_values.append(outputs_group)
    outputs_values = np.sum(outputs_values, axis=0)
    outputs_values = np.maximum(outputs_values, 0).reshape(-1, 1)

    sarima_model = joblib.load('vmd-lstm-resid/models/sarima.pkl')
    feature_values = []
    for inputs, targets in test_loader:
        resid_values.extend(targets.cpu().numpy())
        inputs_flatten = inputs.cpu().numpy().reshape(inputs.size(0), -1)
        feature_values.extend(inputs_flatten)
    resid_values = np.array(resid_values)
    feature_values = np.array(feature_values)

    df = pd.DataFrame(feature_values)
    resid_values = sarima_model.predict(exog=df)
    resid_values = np.array(resid_values[:len(outputs_values)])
    outputs_values = outputs_values + resid_values.reshape(-1, 1)
    outputs_values = np.maximum(outputs_values, 0).reshape(-1, 1)

    filter = pd.DataFrame({"timestamp": test_loader.dataset.data.index[-len(outputs_values):], "values": outputs_values.flatten()})
    filter.set_index("timestamp", inplace=True)
    filter["hour"] = filter.index.hour
    filter["values"] = filter.apply(lambda x: 0 if x["hour"] <= 6 or x["hour"] >= 20 else x["values"], axis=1)
    outputs_values = filter["values"].values
        
    for inputs, targets in test_loader:
        targets_values.extend(targets.cpu().numpy())
    targets_values = np.array(targets_values)
    return outputs_values, targets_values