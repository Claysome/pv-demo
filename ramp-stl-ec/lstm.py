import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib


class LSTMModel(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm1(x)
        # out, _ = self.lstm2(out)
        out = self.fc(out[:, -1, :])
        return out
    

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
    torch.save(model.state_dict(), f'ramp-stl-ec/models/lstm.pth')
    print('Model Saved')


def eval_lstm(model, test_loader, device):
    model.eval()
    outputs_values, targets_values = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            outputs_values.extend(outputs.cpu().numpy())
            targets_values.extend(targets.cpu().numpy())
    filter = pd.DataFrame({"timestamp": test_loader.dataset.data.index[-len(outputs_values):], "values": outputs_values})
    filter.set_index("timestamp", inplace=True)
    filter["hour"] = filter.index.hour
    filter["values"] = filter.apply(lambda x: 0 if x["hour"] <= 6 or x["hour"] >= 20 else x["values"], axis=1)
    outputs_values = filter["values"].values
    return np.array(outputs_values), np.array(targets_values)


def train_pca_lstm(model, train_loader, criterion, optimizer, num_epochs, device):
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
    torch.save(model.state_dict(), f'ramp-stl-ec/models/pca_lstm.pth')
    print('Model Saved')


def train_pca_stl_lstm(model, train_loader, criterion, optimizer, num_epochs, device, stl_label):
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
    torch.save(model.state_dict(), f'ramp-stl-ec/models/pca_stl_lstm_{stl_label}.pth')
    print('Model Saved')


def eval_pca_stl_lstm(model, test_loader, device, labels):
    targets_values = []
    outputs_values = []
    for label in labels:
        model.load_state_dict(torch.load(f'ramp-stl-ec/models/pca_stl_lstm_{label}.pth'))
        model.eval()
        outputs_label = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                outputs_label.extend(outputs.cpu().numpy())
        outputs_values.append(outputs_label)
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