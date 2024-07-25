import torch
import torch.nn as nn
import numpy as np
import pandas as pd


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
    for epoch in range(num_epochs):
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
    # outputs_values = np.array([0 if x < 3 else x for x in outputs_values.flatten()]).reshape(-1, 1)
    filter = pd.DataFrame({"timestamp": test_loader.dataset.data.index[:len(outputs_values)], "values": outputs_values.flatten()})
    filter.set_index("timestamp", inplace=True)
    filter["hour"] = filter.index.dt.hour
    filter["values"] = filter.apply(lambda x: 0 if x["hour"] <= 6 or x["hour"] >= 20 else x["values"], axis=1)
    outputs_values = filter["values"].values
        
    for inputs, targets in test_loader:
        targets_values.extend(targets.cpu().numpy())
    targets_values = np.array(targets_values)
    return outputs_values, targets_values


def train_lstm(model, train_loader, criterion, optimizer, num_epochs, device):
    for epoch in range(num_epochs):
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
    total_loss = 0
    outputs_values, targets_values = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            outputs_values.extend(outputs.cpu().numpy())
            targets_values.extend(targets.cpu().numpy())
            total_loss += loss.item()
    print(f'Test Loss: {total_loss/len(test_loader):.4f}')
    return outputs_values, targets_values


