import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd

from preprocess import Preprocess
from dataset import PVData


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
    torch.save(model.state_dict(), 'models/lstm.pth')
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


if __name__ == '__main__':
    file_path = 'data/cleaned_pv.csv'
    imf, resid, test = Preprocess(file_path).get_rep_data()
    train_dataset = PVData(imf)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_size = imf.shape[1] - 1
    hidden_size1 = 100
    hidden_size2 = 200
    num_layers = 1
    output_size = 1

    model = LSTMModel(input_size, hidden_size1, hidden_size2, num_layers, output_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # # 训练模型
    # num_epochs = 15
    # train_lstm(model, train_loader, criterion, optimizer, num_epochs, device)

    resid_dataset = PVData(resid)
    test_loader = DataLoader(resid_dataset, batch_size=32, shuffle=False)
    model.load_state_dict(torch.load('models/lstm.pth'))
    outputs_values, targets_values = eval_lstm(model, test_loader, criterion)

    # import matplotlib.pyplot as plt
    # plt.plot(targets_values, label='True')
    # plt.plot(outputs_values, label='Predicted')
    # plt.legend()
    # plt.show()
    from sklearn.metrics import r2_score
    print(r2_score(targets_values, outputs_values))