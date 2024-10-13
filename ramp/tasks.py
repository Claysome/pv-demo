# -*- coding: utf-8 -*-

import torch
from tqdm import tqdm
import numpy as np
import pandas as pd


class PredictionTasks:

    @staticmethod
    def split_train_test(data, train_size=0.8):
        train_size = int(len(data) * train_size)
        return data[:train_size], data[train_size:]
    
    @staticmethod
    def train_lstm(model, train_loader, criterion, optimizer, num_epochs, device, model_name):
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
        torch.save(model.state_dict(), f'ramp/models/{model_name}.pth')
        print(f'Model Saved: {model_name}')

    @staticmethod
    def eval_lstm(model, test_loader, device):
        model.eval()
        outputs_values, targets_values = [], []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                outputs_values.extend(outputs.cpu().numpy())
                targets_values.extend(targets.cpu().numpy())
        # filter = pd.DataFrame({"timestamp": test_loader.dataset.data.index[-len(outputs_values):], "values": outputs_values})
        # filter.set_index("timestamp", inplace=True)
        # filter["hour"] = filter.index.hour
        # filter["values"] = filter.apply(lambda x: 0 if x["hour"] <= 6 or x["hour"] >= 20 else x["values"], axis=1)
        # outputs_values = filter["values"].values
        data = pd.DataFrame({"outputs": outputs_values, "targets": targets_values})
        return data["outputs"].values, data["targets"].values