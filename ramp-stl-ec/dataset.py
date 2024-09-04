import torch
from torch.utils.data import Dataset

from preprocess import Preprocess


class PVData(Dataset):
    
    def __init__(self, data, input_size=24, target_col="Active_Power"):
        self.data = data
        self.input_size = input_size
        self.target_col = target_col
        self.data_size = len(data)
        # sort by timestamp
        self.data.sort_index(inplace=True)

    def __len__(self):
        return self.data_size - self.input_size
    
    def __getitem__(self, idx):
        end_idx = idx + self.input_size

        input_data = self.data.iloc[idx:end_idx].drop(columns=[self.target_col]).values
        target_data = self.data.iloc[end_idx][self.target_col]
        # to tensor
        input_data = torch.tensor(input_data, dtype=torch.float32)
        target_data = torch.tensor(target_data, dtype=torch.float32).unsqueeze(0)

        return input_data, target_data
    

if __name__ == '__main__':
    processor = Preprocess('data/cleaned_pv.csv')
    train_sig_data, train_resid_data, test_data = processor.get_data(pca=True)
    print(len(train_sig_data), len(train_resid_data), len(test_data))