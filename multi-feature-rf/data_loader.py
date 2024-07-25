import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import pandas as pd


class DataLoader:

    def __init__(self, file_path):
        self.file_path  = file_path
        self.__post_init__()

    def __post_init__(self):
        match self.file_path:
            case _ if self.file_path.endswith(".csv"):
                self.data = pd.read_csv(self.file_path)
            case _ if self.file_path.endswith(".parquet"):
                self.data = pd.read_parquet(self.file_path)

    def get_data(self):
        return self.data
    
    def get_data_shape(self):
        return self.data.shape
    
    def get_data_columns(self):
        return self.data.columns
    

if __name__ == "__main__":
    d = DataLoader("data/solar.csv")
    print(d.get_data())
    print(d.get_data_shape())
    print(d.get_data_columns())