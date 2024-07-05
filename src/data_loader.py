import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import pandas as pd
from utils import TimeFrame


class DataLoader:

    def __init__(self, data_path = Path(__file__).parent.parent / "data" / "pv_data.parquet"):
        self.data_path = data_path

    def load_data(self, start_date=None, end_date=None):
        df = pd.read_parquet(self.data_path).drop(columns=["Active_Energy_Delivered_Received", "Current_Phase_Average", "Performance_Ratio"])
        df["date_time"] = pd.to_datetime(df["timestamp"])
        df.set_index("date_time", inplace=True)
        if start_date and end_date:
            df = df.loc[start_date:end_date]
        df.drop(columns=["timestamp"], inplace=True)
        return df
    
    @staticmethod
    def create_window(data, window_size_hourly = 7 * 24, step_size = 48, freq = TimeFrame.MINUTE_5):
        X, y = [], []
        window_size = window_size_hourly * freq.value
        for i in range(0, len(data) - window_size - step_size + 1, step_size):
            X.append(data[i:i + window_size])
            y.append(data[i + window_size:i + window_size + step_size])
        return X, y
    



if __name__ == "__main__":

    dl = DataLoader()
    df = dl.load_data("2015-01-01", "2015-5-31")
    from src.pca_application import PCAApplication
    df.dropna(inplace=True)
    PCAApplication.plot_explained_variance(df)
    # data = dl.create_window(df)
    # print(data)