import sys
import json
import warnings
from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")

from src.data_loader import DataLoader
from src.feature_eng import FeatureEng
from src.pca_application import PCAApplication
from src.vmd_optimizer import VmdOptimizer
from utils import TimeFrame


class Pipeline:

    def __init__(self, config_path = "config.json"):
        self.config = self.load_config(config_path)

    def load_config(self, file_path=None):
        if file_path.endswith(".json"):
            with open(file_path, "r") as f:
                config = json.load(f)
        return config

    def run(self):
        self.train_data = DataLoader(data_path=self.config["data"].get("path")).load_data("2019-01-01", "2019-4-30")
        # self.test_data = DataLoader(data_path=self.config["data"].get("path")).load_data("2019-05-01", "2019-05-31")
        # print(f"Train data shape: {self.train_data.shape}, Test data shape: {self.test_data.shape}")
        self.train_x, self.train_y = DataLoader.create_window(self.train_data)
        self.fit_model(self.train_x)
        
    def fit_model(self, train_data):
        train_data = train_data[:3]
        label = np.array([d["Active_Power"].values for d in train_data]).flatten()  # 将生成器转换为数组
        processd_data = []
        for i, d in tqdm(enumerate(train_data)):
            d.drop(columns=["Active_Power"], inplace=True)
            d = self.feature_eng(d)
            processd_data.append(d)
        processd_data = pd.concat(processd_data, axis=0)
        
        # 训练模型
        model = XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=5, learning_rate=0.1)
        model.fit(processd_data, label)
        
        # 保存模型
        model_path = Path(__file__).parent.parent / "models" / "xgb_model.json"
        model.save_model(str(model_path))
        print("Model saved successfully.")


    def feature_eng(self, data):
        stats_data = FeatureEng.statistical_features(data).drop(columns=data.columns)
        advanced_stats_data = FeatureEng.advanced_stats_features(data).drop(columns=data.columns)
        freq_data = FeatureEng.frequency_features(data).drop(columns=data.columns)
        factor_data = FeatureEng.factor_features(data).drop(columns=data.columns)
        time_data = FeatureEng.time_features(data).drop(columns=data.columns)

        eng_data = pd.concat([data, stats_data, advanced_stats_data, freq_data, factor_data, time_data], axis=1)
        return eng_data


if __name__ == '__main__':
    pipe = Pipeline("config.json").run()
