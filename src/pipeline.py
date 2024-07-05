import sys
import json
import pandas as pd
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import DataLoader
from src.feature_eng import FeatureEng
from src.pca_application import PCAApplication


class Pipeline:

    def __init__(self, config_path = "config.json"):
        self.config = self.load_config(config_path)

    def load_config(self, file_path=None):
        if file_path.endswith(".json"):
            with open(file_path, "r") as f:
                config = json.load(f)
        return config

    def run(self):
        self.data = DataLoader(data_path=self.config["data"].get("path")).load_data("2019-01-01", "2019-5-31")
        print(self.data.info())
        # self.eng_data = self.feature_eng(self.data)

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
