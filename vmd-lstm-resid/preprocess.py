import pandas as pd
from feature_eng import FeatureEng
from vmd import VMDecomposition


class Preprocess:
    train_imf_size=0.4
    train_resid_size = 0.4
    test_size = 0.2
    
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.__post_init__()

    def __post_init__(self):
        if "timestamp" in self.data.columns:
            self.data["timestamp"] = pd.to_datetime(self.data["timestamp"])
            self.data.set_index("timestamp", inplace=True)

    def __aggregate__(self, freq="15min"):
        agg_rules = {
            'Active_Power': 'sum',
            'Weather_Temperature_Celsius': 'mean',
            'Weather_Relative_Humidity': 'mean',
            'Global_Horizontal_Radiation': 'mean',
            'Diffuse_Horizontal_Radiation': 'mean',
            'Wind_Direction': 'mean',
            'Weather_Daily_Rainfall': 'sum',
        }
        self.data = self.data.resample(freq).agg(agg_rules)
        self.data.reset_index(inplace=True)

    def __feature_eng__(self):
        self.data = FeatureEng.get_time_feature(self.data)

    def __vmd_decompose__(self, target_group):
        if target_group is None:
            print("No VMD decomposition")
            return
        print(f"Get Imf group: {target_group}")
        imfs, _ = VMDecomposition.get_imfs(self.data["Active_Power"].values, k=6)
        VMDecomposition.plot_imfs(imfs)
        self.data["Active_Power"] = imfs[target_group-1]

    def __normalize__(self, exclude=["Active_Power"]):
        for col in self.data.columns:
            if col not in exclude:
                self.data[col] = (self.data[col] - self.data[col].min()) / (self.data[col].max() - self.data[col].min())

    def __remove_outliers__(self, min_scale=10.0):
        day_power = self.data.resample('D').sum()
        valid_days = day_power[day_power["Active_Power"] > min_scale].index
        self.data = self.data[self.data.index.normalize().isin(valid_days)]

    def get_data(self, target_group):
        self.__aggregate__()
        self.__feature_eng__()
        self.__remove_outliers__()
        self.__vmd_decompose__(target_group)
        self.__normalize__()
        return self.data

    def get_rep_data(self, target_group=None):
        data = self.get_data(target_group)
        data.dropna(inplace=True)
        # 生成训练集1
        train_imf_size = int(len(data) * self.train_imf_size)
        train_imf_data = data.iloc[:train_imf_size]
        # 生成训练集2
        train_resid_size = int(len(data) * self.train_resid_size)
        train_resid_data = data.iloc[train_imf_size:train_imf_size+train_resid_size]
        # 生成测试集
        test_data = data.iloc[train_imf_size+train_resid_size:]
        print(f"train_imf_data: {len(train_imf_data)}, train_resid_data: {len(train_resid_data)}, test_data: {len(test_data)}")
        return train_imf_data, train_resid_data, test_data
