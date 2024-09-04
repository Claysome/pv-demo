import pandas as pd
from feature_eng import FeatureEng
from sklearn.decomposition import PCA
from statsmodels.tsa.seasonal import STL


class Preprocess:
    train_sig_size = 0.4
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

    def __pca__(self):
        pca = PCA(n_components=3)
        if self.data.isnull().sum().sum() > 0:
            self.data.fillna(method='ffill', inplace=True)
        pca_result = pca.fit_transform(self.data.drop(columns=["timestamp", "Active_Power"]))
        self.data["pca1"] = pca_result[:, 0]
        self.data["pca2"] = pca_result[:, 1]
        self.data["pca3"] = pca_result[:, 2]
        self.data.drop(columns=self.data.columns.difference(["timestamp", "Active_Power", "pca1", "pca2", "pca3"]), inplace=True)

    def __pca_n_components__(self):
        if self.data.isnull().sum().sum() > 0:
            self.data.fillna(method='ffill', inplace=True)
        # 通过PCA的方差贡献率来确定n_components
        pca = PCA()
        pca.fit(self.data.drop(columns=["timestamp", "Active_Power"]))
        var_ratio = pca.explained_variance_ratio_
        # 累计贡献率画图
        import matplotlib.pyplot as plt
        plt.plot(range(1, len(var_ratio) + 1), var_ratio.cumsum())
        plt.show()

    def __stl__(self, label=None):
        # 内部循环迭代次数为1，外部循环迭代次数为0 、周期序列平滑参数为96 、低通滤波器的平滑参数为97，145
        stl = STL(self.data["Active_Power"], period=96, robust=True)
        res = stl.fit()
        if label == "trend":
            self.data["Active_Power"] = res.trend
        if label == "seasonal":
            self.data["Active_Power"] = res.seasonal
        if label == "resid":
            self.data["Active_Power"] = res.resid
        # 画图
        # import matplotlib.pyplot as plt
        # res.plot()
        # plt.show()

    def __normalize__(self, exclude=["Active_Power"]):
        for col in self.data.columns:
            if col not in exclude:
                self.data[col] = (self.data[col] - self.data[col].min()) / (self.data[col].max() - self.data[col].min())

    def __remove_outliers__(self, min_scale=5.0):
        day_power = self.data.resample('D').sum()
        valid_days = day_power[day_power["Active_Power"] > min_scale].index
        self.data = self.data[self.data.index.normalize().isin(valid_days)]

    def get_data(self, pca=False, stl=False, label=None):
        self.__aggregate__()
        if pca:
            # self.__pca_n_components__()
            self.__pca__()
        if stl:
            self.__stl__(label)
        self.__feature_eng__()
        self.__remove_outliers__()
        self.__normalize__()
        self.data.dropna(inplace=True)
        # split data
        train_sig_size = int(len(self.data) * self.train_sig_size)
        train_sig_data = self.data.iloc[:train_sig_size]
        train_resid_data = self.data.iloc[train_sig_size:train_sig_size + int(len(self.data) * self.train_resid_size)]
        test_data = self.data.iloc[train_sig_size + int(len(self.data) * self.train_resid_size):]
        return train_sig_data, train_resid_data, test_data