# -*- coding: utf-8 -*-

import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
import itertools
import numpy as np
from sklearn.metrics import r2_score
from copy import deepcopy
import seaborn as sns

from feature_eng import FeatureEng


class Preprocess:
    
    def __init__(self, data_path):
        if data_path.endswith('.csv'):    
            self.data = pd.read_csv(data_path)

    def run(self):
        self.reindex()
        self.filter_valid_data()
        self.aggregation()
        self.remove_outliers()
        self.feature_eng()
        self.normalize()
        # self.corr_analysis()

        data_trend, data_seasonal, data_residual = self.stl_decomposition()
        data_org = deepcopy(self.data)

        return data_org, data_trend, data_seasonal, data_residual

    def reindex(self):
        date_format = "%Y-%m-%d %H:%M:%S"
        self.data['DATETIME'] = pd.to_datetime(self.data['DATETIME'], format=date_format)
        self.data.set_index('DATETIME', inplace=True)

    def filter_valid_data(self):
        self.data = self.data[self.data['TARGET'] != 0.0]

    def aggregation(self, freq='15min'):
        agg_rules = {
            'TARGET': 'sum',
            'Weather_Temperature_Celsius': 'mean',
            'Weather_Relative_Humidity': 'mean',
            'Global_Horizontal_Radiation': 'mean',
            'Diffuse_Horizontal_Radiation': 'mean',
            'Wind_Direction': 'mean',
            'Weather_Daily_Rainfall': 'sum',
        }
        self.data = self.data.resample(freq).agg(agg_rules)
        self.data.dropna(inplace=True)

    def remove_outliers(self):
        std = self.data['TARGET'].std()
        mean = self.data['TARGET'].mean()
        self.data = self.data[(self.data['TARGET'] < mean + 3*std) & (self.data['TARGET'] > mean - 3*std)]

    def acf(self):
        target = self.data['TARGET']
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_acf(target, ax=ax)
        plt.show()

    def stl_decomposition(self, period=9, seasonal=5, trend=13):
        stl = STL(self.data['TARGET'], period=period, seasonal=seasonal, trend=trend)
        res = stl.fit()
        # fig = res.plot()
        # plt.show()
        # self.data['TREND'] = res.trend
        # self.data['SEASONAL'] = res.seasonal
        # self.data['RESIDUAL'] = res.resid
        data_trend = deepcopy(self.data)
        data_seasonal = deepcopy(self.data)
        data_residual = deepcopy(self.data)
        data_trend['TARGET'] = res.trend
        data_seasonal['TARGET'] = res.seasonal
        data_residual['TARGET'] = res.resid
        return data_trend, data_seasonal, data_residual

    def search_optim_stl_decomposition(self):
        periods  = range(5, 12)
        seasonals = [5, 13, 25, 97]
        trends = [7, 13, 25, 97]
        for i in trends:
            if i % 2 == 1 and i > max(periods):
                pass
            else:
                trends.remove(i)

        best_r2 = -np.inf
        best_params = None

        for p, s, t in itertools.product(periods , seasonals, trends):
            stl = STL(self.data['TARGET'], period=p, seasonal=s, trend=t)
            res = stl.fit()
            r2 = r2_score(self.data['TARGET'], res.trend + res.seasonal)
            
            if r2 > best_r2:
                best_r2 = r2
                best_params = {'period': p, 'seasonal': s, 'trend': t}
            print(f"Period: {p}, Seasonal: {s}, Trend: {t} ..., Best R2: {best_r2}, Best parameters: {best_params}")
        print(f"Best R2: {best_r2}")
        print(f"Best parameters: {best_params}")

    def feature_eng(self):
        self.data = FeatureEng.get_time_feature(self.data)

    def normalize(self, exclude=["TARGET"]):
        for col in self.data.columns:
            if col not in exclude:
                self.data[col] = (self.data[col] - self.data[col].min()) / (self.data[col].max() - self.data[col].min())

    def corr_analysis(self, target='TARGET'):
        sns.heatmap(self.data.corr(), annot=True)
        plt.show()
        
