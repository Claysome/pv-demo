# -*- coding: utf-8 -*-

import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
import itertools
import numpy as np
from sklearn.metrics import r2_score
from copy import deepcopy


class Preprocess:
    
    def __init__(self, data_path):
        if data_path.endswith('.csv'):    
            self.data = pd.read_csv(data_path)

    def run(self):
        self.reindex()
        self.filter_valid_data()
        self.remove_outliers()
        self.stl_decomposition()
        self.ramp_event_detection()

    def run_on_decomp(self):
        self.reindex()
        self.filter_valid_data()
        self.remove_outliers()
        data_trend, data_seasonal, data_residual = self.stl_decomposition()

        return data_trend, data_seasonal, data_residual


    def reindex(self):
        date_format = "%d/%m/%Y %H:%M:%S"
        self.data['DATATIME'] = pd.to_datetime(self.data['DATATIME'], format=date_format)
        self.data.set_index('DATATIME', inplace=True)

    def filter_valid_data(self):
        self.data = self.data[self.data['TARGET'] != 0.0]

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

    def ramp_event_detection(self):
        criteria = 0.66
        # 在滑动窗口内，|最大值/最小值-1|的差值大于阈值，则认为是事件
        self.data['MAX'] = self.data['TARGET'].rolling(window=5).max()
        self.data['MIN'] = self.data['TARGET'].rolling(window=5).min()
        self.data['EVENT'] = abs(self.data['MAX']/self.data['MIN'] - 1) > criteria
        # trend
        self.data['TREND_MAX'] = self.data['TREND'].rolling(window=5).max()
        self.data['TREND_MIN'] = self.data['TREND'].rolling(window=5).min()
        self.data['EVENT_TREND'] = abs(self.data['TREND_MAX']/self.data['TREND_MIN'] - 1) > criteria
        # seasonal
        self.data['SEASONAL_MAX'] = self.data['SEASONAL'].rolling(window=5).max()
        self.data['SEASONAL_MIN'] = self.data['SEASONAL'].rolling(window=5).min()
        self.data['EVENT_SEASONAL'] = abs(self.data['SEASONAL_MAX']/self.data['SEASONAL_MIN'] - 1) > criteria
        # residual
        self.data['RESIDUAL_MAX'] = self.data['RESIDUAL'].rolling(window=5).max()
        self.data['RESIDUAL_MIN'] = self.data['RESIDUAL'].rolling(window=5).min()
        self.data['EVENT_RESIDUAL'] = abs(self.data['RESIDUAL_MAX']/self.data['RESIDUAL_MIN'] - 1) > criteria
        # trend / seasonal / residual 取交集
        self.data['EVENT_STL'] = self.data['EVENT_TREND'] & self.data['EVENT_SEASONAL'] & self.data['EVENT_RESIDUAL']

        # plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.data['TARGET'])
        # EVENT 和 EVENT_STL，一个黄色一个绿色
        ax.scatter(self.data[self.data['EVENT']].index, self.data[self.data['EVENT']]['TARGET'], color='yellow')
        ax.scatter(self.data[self.data['EVENT_STL']].index, self.data[self.data['EVENT_STL']]['TARGET'], color='green')

        # 重合率
        intersection = self.data[self.data['EVENT'] & self.data['EVENT_STL']]
        intersection_rate = len(intersection) / len(self.data[self.data['EVENT_STL']==True])
        print(intersection_rate)
        

        plt.show()




d = Preprocess('data/wind_data.csv')
dt, ds, dr = d.run_on_decomp()

print(dt.head())
print(ds.head())
print(dr.head())

print(d.data['TARGET'].sum())
print(dt['TARGET'].sum() + ds['TARGET'].sum() + dr['TARGET'].sum())