# -*- coding: utf-8 -*-


import pandas as pd


class FeatureEng:

    @staticmethod
    def get_time_feature(df):
        # df['DATETIME'] = pd.to_datetime(df['DATETIME'])
        # df.set_index('DATETIME', inplace=True)
        df['month'] = df.index.month
        df['day'] = df.index.day
        df['hour'] = df.index.hour
        return df