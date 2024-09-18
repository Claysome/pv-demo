# -*- coding: utf-8 -*-

import pandas as pd


def clean_data(data_path):
    if data_path.endswith('.csv'):    
        d = pd.read_csv(data_path)
    d.drop(['PREPOWER', 'ROUND(A.WS,1)', 'ROUND(A.POWER,0)'], axis=1, inplace=True)
    d.rename(columns={'YD15': 'TARGET'}, inplace=True)
    d.dropna(inplace=True)
    d.drop_duplicates(subset='DATATIME', inplace=True)
    return d
