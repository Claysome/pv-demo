import pandas as pd


class FeatureEng:

    @staticmethod
    def get_time_feature(df):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['month'] = df['timestamp'].dt.month
        df['day'] = df['timestamp'].dt.day
        df['hour'] = df['timestamp'].dt.hour

        return df
    
    @staticmethod
    def get_statistical_feature(df):
        df['sum_roll_radiation'] = df['Global_Horizontal_Radiation'].rolling(window=5*12*3).sum()

        return df
    
