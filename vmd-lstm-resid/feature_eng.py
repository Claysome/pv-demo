import pandas as pd


class FeatureEng:

    @staticmethod
    def get_time_feature(df):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df['month'] = df.index.month
        df['day'] = df.index.day
        df['hour'] = df.index.hour
        return df