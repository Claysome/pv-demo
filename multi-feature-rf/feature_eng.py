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
        # time series
        df['sum_roll_radiation_3d'] = df['Global_Horizontal_Radiation'].rolling(window=5*12*3).sum()
        df['mean_roll_temp_3h'] = df['Weather_Temperature_Celsius'].rolling(window=3).mean()
        df['mean_roll_radiation_12h'] = df['Global_Horizontal_Radiation'].rolling(window=12).mean()
        # ma std
        df['std_roll_humidity_6h'] = df['Weather_Relative_Humidity'].rolling(window=6).std()
        # extreme
        df['max_roll_radiation_24h'] = df['Global_Horizontal_Radiation'].rolling(window=24).max()
        df['min_roll_radiation_24h'] = df['Global_Horizontal_Radiation'].rolling(window=24).min()
        # median
        df['median_roll_temp_12h'] = df['Weather_Temperature_Celsius'].rolling(window=12).median()
        # weather
        df['humidity_change_rate'] = df['Weather_Relative_Humidity'].diff()
        df['wind_vector'] = df['Wind_Direction'] * df['Weather_Relative_Humidity']

        return df
    
