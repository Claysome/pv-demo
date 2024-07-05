import numpy as np
import pandas as pd


class FeatureEng:

    @staticmethod
    def time_features(data):
        if isinstance(data.index, pd.DatetimeIndex):
            data["minute"] = data.index.minute
            data["hour"] = data.index.hour
            data["day_of_week"] = data.index.dayofweek
            data["day_of_month"] = data.index.day
            data["month"] = data.index.month
            data["year"] = data.index.year
        return data

    @staticmethod
    def statistical_features(data):
        new_features = {}
        for f in data.columns:
            new_features[f"{f}_mean"] = data[f].mean()
            new_features[f"{f}_std"] = data[f].std()
            new_features[f"{f}_var"] = data[f].var()
            new_features[f"{f}_min"] = data[f].min()
            new_features[f"{f}_max"] = data[f].max()
            new_features[f"{f}_median"] = data[f].median()
            new_features[f"{f}_sum"] = data[f].sum()
        return pd.concat([data, pd.DataFrame(new_features, index=data.index)], axis=1)
    
    @staticmethod
    def advanced_stats_features(data):
        new_features = {}
        for f in data.columns:
            new_features[f"{f}_skew"] = data[f].skew()
            new_features[f"{f}_kurtosis"] = data[f].kurtosis()
            new_features[f"{f}_rms"] = np.sqrt(np.mean(data[f]**2))
            new_features[f"{f}_sk_mean"] = data[f].apply(lambda x: np.mean(np.abs(x - np.mean(x))**3)**(1/3))
            new_features[f"{f}_sk_std"] = data[f].apply(lambda x: np.std(np.abs(x - np.mean(x))**3)**(1/3))
            new_features[f"{f}_sk_skew"] = data[f].apply(lambda x: np.mean((x - np.mean(x))**3) / np.std((x - np.mean(x))**3) if np.std((x - np.mean(x))**3) != 0 else 0)
            new_features[f"{f}_sk_kurtosis"] = data[f].apply(lambda x: np.mean((x - np.mean(x))**4) / np.std((x - np.mean(x))**4) if np.std((x - np.mean(x))**4) != 0 else 0)
        return pd.concat([data, pd.DataFrame(new_features, index=data.index)], axis=1)
    
    @staticmethod
    def frequency_features(data):
        new_features = {}
        for f in data.columns:
            fft_result = np.fft.fft(data[f].values)
            new_features[f"{f}_FC"] = fft_result.real
            new_features[f"{f}_MSF"] = fft_result.imag
            new_features[f"{f}_RMSF"] = np.sqrt(new_features[f"{f}_FC"]**2 + new_features[f"{f}_MSF"]**2)
            new_features[f"{f}_VF"] = np.fft.fftfreq(len(data[f].values))
            new_features[f"{f}_RVF"] = np.std(new_features[f"{f}_VF"])
        return pd.concat([data, pd.DataFrame(new_features, index=data.index)], axis=1)
            
    @staticmethod
    def factor_features(data):
        new_features = {}
        for f in data.columns:
            new_features[f"{f}_waveform_factor"] = np.abs(data[f].mean()) / np.sqrt(np.mean(data[f]**2))
            new_features[f"{f}_peak_factor"] = data[f].max() / np.sqrt(np.mean(data[f]**2))
            new_features[f"{f}_impulse_factor"] = data[f].max() / np.abs(data[f].mean())
            new_features[f"{f}_clearance_factor"] = data[f].max() / np.abs(data[f].min())
        return pd.concat([data, pd.DataFrame(new_features, index=data.index)], axis=1)
