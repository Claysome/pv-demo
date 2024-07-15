import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from core.data_loader import DataLoader


class EDA:

    def __init__(self, file_path):
        self.data_loader = DataLoader(file_path)
        self.data = self.data_loader.get_data()
        self.__post_init__()

    def __post_init__(self):
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        self.data['month'] = self.data['timestamp'].dt.month
        self.data['day'] = self.data['timestamp'].dt.day
        self.data['hour'] = self.data['timestamp'].dt.hour

    def plot_corr(self):
        plt.figure(figsize=(12, 10))
        corr = self.data.drop('timestamp', axis=1).corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=~mask, annot=True, cmap='viridis', square=True)
        plt.title('Correlation Matrix')
        plt.savefig('images/corr_matrix.png')

    def plot_monthly(self):
        fig, axes = plt.subplots(3, 1, figsize=(9, 9))
        data = self.data.set_index('timestamp')
        months = ['January', 'February', 'March']

        for i, (ax, month) in enumerate(zip(axes, range(1, 4))):
            month_data = data[data['month'] == month]
            days = month_data.index.day.unique()

            for day in days:
                day_data = month_data[month_data.index.day == day]
                times = day_data.index.time
                times_seconds = [(t.hour * 3600 + t.minute * 60 + t.second) for t in times]  # Convert to seconds
                ax.plot(times_seconds, day_data['Active_Power'], label=f'Day {day}')

            ax.set_title(f'{months[i]} Daily Power Curves')
            ax.set_xlabel('Time (seconds since start of day)')
            ax.set_ylabel('Active Power')
            ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

        plt.tight_layout()
        plt.savefig('images/monthly_power_curves.png')

    

    


if __name__ == '__main__':
    eda = EDA('data/pv.csv')
    print(eda.plot_monthly())