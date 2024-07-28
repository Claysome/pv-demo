import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score


class Plotter:

    def __init__(self, path=None) -> None:
        self.path = path
        self.data = pd.DataFrame()

    def _load_data(self):
        self.data["True"] = pd.read_csv(Path(self.path) / 'results1.csv')["True"]
        self.data["svm"] = pd.read_csv(Path(self.path) / 'results1.csv')["Predicted"]
        self.data["lstm"] = pd.read_csv(Path(self.path) / 'results2.csv')["Predicted"]
        self.data["vmd-lstm"] = pd.read_csv(Path(self.path) / 'results3.csv')["Predicted"]
        self.data["vmd-lstm-resid"] = pd.read_csv(Path(self.path) / 'results4.csv')["Predicted"]
        self.data['lstm'] = [float(pred.strip('[]')) for pred in self.data['lstm']]

    def plot(self, start_index, end_index):
        self.data[start_index:end_index].plot(figsize=(16, 10))
        self.metrics(self.data[start_index:end_index])
        plt.show()
        plt.savefig("vmd-lstm-resid/res/plot_day.png")

    def metrics(self, data):
        for col in data.columns[1:]:
            print(f"{col}:")
            print(f"R2 Score: {r2_score(data['True'], data[col])}")
            print(f"Mean Squared Error: {mean_squared_error(data['True'], data[col], squared=False)}")
            print("\n")





if __name__ == '__main__':
    plotter = Plotter("vmd-lstm-resid/res")
    plotter._load_data()
    print(plotter.data.head())
    plotter.plot(-96, -1)