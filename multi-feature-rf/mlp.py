import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error, mean_absolute_error

class MLP:

    def __init__(self, data):
        self.data = data

    def set_model(self, hidden_layer_sizes=(100,), activation='relu', solver='adam', random_state=42):
        self._model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, random_state=random_state)

    @property
    def model(self):
        return self._model
    
    def train(self, X_train, y_train):
        self._model.fit(X_train, y_train)

    def predict(self, X_test):
        return self._model.predict(X_test)
    
    def evaluate(self, y_test, y_pred):
        r2 = r2_score(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        print(f'R2: {r2}')
        print(f'RMSE: {rmse}')
        print(f'MAE: {mae}')
        return r2, rmse, mae

    @classmethod
    def plot(cls, y_test, y_pred):
        import matplotlib.pyplot as plt
        plt.plot(y_test.values, label='actual')
        plt.plot(y_pred, label='prediction')
        plt.legend()
        plt.savefig('images/mlp_prediction.png')
        plt.show()
