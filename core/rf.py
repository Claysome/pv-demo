import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV


class RF:

    def __init__(self, data):
        self.data = data

    def set_model(self, n_estimators=100, random_state=42):
        # Best parameters found: {'bootstrap': True, 'max_depth': 30, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
        self._model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        # self._model = RandomForestRegressor(n_estimators=200, max_depth=30, min_samples_leaf=1, min_samples_split=2, bootstrap=True, random_state=random_state)

    @property
    def model(self):
        return self._model
    
    def train(self, X_train, y_train):
        self._model.fit(X_train, y_train)

    def predict(self, X_test):
        return self._model.predict(X_test)
    
    def feature_importance(self, feature_names):
        importances = self._model.feature_importances_
        indices = importances.argsort()[::-1]
        top_features = [(feature_names[i], importances[i]) for i in indices[:10]]
        
        print("Top 10 important features:")
        for feature, importance in top_features:
            print(f"{feature}: {importance:.4f}")
        
        return [feature_names[i] for i in indices[:10]]

    def hyperparameter_search(self, X_train, y_train):
        # 超参数搜索的参数网格
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }
        # 网格搜索
        grid_search = GridSearchCV(estimator=self._model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)
        
        print("Best parameters found:", grid_search.best_params_)
        self._model = grid_search.best_estimator_
    
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
        plt.savefig('images/rf_prediction.png')
        plt.show()