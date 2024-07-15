from sklearn.model_selection import train_test_split


class Preprocess:

    @classmethod
    def train_test_split(cls, data, target_name, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(data, data[target_name], test_size=test_size, random_state=random_state)
        X_train = X_train.drop(columns=[target_name])
        X_test = X_test.drop(columns=[target_name])
        return X_train, X_test, y_train, y_test

    @classmethod
    def scale(cls, data, exclude=['timestamp']):
        pass

    


if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from core.data_loader import DataLoader
    d = DataLoader('data/pv.csv')
    data = d.get_data()
    X_train, X_test, y_train, y_test = Preprocess.train_test_split(data, 'Active_Power')
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
