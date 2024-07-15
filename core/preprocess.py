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
        for col in data.columns:
            if col not in exclude:
                data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())
        return data

