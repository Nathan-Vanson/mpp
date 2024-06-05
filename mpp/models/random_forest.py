from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

class RandomForestModel:
    def __init__(self, n_estimators=100, max_depth=5):
        self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=5)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        return mse