from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

class GradientBoostingModel:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        return mse