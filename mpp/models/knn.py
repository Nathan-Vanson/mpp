from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

class BaselineKNN:
    def __init__(self, n_neighbors=5):
        self.model = KNeighborsRegressor(n_neighbors=n_neighbors)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        return mse