from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import numpy as np
from typing import Optional

class BaselineKNN:
    def __init__(self, n_neighbors: int = 5):
        """
        Initialise un modèle de k-plus proches voisins (KNN) pour la régression.

        Args:
        - n_neighbors (int): Nombre de voisins à utiliser pour la prédiction (par défaut: 5).
        """
        self.model = KNeighborsRegressor(n_neighbors=n_neighbors)


    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Entraîne le modèle KNN sur les données d'entraînement fournies.

        Args:
        - X_train (np.ndarray): Données d'entraînement.
        - y_train (np.ndarray): Étiquettes d'entraînement.

        Returns:
        - None
        """
        self.model.fit(X_train, y_train)


    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        Évalue le modèle KNN sur les données de test en utilisant l'erreur quadratique moyenne (MSE).

        Args:
        - X_test (np.ndarray): Données de test.
        - y_test (np.ndarray): Étiquettes de test.

        Returns:
        - float: Erreur quadratique moyenne (MSE) entre les prédictions et les vérités terrain.
        - float: Racine de l'erreur quadratique moyenne (RMSE)
        - float: Mean Absolute Error (MAE) 
        - float: Mean Absolute Percentage Error (MAPE)
        """
        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)

        mae = mean_absolute_error(y_test, predictions)
        mape = mean_absolute_percentage_error(y_test, predictions)
        return mse, rmse, mae, mape
