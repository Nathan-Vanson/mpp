from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from typing import Optional

class GradientBoostingModel:
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 3):
        """
        Initialise un modèle de Gradient Boosting pour la régression.

        Args:
        - n_estimators (int): Nombre d'estimateurs de boosting à utiliser (par défaut: 100).
        - learning_rate (float): Taux d'apprentissage (par défaut: 0.1).
        - max_depth (int): Profondeur maximale des arbres de décision (par défaut: 3).
        """
        self.model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)


    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Entraîne le modèle de Gradient Boosting sur les données d'entraînement fournies.

        Args:
        - X_train (np.ndarray): Données d'entraînement.
        - y_train (np.ndarray): Étiquettes d'entraînement.

        Returns:
        - None
        """
        self.model.fit(X_train, y_train)


    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        Évalue le modèle de Gradient Boosting sur les données de test en utilisant l'erreur quadratique moyenne (MSE).

        Args:
        - X_test (np.ndarray): Données de test.
        - y_test (np.ndarray): Étiquettes de test.

        Returns:
        - float: Erreur quadratique moyenne (MSE) entre les prédictions et les vérités terrain.
        """
        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        return mse
