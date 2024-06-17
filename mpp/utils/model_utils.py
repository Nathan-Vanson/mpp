from mpp.models.knn import BaselineKNN
from mpp.models.neural_network import NeuralNetworkModel
from mpp.models.random_forest import RandomForestModel
from mpp.models.gradient_boosting import GradientBoostingModel
import numpy as np
from typing import Dict, Tuple, List, Union

def train_and_evaluate_models(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> Tuple[Dict[str, float], Dict[str, List[float]], Dict[str, List[float]]]:
    """
    Entraîne et évalue plusieurs modèles de machine learning sur les données fournies.

    Args:
    - X_train (np.ndarray): Données d'entraînement.
    - X_test (np.ndarray): Données de test.
    - y_train (np.ndarray): Étiquettes d'entraînement.
    - y_test (np.ndarray): Étiquettes de test.

    Returns:
    - Tuple[Dict[str, float], Dict[str, List[float]], Dict[str, List[float]]]:
        - Un dictionnaire contenant les résultats des modèles évalués (MSE pour chaque modèle).
        - Un dictionnaire contenant les historiques de perte d'entraînement pour chaque modèle.
        - Un dictionnaire contenant les historiques de perte de validation pour chaque modèle.
    """
    models = {
        "KNN": BaselineKNN(),
        "Neural Network": NeuralNetworkModel(input_size=X_train.shape[1], hidden_sizes=[64, 32], epochs=50),
        "Random Forest": RandomForestModel(),
        "Gradient Boosting": GradientBoostingModel()
    }
    results: Dict[str, float] = {}
    training_losses: Dict[str, List[float]] = {}
    validation_losses: Dict[str, List[float]] = {}

    #  Modèles d'entraînement et d'évaluation
    for name, model in models.items():
        print(f"Training {name}...")
        if isinstance(model, NeuralNetworkModel):
            training_loss, validation_loss = model.train(X_train, y_train, X_test, y_test)
            training_losses[name] = training_loss
            validation_losses[name] = validation_loss
        else:
            model.train(X_train, y_train)
        mse = model.evaluate(X_test, y_test)
        results[name] = mse
        print(f"{name} Model MSE: {mse}")

    return results, training_losses, validation_losses
