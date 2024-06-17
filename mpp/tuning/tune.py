import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from mpp.models.knn import BaselineKNN
from mpp.models.neural_network import NeuralNetworkModel
from mpp.models.random_forest import RandomForestModel
from mpp.models.gradient_boosting import GradientBoostingModel
from tqdm.notebook import tqdm
from typing import Dict, Union

class HyperparameterTuning:
    def __init__(self, X: np.ndarray, y: pd.Series, model_name: str):
        """
        Initialise un objet pour l'optimisation des hyperparamètres d'un modèle spécifié.

        Args:
        - X (np.ndarray): Données d'entraînement.
        - y (pd.Series): Cibles d'entraînement.
        - model_name (str): Nom du modèle à optimiser ("KNN", "Neural Network", "Random Forest", "Gradient Boosting").
        """
        self.X = X
        self.y = y
        self.model_name = model_name

    def objective(self, trial: optuna.trial.Trial) -> Union[float, np.ndarray]:
        """
        Fonction objectif pour l'optimisation des hyperparamètres.

        Args:
        - trial (optuna.trial.Trial): Instance de l'essai d'optimisation.

        Returns:
        - Union[float, np.ndarray]: Perte de validation finale pour les modèles de régression
        ou score moyen de l'erreur quadratique négative pour les modèles non-neuronaux.
        """
        if self.model_name == "KNN":
            # Pour le modèle KNN, suggère le nombre de voisins et initialise le modèle
            n_neighbors = trial.suggest_int("n_neighbors", 1, 30)
            model = BaselineKNN(n_neighbors=n_neighbors)
            
        elif self.model_name == "Neural Network":
            
            # Pour le modèle de réseau neuronal, suggère les hyperparamètres et initialise le modèle
            lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
            epochs = trial.suggest_int("epochs", 10, 100)
            batch_size = trial.suggest_int("batch_size", 16, 128)
            hidden_layers = trial.suggest_int("hidden_layers", 1, 3)
            hidden_sizes = [trial.suggest_int(f"hidden_size_{i}", 32, 256) for i in range(hidden_layers)]
            patience = trial.suggest_int("patience", 5, 20)
            model = NeuralNetworkModel(input_size=self.X.shape[1], hidden_sizes=hidden_sizes, lr=lr,
                                    epochs=epochs, batch_size=batch_size, patience=patience)
            
        elif self.model_name == "Random Forest":
            
            # Pour le modèle de forêt aléatoire, suggère les hyperparamètres et initialise le modèle
            n_estimators = trial.suggest_int("n_estimators", 50, 200)
            max_depth = trial.suggest_int("max_depth", 1, 32)
            model = RandomForestModel(n_estimators=n_estimators, max_depth=max_depth)
            
        elif self.model_name == "Gradient Boosting":
            
            # Pour le modèle de gradient boosting, suggère les hyperparamètres et initialise le modèle
            n_estimators = trial.suggest_int("n_estimators", 50, 200)
            learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-1)
            max_depth = trial.suggest_int("max_depth", 1, 32)
            model = GradientBoostingModel(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
            
        else:
            # Si le nom du modèle est inconnu, lève une erreur
            raise ValueError("Unknown model name")

        # Séparation des données d'entraînement et de validation
        X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        # Entraînement et évaluation du modèle en fonction du type
        if isinstance(model, NeuralNetworkModel):
            _, validation_loss = model.train(X_train, y_train, X_val, y_val)
            return validation_loss[-1]  # Retourne la dernière perte de validation
        else:
            score = cross_val_score(model.model, X_train, y_train, cv=3, scoring="neg_mean_squared_error")
            return -score.mean()  # Retourne la moyenne des erreurs quadratiques négatives


    def tune(self, n_trials: int = 50) -> Dict[str, Union[int, float]]:
        """
        Méthode principale pour lancer l'optimisation des hyperparamètres.

        Args:
        - n_trials (int): Nombre d'essais d'optimisation à effectuer.

        Returns:
        - Dict[str, Union[int, float]]: Dictionnaire des meilleurs hyperparamètres trouvés par Optuna.
        """
        # Création d'une étude Optuna pour minimiser la fonction objectif
        study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
        
        # Boucle sur le nombre d'essais spécifié
        for _ in tqdm(range(n_trials), desc=f"Hyperparameter Tuning ({self.model_name})"):
            study.optimize(self.objective, n_trials=1)  # Optimisation d'un seul essai à chaque itération
        
        # Affichage des meilleurs hyperparamètres trouvés
        print(f"Best trial for {self.model_name}: {study.best_trial.params}")
        
        # Retourne les meilleurs hyperparamètres sous forme de dictionnaire
        return study.best_trial.params
