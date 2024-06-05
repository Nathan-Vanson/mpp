import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from mpp.models.knn import BaselineKNN
from mpp.models.neural_network import NeuralNetworkModel
from mpp.models.random_forest import RandomForestModel
from mpp.models.gradient_boosting import GradientBoostingModel
from tqdm.notebook import tqdm

class HyperparameterTuning:
    def __init__(self, X, y, model_name):
        self.X = X
        self.y = y
        self.model_name = model_name

    def objective(self, trial):
        if self.model_name == "KNN":
            n_neighbors = trial.suggest_int("n_neighbors", 1, 30)
            model = BaselineKNN(n_neighbors=n_neighbors)
        elif self.model_name == "Neural Network":
            lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
            epochs = trial.suggest_int("epochs", 10, 100)
            batch_size = trial.suggest_int("batch_size", 16, 128)
            hidden_layers = trial.suggest_int("hidden_layers", 1, 3)
            hidden_sizes = [trial.suggest_int(f"hidden_size_{i}", 32, 256) for i in range(hidden_layers)]
            patience = trial.suggest_int("patience", 5, 20)
            model = NeuralNetworkModel(input_size=self.X.shape[1], hidden_sizes=hidden_sizes, lr=lr, epochs=epochs, batch_size=batch_size, patience=patience)
        elif self.model_name == "Random Forest":
            n_estimators = trial.suggest_int("n_estimators", 50, 200)
            max_depth = trial.suggest_int("max_depth", 1, 32)
            model = RandomForestModel(n_estimators=n_estimators, max_depth=max_depth)
        elif self.model_name == "Gradient Boosting":
            n_estimators = trial.suggest_int("n_estimators", 50, 200)
            learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-1)
            max_depth = trial.suggest_int("max_depth", 1, 32)
            model = GradientBoostingModel(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
        else:
            raise ValueError("Unknown model name")

        X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        if isinstance(model, NeuralNetworkModel):
            _, validation_loss = model.train(X_train, y_train, X_val, y_val)
            return validation_loss[-1]  # Return the last validation loss
        else:
            score = cross_val_score(model.model, X_train, y_train, cv=3, scoring="neg_mean_squared_error")
            return -score.mean()  # Return the average of negative mean squared error

    def tune(self, n_trials=50):
        study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
        for _ in tqdm(range(n_trials), desc=f"Hyperparameter Tuning ({self.model_name})"):
            study.optimize(self.objective, n_trials=1)
        print(f"Best trial for {self.model_name}: {study.best_trial.params}")
        return study.best_trial.params
