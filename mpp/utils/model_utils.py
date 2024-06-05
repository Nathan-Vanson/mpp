from mpp.models.knn import BaselineKNN
from mpp.models.neural_network import NeuralNetworkModel
from mpp.models.random_forest import RandomForestModel
from mpp.models.gradient_boosting import GradientBoostingModel

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        "KNN": BaselineKNN(),
        "Neural Network": NeuralNetworkModel(input_size=X_train.shape[1], hidden_sizes=[64, 32], epochs=50),
        "Random Forest": RandomForestModel(),
        "Gradient Boosting": GradientBoostingModel()
    }
    results = {}
    training_losses = {}
    validation_losses = {}

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
