from mpp.utils.data_processing import DataProcessor
from mpp.utils.model_utils import train_and_evaluate_models
from mpp.models.neural_network import NeuralNetworkModel
from mpp.utils.visualization import plot_correlation_matrix, plot_results, plot_training_process, plot_multiple_training_processes
from mpp.tuning.tune import HyperparameterTuning

# Data Processing
processor = DataProcessor("data/data.csv")
data = processor.load_data()
data_with_features = processor.extract_features()

# Dataset Splitting
X_train, X_test, y_train_pic50, y_test_pic50, y_train_logP, y_test_logP = processor.split_data()

# Feature Visualization
plot_correlation_matrix(data_with_features)

# Hyperparameter Tuning for Neural Network (pIC50)
tuner_pic50 = HyperparameterTuning(X_train, y_train_pic50, "Neural Network")
best_params_pic50 = tuner_pic50.tune(n_trials=50)

# Neural Network Model Training and Evaluation (pIC50)
nn_model_pic50 = NeuralNetworkModel(
    input_size=X_train.shape[1],
    hidden_sizes=[best_params_pic50[f"hidden_size_{i}"] for i in range(best_params_pic50["hidden_layers"])],
    lr=best_params_pic50["lr"],
    epochs=best_params_pic50["epochs"],
    batch_size=best_params_pic50["batch_size"],
    patience=best_params_pic50["patience"]
)
training_loss_pic50, validation_loss_pic50 = nn_model_pic50.train(X_train, y_train_pic50, X_test, y_test_pic50)
mse_pic50 = nn_model_pic50.evaluate(X_test, y_test_pic50)
print(f"Neural Network Model MSE for pIC50: {mse_pic50}")

# Visualize Training Process (pIC50)
plot_training_process(training_loss_pic50, validation_loss_pic50, output_path="training_process_pic50.png")

# Hyperparameter Tuning for Neural Network (logP)
tuner_logP = HyperparameterTuning(X_train, y_train_logP, "Neural Network")
best_params_logP = tuner_logP.tune(n_trials=50)

# Neural Network Model Training and Evaluation (logP)
nn_model_logP = NeuralNetworkModel(
    input_size=X_train.shape[1],
    hidden_sizes=[best_params_logP[f"hidden_size_{i}"] for i in range(best_params_logP["hidden_layers"])],
    lr=best_params_logP["lr"],
    epochs=best_params_logP["epochs"],
    batch_size=best_params_logP["batch_size"],
    patience=best_params_logP["patience"]
)
training_loss_logP, validation_loss_logP = nn_model_logP.train(X_train, y_train_logP, X_test, y_test_logP)
mse_logP = nn_model_logP.evaluate(X_test, y_test_logP)
print(f"Neural Network Model MSE for logP: {mse_logP}")

# Visualize Training Process (logP)
plot_training_process(training_loss_logP, validation_loss_logP, output_path="training_process_logP.png")

# Train and evaluate other models for pIC50
results_pic50, training_losses_pic50, validation_losses_pic50 = train_and_evaluate_models(X_train, X_test, y_train_pic50, y_test_pic50)

# Train and evaluate other models for logP
results_logP, training_losses_logP, validation_losses_logP = train_and_evaluate_models(X_train, X_test, y_train_logP, y_test_logP)

# Results visualization for pIC50
plot_results(results_pic50, output_path="results_pic50.png")

# Results visualization for logP
plot_results(results_logP, output_path="results_logP.png")

# Visualize training processes for all models (pIC50)
plot_multiple_training_processes(training_losses_pic50, validation_losses_pic50, output_path="training_processes_pic50.png")

# Visualize training processes for all models (logP)
plot_multiple_training_processes(training_losses_logP, validation_losses_logP, output_path="training_processes_logP.png")
