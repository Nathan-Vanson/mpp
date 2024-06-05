import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_matrix(data, output_path="correlation_matrix.png"):
    plt.figure(figsize=(12, 10))
    numeric_data = data.select_dtypes(include=[float, int])  # Exclude non-numeric columns
    correlation_matrix = numeric_data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap=plt.cm.Reds)
    plt.title("Correlation Matrix")
    # plt.savefig(output_path)
    plt.show()
    
def plot_feature_correlations(data, target_column, output_path="feature_correlations.png", threshold=0.1):
    plt.figure(figsize=(12, 8))
    correlations = data.corr()[target_column].sort_values(ascending=False)
    correlations.drop(target_column, inplace=True)  # Drop the target itself
    bars = plt.barh(correlations.index, correlations.values, color='skyblue')
    plt.axvline(x=threshold, color='red', linestyle='--')
    plt.axvline(x=-threshold, color='red', linestyle='--')
    plt.title(f"Feature Correlations with {target_column}\nThreshold: {threshold}")
    plt.xlabel(f"Correlation with {target_column}")
    #plt.savefig(output_path)
    plt.show()

def plot_results(results, output_path="results.png"):
    names = list(results.keys())
    mse_values = list(results.values())

    plt.figure(figsize=(10, 6))
    plt.barh(names, mse_values, color='skyblue')
    plt.xlabel("Mean Squared Error")
    plt.title("Model Performance")
    # plt.savefig(output_path)
    plt.show()

def plot_training_process(training_loss, validation_loss, output_path="training_process.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(training_loss, label="Training Loss")
    plt.plot(validation_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Process")
    # plt.savefig(output_path)
    plt.show()

def plot_multiple_training_processes(training_losses, validation_losses, output_path="training_processes.png"):
    plt.figure(figsize=(10, 6))
    for model_name in training_losses:
        plt.plot(training_losses[model_name], label=f"{model_name} Training Loss")
        plt.plot(validation_losses[model_name], label=f"{model_name} Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Processes")
    # plt.savefig(output_path)
    plt.show()
