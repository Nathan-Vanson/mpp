import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List

def plot_correlation_matrix(data: pd.DataFrame, output_path: str = "correlation_matrix.png") -> None:
    """
    Trace la matrice de corrélation pour les colonnes numériques dans le DataFrame donné.

    Args:
    - data (pd.DataFrame): DataFrame d'entrée contenant des colonnes numériques.
    - output_path (str): Chemin du fichier pour sauvegarder le graphique (par défaut : 'correlation_matrix.png').

    Returns:
    - None
    """
    plt.figure(figsize=(12, 10))
    data_numerique = data.select_dtypes(include=[float, int])  # Exclut les colonnes non numériques
    matrice_correlation = data_numerique.corr()
    sns.heatmap(matrice_correlation, annot=True, cmap=plt.cm.Reds)
    plt.title("Matrice de Corrélation")
    # Décommentez la ligne ci-dessous pour sauvegarder le graphique dans un fichier
    # plt.savefig(output_path)
    plt.show()


def plot_feature_correlations(data: pd.DataFrame, target_column: str, output_path: str = "feature_correlations.png", threshold: float = 0.1) -> None:
    """
    Trace les corrélations des caractéristiques avec la colonne cible spécifiée.

    Args:
    - data (pd.DataFrame): DataFrame d'entrée contenant les caractéristiques et la colonne cible.
    - target_column (str): Nom de la colonne cible pour l'analyse de corrélation.
    - output_path (str): Chemin du fichier pour sauvegarder le graphique (par défaut : 'feature_correlations.png').
    - threshold (float): Valeur seuil pour mettre en évidence les corrélations (par défaut : 0.1), permet la pertinence des caractéristiques obtenues lors de la visualisation.

    Returns:
    - None
    """
    plt.figure(figsize=(12, 8))
    correlations = data.corr()[target_column].sort_values(ascending=False)
    correlations.drop(target_column, inplace=True)  # Supprime la cible elle-même
    plt.barh(correlations.index, correlations.values, color='skyblue')
    plt.axvline(x=threshold, color='red', linestyle='--')
    plt.axvline(x=-threshold, color='red', linestyle='--')
    plt.title(f"Corrélations des Caractéristiques avec {target_column}\nSeuil : {threshold}")
    plt.xlabel(f"Corrélation avec {target_column}")
    # Décommentez la ligne ci-dessous pour sauvegarder le graphique dans un fichier
    # plt.savefig(output_path)
    plt.show()


def plot_results(results: Dict[str, float], title="Performance du Modèle", erreur="Erreur Quadratique Moyenne", output_path: str = "results.png") -> None:
    """
    Trace les résultats de performance (par exemple, l'Erreur Quadratique Moyenne) pour différents modèles.

    Args:
    - results (Dict[str, float]): Dictionnaire contenant les noms des modèles en clé et les valeurs d'EQM en valeur.
    - output_path (str): Chemin du fichier pour sauvegarder le graphique (par défaut : 'results.png').

    Returns:
    - None
    """
    # Trier les résultats par valeur
    sorted_results = dict(sorted(results.items(), key=lambda item: item[1]))

    names = list(sorted_results.keys())
    mse_values = list(sorted_results.values())

    plt.figure(figsize=(10, 6))
    plt.barh(names, mse_values, color='skyblue')
    plt.xlabel(erreur)
    plt.title(title)
    # Décommentez la ligne ci-dessous pour sauvegarder le graphique dans un fichier
    # plt.savefig(output_path)
    plt.show()


def plot_training_process(training_loss: List[float], validation_loss: List[float], output_path: str = "training_process.png") -> None:
    """
    Trace l'évolution de la perte d'entraînement et de validation au fil des epochs.

    Args:
    - training_loss (List[float]): Liste des valeurs de perte d'entraînement par epoch.
    - validation_loss (List[float]): Liste des valeurs de perte de validation par epoch.
    - output_path (str): Chemin du fichier pour sauvegarder le graphique (par défaut : 'training_process.png').

    Returns:
    - None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(training_loss, label="Perte d'Entraînement")
    plt.plot(validation_loss, label="Perte de Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Perte")
    plt.legend()
    plt.title("Processus d'Entraînement")
    # Décommentez la ligne ci-dessous pour sauvegarder le graphique dans un fichier
    # plt.savefig(output_path)
    plt.show()


def plot_multiple_training_processes(training_losses: Dict[str, List[float]], validation_losses: Dict[str, List[float]], output_path: str = "training_processes.png") -> None:
    """
    Trace la perte d'entraînement et de validation pour plusieurs modèles au fil des epochs.

    Args:
    - training_losses (Dict[str, List[float]]): Dictionnaire où les clés sont les noms des modèles et les valeurs sont des listes des valeurs de perte d'entraînement par epoch.
    - validation_losses (Dict[str, List[float]]): Dictionnaire où les clés sont les noms des modèles et les valeurs sont des listes des valeurs de perte de validation par epoch.
    - output_path (str): Chemin du fichier pour sauvegarder le graphique (par défaut : 'training_processes.png').

    Returns:
    - None
    """
    plt.figure(figsize=(10, 6))
    for model_name in training_losses:
        plt.plot(training_losses[model_name], label=f"Perte d'Entraînement {model_name}")
        plt.plot(validation_losses[model_name], label=f"Perte de Validation {model_name}")
    plt.xlabel("Epochs")
    plt.ylabel("Perte")
    plt.legend()
    plt.title("Processus d'Entraînement pour Plusieurs Modèles")
    # Décommentez la ligne ci-dessous pour sauvegarder le graphique dans un fichier
    # plt.savefig(output_path)
    plt.show()

def plot_prediction(y_test_valeurs: np.ndarray, predictions: pd.Series, title: str ='Valeurs Réelles vs Prédictions') -> None:
    """
    Méthode pour tracer les prédictions du modèle par rapport aux valeurs réelles.

    Args:
    - y_test (pd.Series): Valeurs réelles des données de test.
    - predictions (np.ndarray): valeurs prédites
    """

    # Tracé des valeurs réelles vs prédictions
    plt.figure(figsize=(15, 10))
    plt.plot(y_test_valeurs, label='Valeurs réelles')
    plt.plot(predictions, label='Prédictions', alpha=0.7)
    plt.xlabel('Index')
    plt.ylabel('Valeurs')
    plt.title(title)
    plt.legend()
    plt.show()
