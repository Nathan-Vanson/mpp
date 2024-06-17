import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tqdm.notebook import tqdm
from typing import List, Tuple, Union
import numpy as np
import pandas as pd

class NeuralNetwork(nn.Module):
    """
    Classe définissant l'architecture du réseau de neurones.

    Args:
    - input_size (int): Taille de l'entrée (nombre de caractéristiques en entrée).
    - hidden_sizes (list): Liste des tailles des couches cachées.
    - output_size (int): Taille de la sortie (par défaut : 1).

    Attributes:
    - network (nn.Sequential): Réseau de neurones séquentiel défini par les couches linéaires et ReLU.

    """
    def __init__(self, input_size, hidden_sizes, output_size=1):
        super(NeuralNetwork, self).__init__()
        layers = []
        in_size = input_size
        
        # Construction des couches du réseau
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))  # Couche linéaire
            layers.append(nn.ReLU())  # Fonction d'activation ReLU
            in_size = hidden_size
            
        # Couche de sortie
        layers.append(nn.Linear(in_size, output_size))
        
        # Définition du réseau séquentiel
        self.network = nn.Sequential(*layers)


    def forward(self, x: torch.Tensor):
        """
        Méthode de propagation avant (forward pass) du réseau de neurones.

        Args:
        - x (torch.Tensor): Tensor d'entrée.

        Returns:
        - torch.Tensor: Sortie du réseau après propagation avant.
        """
        return self.network(x)

class NeuralNetworkModel:
    """
    Classe pour l'entraînement, l'évaluation et le chargement d'un modèle de réseau de neurones utilisant PyTorch.

    Args:
    - input_size (int): Taille de l'entrée du modèle (nombre de caractéristiques).
    - hidden_sizes (List[int]): Liste des tailles des couches cachées du réseau.
    - lr (float): Taux d'apprentissage pour l'optimiseur Adam (par défaut : 0.001).
    - epochs (int): Nombre d'époques d'entraînement (par défaut : 50).
    - batch_size (int): Taille des mini-lots pour l'entraînement (par défaut : 32).
    - patience (int): Nombre d'époques de patience pour l'arrêt anticipé (par défaut : 10).

    Attributes:
    - model (NeuralNetwork): Instance du modèle de réseau de neurones défini par la classe NeuralNetwork.
    - lr (float): Taux d'apprentissage pour l'optimiseur Adam.
    - epochs (int): Nombre d'époques d'entraînement.
    - batch_size (int): Taille des mini-lots pour l'entraînement.
    - patience (int): Nombre d'époques de patience pour l'arrêt anticipé.
    - criterion (nn.MSELoss): Fonction de perte (Mean Squared Error).
    - optimizer (optim.Adam): Optimiseur Adam pour la mise à jour des poids du modèle.
    - scaler (StandardScaler): Objet de mise à l'échelle des données pour normaliser les données d'entrée.
    - best_loss (float): Meilleure perte de validation observée lors de l'entraînement.
    - early_stop (bool): Indicateur pour l'arrêt anticipé si la performance ne s'améliore pas.

    """
    def __init__(self, input_size, hidden_sizes, lr=0.001, epochs=50, batch_size=32, patience=10):
        self.model = NeuralNetwork(input_size, hidden_sizes)  # Initialisation du modèle de réseau de neurones
        self.lr = lr  # Taux d'apprentissage pour l'optimiseur Adam
        self.epochs = epochs  # Nombre d'époques d'entraînement
        self.batch_size = batch_size  # Taille des mini-lots pour l'entraînement
        self.patience = patience  # Nombre d'époques de patience pour l'arrêt anticipé
        self.criterion = nn.MSELoss()  # Fonction de perte (Mean Squared Error)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)  # Optimiseur Adam
        self.scaler = StandardScaler()  # Objet de mise à l'échelle des données
        self.best_loss = float('inf')  # Meilleure perte de validation initialisée à l'infini
        self.early_stop = False  # Indicateur pour l'arrêt anticipé initialisé à False


    def train(self, X_train: np.ndarray, y_train: pd.Series, X_val: np.ndarray, y_val: pd.Series) -> Tuple[List[float], List[float]]:
        """
        Méthode pour l'entraînement du modèle de réseau de neurones.

        Args:
        - X_train (np.ndarray): Données d'entraînement.
        - y_train (pd.Series): Cibles d'entraînement.
        - X_val (np.ndarray): Données de validation.
        - y_val (pd.Series): Cibles de validation.

        Returns:
        - Tuple[List[float], List[float]]: Liste des pertes d'entraînement et de validation par époque.
        """
        # Mise à l'échelle des données d'entraînement et de validation
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)

        # Création des ensembles de données TensorDataset pour les données d'entraînement et de validation
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train.values, dtype=torch.float32))
        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val.values, dtype=torch.float32))

        # Création des chargeurs de données DataLoader pour les ensembles de données d'entraînement et de validation
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Initialisation des listes pour stocker les pertes d'entraînement et de validation par époque
        training_loss = []
        validation_loss = []

        # Variable pour compter les époques consécutives sans amélioration de la perte de validation
        epochs_no_improve = 0
        
        # Boucle sur le nombre d'époques
        for epoch in tqdm(range(self.epochs), desc=f"Entraînement des Époques ({self.epochs})"):
            epoch_train_loss = 0
            self.model.train()  # Passage en mode entraînement du modèle
            # Boucle sur les mini-lots dans l'ensemble de données d'entraînement
            for X_batch, y_batch in train_loader:
                self.optimizer.zero_grad()  # Réinitialisation des gradients à zéro
                outputs = self.model(X_batch)  # Prédiction avec le modèle
                loss = self.criterion(outputs.squeeze(), y_batch)  # Calcul de la perte
                loss.backward()  # Rétropropagation pour calculer les gradients
                self.optimizer.step()  # Mise à jour des poids du modèle
                epoch_train_loss += loss.item()  # Accumulation de la perte par mini-lot
            epoch_train_loss /= len(train_loader)  # Calcul de la perte moyenne par époque
            training_loss.append(epoch_train_loss)  # Ajout de la perte d'entraînement à la liste

            epoch_val_loss = 0
            self.model.eval()  # Passage en mode évaluation du modèle
            with torch.no_grad():
                # Boucle sur les mini-lots dans l'ensemble de données de validation
                for X_batch, y_batch in val_loader:
                    outputs = self.model(X_batch)  # Prédiction avec le modèle
                    loss = self.criterion(outputs.squeeze(), y_batch)  # Calcul de la perte
                    epoch_val_loss += loss.item()  # Accumulation de la perte par mini-lot
            epoch_val_loss /= len(val_loader)  # Calcul de la perte moyenne par époque
            validation_loss.append(epoch_val_loss)  # Ajout de la perte de validation à la liste

            #print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {epoch_train_loss}, Val Loss: {epoch_val_loss}")

            # Comparaison avec la meilleure perte de validation observée
            if epoch_val_loss < self.best_loss:
                self.best_loss = epoch_val_loss  # Mise à jour de la meilleure perte de validation
                epochs_no_improve = 0  # Réinitialisation du compteur d'époques sans amélioration
                # Sauvegarde du meilleur modèle (optionnel)
                # torch.save(self.model.state_dict(), f'models/best_model_{epoch}.pth')
            else:
                epochs_no_improve += 1  # Incrémentation du compteur d'époques sans amélioration

            # Vérification de l'arrêt anticipé si la perte de validation n'améliore pas depuis 'patience' époques
            if epochs_no_improve >= self.patience:
                self.early_stop = True  # Activation de l'arrêt anticipé
                break

        return training_loss, validation_loss  # Retourne les listes de pertes d'entraînement et de validation par époque


    def evaluate(self, X_test: np.ndarray, y_test: pd.Series) -> float:
        """
        Méthode pour évaluer les performances du modèle sur un ensemble de test.

        Args:
        - X_test (np.ndarray): Données de test.
        - y_test (pd.Series): Cibles de test.

        Returns:
        - float: Erreur quadratique moyenne (MSE) entre les prédictions et les cibles.
        """
        # Mise à l'échelle des données de test
        X_test = self.scaler.transform(X_test)

        with torch.no_grad():
            # Prédiction avec le modèle en mode évaluation
            predictions = self.model(torch.tensor(X_test, dtype=torch.float32)).squeeze()

        # Calcul de l'erreur quadratique moyenne (MSE) entre les prédictions et les cibles
        mse = mean_squared_error(y_test, predictions.numpy())
        return mse


    def load_model(self, model_path: str):
        """
        Méthode pour charger les poids d'un modèle préalablement sauvegardé.

        Args:
        - model_path (str): Chemin vers le fichier contenant les poids du modèle.
        """
        self.model.load_state_dict(torch.load(model_path))  # Chargement des poids du modèle
        self.model.eval()  # Passage en mode évaluation du modèle
