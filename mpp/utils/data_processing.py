import pandas as pd
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import Descriptors
from typing import Tuple, List

class DataProcessor:
    def __init__(self, file_path: str):
        """
        Initialise le DataProcessor avec le chemin vers le fichier CSV.

        Args:
        - file_path (str): Chemin vers le fichier CSV contenant les données.
        """
        self.file_path = file_path
        self.data = None


    def load_data(self) -> pd.DataFrame:
        """
        Charge les données à partir du fichier CSV spécifié par self.file_path,
        supprime les lignes avec des valeurs NaN et renvoie le DataFrame.

        Returns:
        - pd.DataFrame: DataFrame contenant les données chargées et nettoyées.
        """
        self.data = pd.read_csv(self.file_path)
        self.data.dropna(inplace=True)
        return self.data


    def extract_features(self) -> pd.DataFrame:
        """
        Extrait les caractéristiques moléculaires à partir des SMILES (simplifié moléculaire) présents dans le DataFrame self.data.

        Utilise RDKit pour calculer le poids moléculaire, le nombre d'atomes lourds, le nombre de liaisons,
        et le nombre d'atomes spécifiques (C, O, N, Cl, F, Br, I).

        Returns:
        - pd.DataFrame: DataFrame mis à jour avec les caractéristiques moléculaires extraites.
        """
        features = []
        for smile in self.data['SMILES']:
            mol = Chem.MolFromSmiles(smile)
            features.append({
                'MolWt': Descriptors.MolWt(mol),
                'NumAtoms': Descriptors.HeavyAtomCount(mol),
                'NumBonds': mol.GetNumBonds(),
                'NumC': smile.count('C'),
                'NumO': smile.count('O'),
                'NumN': smile.count('N'),
                'NumCl': smile.count('Cl'),
                'NumF': smile.count('F'),
                'NumBr': smile.count('Br'),
                'NumI': smile.count('I')
            })
        features_df = pd.DataFrame(features)
        # Concaténer les nouvelles caractéristiques extraites avec les données existantes
        self.data = pd.concat([self.data.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)
        # Supprimer la colonne 'SMILES' puisque les caractéristiques sont maintenant extraites
        self.data.drop(columns=['SMILES'], inplace=True)
        return self.data


    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Divise les données en ensembles d'entraînement et de test pour les variables cibles pIC50 et logP.

        Args:
        - test_size (float): Taille de l'ensemble de test, par défaut 0.2 (20%).
        - random_state (int): Graine aléatoire pour la reproductibilité de la division des données, par défaut 42.

        Returns:
        - Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]: 
          X_train, X_test, y_train_pic50, y_test_pic50, y_train_logP, y_test_logP.
        """
        X = self.data.drop(['pIC50', 'logP'], axis=1)
        y_pic50 = self.data['pIC50']
        y_logP = self.data['logP']
        
        # Diviser X et y_pic50 en ensembles d'entraînement et de test pour pIC50
        X_train, X_test, y_train_pic50, y_test_pic50 = train_test_split(X, y_pic50, test_size=test_size, random_state=random_state)
        
        # Diviser X et y_logP en ensembles d'entraînement et de test pour logP
        X_train, X_test, y_train_logP, y_test_logP = train_test_split(X, y_logP, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train_pic50, y_test_pic50, y_train_logP, y_test_logP
