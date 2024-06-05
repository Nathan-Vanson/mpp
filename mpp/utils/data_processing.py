import pandas as pd
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import Descriptors

class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        self.data.dropna(inplace=True)
        return self.data

    def extract_features(self):
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
        self.data = pd.concat([self.data.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)
        self.data.drop(columns=['SMILES'], inplace=True)
        return self.data

    def split_data(self, test_size=0.2, random_state=42):
        X = self.data.drop(['pIC50', 'logP'], axis=1)
        y_pic50 = self.data['pIC50']
        y_logP = self.data['logP']
        X_train, X_test, y_train_pic50, y_test_pic50 = train_test_split(X, y_pic50, test_size=test_size, random_state=random_state)
        X_train, X_test, y_train_logP, y_test_logP = train_test_split(X, y_logP, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train_pic50, y_test_pic50, y_train_logP, y_test_logP
