import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tqdm.notebook import tqdm

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size=1):
        super(NeuralNetwork, self).__init__()
        layers = []
        in_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size
        layers.append(nn.Linear(in_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class NeuralNetworkModel:
    def __init__(self, input_size, hidden_sizes, lr=0.001, epochs=50, batch_size=32, patience=10):
        self.model = NeuralNetwork(input_size, hidden_sizes)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scaler = StandardScaler()
        self.best_loss = float('inf')
        self.early_stop = False

    def train(self, X_train, y_train, X_val, y_val):
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train.values, dtype=torch.float32))
        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val.values, dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        training_loss = []
        validation_loss = []
        epochs_no_improve = 0
        
        for epoch in tqdm(range(self.epochs), desc=f"Training Epochs ({self.epochs})"):
            epoch_train_loss = 0
            self.model.train()
            for X_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs.squeeze(), y_batch)
                loss.backward()
                self.optimizer.step()
                epoch_train_loss += loss.item()
            epoch_train_loss /= len(train_loader)
            training_loss.append(epoch_train_loss)

            epoch_val_loss = 0
            self.model.eval()
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    outputs = self.model(X_batch)
                    loss = self.criterion(outputs.squeeze(), y_batch)
                    epoch_val_loss += loss.item()
            epoch_val_loss /= len(val_loader)
            validation_loss.append(epoch_val_loss)

            #print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {epoch_train_loss}, Val Loss: {epoch_val_loss}")

            if epoch_val_loss < self.best_loss:
                self.best_loss = epoch_val_loss
                epochs_no_improve = 0
                #torch.save(self.model.state_dict(), f'models/best_model_{epoch}.pth')
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= self.patience:
                #print("Early stopping")
                self.early_stop = True
                break

        return training_loss, validation_loss

    def evaluate(self, X_test, y_test):
        X_test = self.scaler.transform(X_test)
        with torch.no_grad():
            predictions = self.model(torch.tensor(X_test, dtype=torch.float32)).squeeze()
        mse = mean_squared_error(y_test, predictions.numpy())
        return mse

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()  # Set the model to evaluation mode
