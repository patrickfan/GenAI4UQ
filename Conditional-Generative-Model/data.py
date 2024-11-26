import numpy as np
import os
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from config import Config

class DataGenerator:
    def __init__(self, config):
        self.config = config

    def load_data(self):
        """Load data from files or generate synthetic data."""
        if self.config.USE_SYNTHETIC_DATA:
            x_sample, y_sample = self.generate_data()
            print ("Load synthetic data!")
        else:
            current_dir = os.path.abspath(os.path.dirname(__file__))
            x_path = os.path.join(current_dir, self.config.X_SAMPLE_PATH)
            y_path = os.path.join(current_dir, self.config.Y_SAMPLE_PATH)
            x_sample = np.load(x_path)
            y_sample = np.load(y_path)
            
        X_train_val, X_test, y_train_val, y_test = train_test_split(x_sample, y_sample, test_size=0.1, random_state=42)

        return X_train_val.astype(np.float32), y_train_val.astype(np.float32), X_test.astype(np.float32), y_test.astype(np.float32)

    def generate_data(self):
        """Generate synthetic data with quadratic relationship."""
        interval_a, interval_b = -2.0, 2.0
        y_var = 0.01
        x_sample = np.linspace(interval_a, interval_b, self.config.SAMPLE_SIZE).reshape(-1, 1)
        y_sample = x_sample ** 2 + np.random.normal(0, np.sqrt(y_var), (self.config.SAMPLE_SIZE, 1))
        return x_sample, y_sample

class CustomDataset(Dataset):
    def __init__(self, X, y):
        # Ensure the tensors are created on the same device as the input data
        self.X = X if isinstance(X, torch.Tensor) else torch.tensor(X, dtype=torch.float32, device=X.device)
        self.y = y if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.float32, device=y.device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DataModule:
    def __init__(self, config):
        self.config = config
        self.data_generator = DataGenerator(config)
        self.device = Config.DEVICE

    def prepare_dataloaders(self, xTrain, yTrain, batch_size):
        """Prepare training and validation data loaders."""
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            xTrain, yTrain,
            test_size=0.2, random_state=42
        )

        # Convert numpy arrays (if they are) to tensors and ensure they are on the correct device
        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        X_val = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_val = torch.tensor(y_val, dtype=torch.float32).to(self.device)

        # Create DataLoader
        train_loader = DataLoader(
            CustomDataset(X_train, y_train),
            batch_size=batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            CustomDataset(X_val, y_val),
            batch_size=batch_size,
            shuffle=False
        )

        return train_loader, val_loader






