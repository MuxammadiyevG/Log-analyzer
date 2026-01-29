"""
Autoencoder-based anomaly detector
"""

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from app.models.base import BaseAnomalyDetector
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class AutoencoderNetwork(nn.Module):
    """Autoencoder neural network"""

    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int):
        """
        Initialize autoencoder

        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            latent_dim: Latent space dimension
        """
        super().__init__()

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(0.1),
                ]
            )
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(0.1),
                ]
            )
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        """Forward pass"""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AutoencoderDetector(BaseAnomalyDetector):
    """Autoencoder-based anomaly detection"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Autoencoder detector

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)

        # Get parameters
        self.input_dim = config.get("input_dim", 32)
        self.hidden_dims = config.get("hidden_dims", [64, 32, 16, 8])
        self.latent_dim = config.get("latent_dim", 4)
        self.learning_rate = config.get("learning_rate", 0.001)
        self.epochs = config.get("epochs", 50)
        self.batch_size = config.get("batch_size", 256)
        self.threshold_percentile = config.get(
            "reconstruction_threshold_percentile", 95
        )

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model
        self.model = AutoencoderNetwork(
            self.input_dim, self.hidden_dims, self.latent_dim
        ).to(self.device)

        # Threshold for anomaly detection
        self.threshold = None

        logger.info(f"Autoencoder initialized on {self.device}")

    def train(self, X: np.ndarray) -> None:
        """
        Train autoencoder

        Args:
            X: Training data (n_samples, n_features)
        """
        logger.info(f"Training Autoencoder on {X.shape[0]} samples...")

        # Update input_dim if needed
        if X.shape[1] != self.input_dim:
            logger.warning(
                f"Input dim mismatch. Expected {self.input_dim}, got {X.shape[1]}"
            )
            self.input_dim = X.shape[1]
            self.model = AutoencoderNetwork(
                self.input_dim, self.hidden_dims, self.latent_dim
            ).to(self.device)

        # Prepare data
        X_tensor = torch.FloatTensor(X)
        dataset = TensorDataset(X_tensor, X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Optimizer and loss
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, _ in dataloader:
                batch_X = batch_X.to(self.device)

                # Forward pass
                reconstructed = self.model(batch_X)
                loss = criterion(reconstructed, batch_X)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.6f}")

        # Compute threshold
        self.model.eval()
        with torch.no_grad():
            X_tensor = X_tensor.to(self.device)
            reconstructed = self.model(X_tensor)
            reconstruction_errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
            reconstruction_errors = reconstruction_errors.cpu().numpy()

        self.threshold = np.percentile(reconstruction_errors, self.threshold_percentile)

        self.is_trained = True
        logger.info(f"Training complete. Threshold: {self.threshold:.6f}")

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomalies

        Args:
            X: Input data (n_samples, n_features)

        Returns:
            Tuple of (predictions, scores)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            reconstructed = self.model(X_tensor)

            # Compute reconstruction errors
            reconstruction_errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
            scores = reconstruction_errors.cpu().numpy()

        # Predictions based on threshold
        predictions = (scores > self.threshold).astype(int)

        return predictions, scores

    def save(self, path: Path) -> None:
        """
        Save model to disk

        Args:
            path: Path to save model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_path = path / "autoencoder.pt"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": self.config,
                "threshold": self.threshold,
                "input_dim": self.input_dim,
                "is_trained": self.is_trained,
            },
            model_path,
        )

        logger.info(f"Model saved to {model_path}")

    def load(self, path: Path) -> None:
        """
        Load model from disk

        Args:
            path: Path to load model from
        """
        path = Path(path)
        model_path = path / "autoencoder.pt"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)

        self.config = checkpoint["config"]
        self.threshold = checkpoint["threshold"]
        self.input_dim = checkpoint["input_dim"]
        self.is_trained = checkpoint["is_trained"]

        # Recreate model with correct dimensions
        self.model = AutoencoderNetwork(
            self.input_dim, self.hidden_dims, self.latent_dim
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        logger.info(f"Model loaded from {model_path}")
