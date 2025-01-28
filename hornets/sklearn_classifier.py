import logging
from typing import Any, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from hornets.dataset import E2EDatasetLoader
from hornets.model import HorNetsArchitecture 


class HorNetClassifier(BaseEstimator, ClassifierMixin):
    """
    HorNetClassifier is a scikit-learn compatible classifier using the HorNet neural network model.

    Parameters:
        batch_size (int, default=4): Number of samples per batch.
        num_epochs (int, default=1000): Number of training epochs.
        learning_rate (float, default=1e-4): Learning rate for the optimizer.
        stopping_crit (int, default=10): Number of epochs with no improvement to wait before stopping training.
        feature_names (Optional[List[str]], default=None): Names of the input features.
        num_rules (int, default=256): Number of rules for feature combinations.
        activation (str, default="polyclip"): Activation function to use ("polyclip" or others).
        comb_samples_fp (int, default=48): Number of combination samples for feature processing.
        exp_param (int, default=1): Expansion parameter for polynomial clipping.
        order (int, default=5): Order of feature combinations.
        random_state (Optional[int], default=None): Seed for random number generators for reproducibility.
        verbose (bool, default=False): If True, enables verbose output.
    """

    def __init__(
        self,
        batch_size: int = 4,
        num_epochs: int = 1000,
        learning_rate: float = 1e-4,
        stopping_crit: int = 10,
        feature_names: Optional[List[str]] = None,
        num_rules: int = 256,
        activation: str = "polyclip",
        comb_samples_fp: int = 48,
        exp_param: int = 1,
        order: int = 5,
        device: str = "cpu",
        random_state: Optional[int] = None,
        verbose: bool = False,
    ):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.stopping_crit = stopping_crit
        self.feature_names = feature_names
        self.num_rules = num_rules
        self.activation = activation
        self.comb_samples_fp = comb_samples_fp
        self.exp_param = exp_param
        self.order = order
        self.random_state = random_state
        self.verbose = verbose

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
        )
        self.model_ = None
        self.optimizer_ = None
        self.criterion_ = None
        self.label_encoder_ = LabelEncoder()
        self.combination_order_ = None
        self.num_params_ = None
        self.losses_ = []
        self.stopping_iteration_ = 0

        if self.verbose:
            logging.basicConfig(level=logging.INFO)
        else:
            logging.basicConfig(level=logging.WARNING)

    def _set_random_state(self):
        """
        Sets the random seed for numpy, torch, and random modules for reproducibility.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)
            torch.cuda.manual_seed_all(self.random_state)
            import random

            random.seed(self.random_state)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _augment_features_with_synthetic(self, features: np.ndarray) -> np.ndarray:
        """
        Augments the feature matrix with synthetic features based on combinations.

        Parameters:
            features (np.ndarray): Original feature matrix.

        Returns:
            np.ndarray: Augmented feature matrix with synthetic features.
        """
        joint_space = np.zeros((features.shape[0], self.order + features.shape[1]))

        if isinstance(features, np.matrix):
            joint_space[: features.shape[0], : features.shape[1]] = features[:, :]
        else:
            try:
                joint_space[: features.shape[0], : features.shape[1]] = features[
                    :, :
                ].todense()
            except AttributeError:
                joint_space[: features.shape[0], : features.shape[1]] = features[:, :]
        return joint_space

    def fit(self, X: np.ndarray, y: Any, upsample: int = 0) -> "HorNetClassifier":
        """
        Fit the HorNet classifier according to the given training data.

        Parameters:
            X (np.ndarray): Training feature matrix.
            y (array-like): Target values.
            upsample (int, default=0): Number of upsampled samples to include.

        Returns:
            self
        """
        self._set_random_state()

        X, y = check_X_y(X, y)

        y = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_

        X_aug = self._augment_features_with_synthetic(X)

        if self.feature_names is None:
            self.feature_names_ = [f"Feature_{i}" for i in range(X_aug.shape[1])]
        else:
            self.feature_names_ = self.feature_names + [
                f"synth{x}" for x in range(self.order)
            ]

        train_dataset = E2EDatasetLoader(X_aug, y)
        dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=1,
            drop_last=True,
        )

        otpt = len(self.classes_)
        dim = X_aug.shape[1]
        self.combination_order_ = min(dim, self.order)

        if self.verbose:
            logging.info(f"Using combination order: {self.combination_order_}")

        self.model_ = HorNetsArchitecture(
            dim=dim,
            outpt=otpt,
            num_rules=self.num_rules,
            num_features=dim,
            exp_param=self.exp_param,
            feature_names=self.feature_names_,
            activation=self.activation,
            order=self.combination_order_,
            device=self.device,
        ).to(self.device)

        self.optimizer_ = Adam(self.model_.parameters(), lr=self.learning_rate)
        self.criterion_ = nn.CrossEntropyLoss()

        self.num_params_ = sum(p.numel() for p in self.model_.parameters())

        if self.verbose:
            logging.info(f"Number of parameters: {self.num_params_}")
            logging.info(f"Starting training for {self.num_epochs} epochs")

        self.model_.train()
        self.stopping_iteration_ = 0
        best_loss = np.inf

        for epoch in tqdm(
            range(self.num_epochs), desc="Training Epochs", disable=not self.verbose
        ):
            if self.stopping_iteration_ > self.stopping_crit:
                if self.verbose:
                    logging.info("Early stopping triggered.")
                break

            epoch_losses = []
            epoch_loader = tqdm(
                dataloader,
                desc=f"Epoch {epoch+1}/{self.num_epochs}",
                disable=not self.verbose,
                leave=False,
            )
            for batch_features, batch_labels in epoch_loader:
                batch_features = batch_features.float().to(self.device)
                batch_labels = batch_labels.long().to(self.device)

                outputs = self.model_(batch_features, num_samples=self.comb_samples_fp)
                loss = self.criterion_(outputs, batch_labels)

                self.optimizer_.zero_grad()
                loss.backward()
                self.optimizer_.step()

                batch_loss = loss.item()
                epoch_losses.append(batch_loss)

                if self.verbose:
                    epoch_loader.set_postfix({"loss": f"{batch_loss:.4f}"})

            mean_loss = np.mean(epoch_losses)
            self.losses_.append(mean_loss)

            if mean_loss < best_loss - 1e-4:
                best_loss = mean_loss
                self.stopping_iteration_ = 0
            else:
                self.stopping_iteration_ += 1

            if self.verbose:
                logging.info(
                    f"Epoch {epoch+1}/{self.num_epochs} - Mean Loss: {mean_loss:.4f} "
                    f"- Stopping Iteration: {self.stopping_iteration_}"
                )
                # self.get_top_rules(top_k=3)

            if mean_loss <= 0.01:
                if self.verbose:
                    logging.info("Desired loss achieved. Stopping training.")
                break

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Perform classification on samples in X.

        Parameters:
            X (np.ndarray): Input samples.

        Returns:
            np.ndarray: Predicted class labels.
        """
        check_is_fitted(self, "model_")
        X = check_array(X)

        X_aug = self._augment_features_with_synthetic(X)

        test_dataset = E2EDatasetLoader(X_aug, None)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        self.model_.eval()
        predictions = []

        with torch.no_grad():
            for batch_features in test_dataloader:
                batch_features = batch_features.float().to(self.device)
                output = self.model_(batch_features)
                pred = torch.argmax(output, dim=1).cpu().numpy()
                predictions.append(pred[0])

        predictions = self.label_encoder_.inverse_transform(predictions)
        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for X.

        Parameters:
            X (np.ndarray): Input samples.

        Returns:
            np.ndarray: Predicted class probabilities.
        """
        # Ensure the model is fitted
        check_is_fitted(self, "model_")
        X = check_array(X)

        # Augment features with synthetic combinations
        X_aug = self._augment_features_with_synthetic(X)

        # Create dataset and dataloader for prediction
        test_dataset = E2EDatasetLoader(X_aug, None)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        self.model_.eval()
        probabilities = []

        with torch.no_grad():
            for batch_features in test_dataloader:
                batch_features = batch_features.float().to(self.device)
                output = self.model_(batch_features)
                prob = nn.functional.softmax(output, dim=1).cpu().numpy()
                probabilities.append(prob[0])

        return np.array(probabilities)

    def get_top_rules(self, top_k: int = 3) -> None:
        """
        Logs the top K feature combinations based on their scores.

        Parameters:
            top_k (int, optional): Number of top combinations to display. Default is 3.
        """
        if self.feature_names_ is None:
            logging.warning("Feature names are not provided.")
            feature_names = [f"Feature_{i}" for i in range(self.model_.num_features)]
        else:
            feature_names = self.feature_names_

        if not hasattr(self.model_, "comb_scores"):
            logging.warning("Combination scores are not available.")
            return

        combination_scores = self.model_.comb_scores.detach().cpu().numpy()

        # Identify top K combinations based on scores
        top_indices = np.argsort(combination_scores)[::-1][:top_k]
        for idx in top_indices:
            combination = self.model_.comb_indices[idx]
            feature_combination = [
                feature_names[i] for i in combination if "synth" not in feature_names[i]
            ]
            score = combination_scores[idx]
            logging.info(
                f"Feature Combination: {feature_combination}, Score: {score:.4f}"
            )
