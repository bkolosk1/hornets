import torch
import numpy as np
from torch.utils.data import Dataset


class E2EDatasetLoader(Dataset):
    def __init__(self, features, targets=None):
        self.features = features
        self.targets = targets

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, index):
        part = self.features[index]
        instance = torch.from_numpy(part)
        if self.targets is not None:
            target = torch.from_numpy(np.array(self.targets[index]))
            return instance.to(torch.float), target
        return instance.to(torch.float)


def generate_synthetic_data(
    num_features, num_instances=128, operation="xor", random_seed=42
):
    """
    Generate synthetic datasets (X and y) for a specific logical operation.

    Parameters:
    - num_features (int): Number of features to use for synthetic data.
    - num_instances (int): Number of instances to generate for the dataset.
    - operation (str): Logical operation to generate data for (e.g., 'xor', 'and', 'not', 'or', 'xnor').

    Returns:
    - X (np.ndarray): Generated feature matrix of shape (num_instances, num_features).
    - y (np.ndarray): Generated target array of shape (num_instances,).
    """
    np.random.seed(random_seed)

    supported_operations = ["xor", "and", "not", "or", "xnor"]
    if operation not in supported_operations:
        raise ValueError(
            f"Unsupported operation: {operation}. Supported operations are {supported_operations}."
        )

    X = np.random.randint(0, 2, size=(num_instances, num_features))

    if operation == "xor":
        y = np.logical_xor(X[:, 0], X[:, 1]).astype(int)
    elif operation == "and":
        y = np.logical_and(X[:, 0], X[:, 1]).astype(int)
    elif operation == "not":
        y = np.logical_not(X[:, 0]).astype(int)
    elif operation == "or":
        y = np.logical_or(X[:, 0], X[:, 1]).astype(int)
    elif operation == "xnor":
        y = np.logical_not(np.logical_xor(X[:, 0], X[:, 1])).astype(int)

    return X, y
