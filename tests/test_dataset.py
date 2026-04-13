import numpy as np
import pytest
import torch

from hornets.dataset import E2EDatasetLoader, generate_synthetic_data


class TestE2EDatasetLoader:
    def test_basic(self):
        X = np.random.rand(10, 5).astype(np.float32)
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        ds = E2EDatasetLoader(X, y)
        assert len(ds) == 10

    def test_getitem_with_targets(self):
        X = np.random.rand(5, 3).astype(np.float32)
        y = np.array([0, 1, 2, 0, 1])
        ds = E2EDatasetLoader(X, y)
        instance, target = ds[0]
        assert isinstance(instance, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        assert instance.dtype == torch.float32

    def test_getitem_without_targets(self):
        X = np.random.rand(5, 3).astype(np.float32)
        ds = E2EDatasetLoader(X, None)
        instance = ds[0]
        assert isinstance(instance, torch.Tensor)
        assert instance.shape == (3,)

    def test_length_mismatch_raises(self):
        X = np.random.rand(5, 3)
        y = np.array([0, 1, 2])
        with pytest.raises(ValueError, match="mismatch"):
            E2EDatasetLoader(X, y)


class TestGenerateSyntheticData:
    @pytest.mark.parametrize("op", ["xor", "and", "not", "or", "xnor"])
    def test_all_operations(self, op):
        X, y = generate_synthetic_data(num_features=8, num_instances=64, operation=op)
        assert X.shape == (64, 8)
        assert y.shape == (64,)
        assert set(np.unique(y)).issubset({0, 1})

    def test_invalid_operation(self):
        with pytest.raises(ValueError, match="Unsupported"):
            generate_synthetic_data(num_features=4, operation="nand")

    def test_reproducibility(self):
        X1, y1 = generate_synthetic_data(num_features=8, random_seed=42)
        X2, y2 = generate_synthetic_data(num_features=8, random_seed=42)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_different_seeds_differ(self):
        X1, _ = generate_synthetic_data(num_features=8, random_seed=42)
        X2, _ = generate_synthetic_data(num_features=8, random_seed=99)
        assert not np.array_equal(X1, X2)

    def test_no_global_seed_pollution(self):
        np.random.seed(123)
        before = np.random.rand()
        np.random.seed(123)
        generate_synthetic_data(num_features=4, random_seed=999)
        after = np.random.rand()
        assert before == after

    def test_binary_values_only(self):
        X, _ = generate_synthetic_data(num_features=16, num_instances=200)
        assert set(np.unique(X)).issubset({0, 1})

    def test_xor_correctness(self):
        X, y = generate_synthetic_data(num_features=4, num_instances=100)
        expected = np.logical_xor(X[:, 0], X[:, 1]).astype(int)
        np.testing.assert_array_equal(y, expected)
