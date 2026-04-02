import numpy as np
import pytest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from hornets import HorNetClassifier, generate_synthetic_data

NUM_INSTANCES = 128
NUM_FEATURES = 8
BATCH_SIZE = 16
NUM_RULES = 256


def get_data(problem="xor"):
    X, y = generate_synthetic_data(
        num_instances=NUM_INSTANCES, num_features=NUM_FEATURES, operation=problem
    )
    return train_test_split(X, y, test_size=0.10, random_state=42)


def get_model(**overrides):
    defaults = dict(
        num_rules=NUM_RULES,
        exp_param=4,
        feature_names=[f"feature_{i}" for i in range(NUM_FEATURES)],
        activation="polyclip",
        order=5,
        learning_rate=0.1,
        batch_size=BATCH_SIZE,
        stopping_crit=100,
        comb_samples_fp=48,
        num_epochs=300,
        device="cpu",
        verbose=False,
    )
    defaults.update(overrides)
    return HorNetClassifier(**defaults)


class TestFit:
    def test_fit_returns_self(self):
        X_train, _, y_train, _ = get_data()
        model = get_model(num_epochs=5)
        result = model.fit(X_train, y_train)
        assert result is model

    def test_model_attributes_after_fit(self):
        X_train, _, y_train, _ = get_data()
        model = get_model(num_epochs=5)
        model.fit(X_train, y_train)
        assert model.model_ is not None
        assert model.classes_ is not None
        assert len(model.losses_) > 0

    def test_losses_recorded(self):
        X_train, _, y_train, _ = get_data()
        model = get_model(num_epochs=10, stopping_crit=100)
        model.fit(X_train, y_train)
        assert len(model.losses_) == 10

    def test_early_stopping(self):
        X_train, _, y_train, _ = get_data()
        model = get_model(num_epochs=1000, stopping_crit=5, learning_rate=1e-8)
        model.fit(X_train, y_train)
        assert len(model.losses_) < 1000

    def test_feature_names_generated(self):
        X_train, _, y_train, _ = get_data()
        model = get_model(num_epochs=5, feature_names=None)
        model.fit(X_train, y_train)
        assert model.feature_names_ is not None
        assert len(model.feature_names_) == NUM_FEATURES + model.order


class TestPredict:
    @pytest.mark.parametrize("op", ["xor", "not"])
    def test_accuracy(self, op):
        X_train, X_test, y_train, y_test = get_data(op)
        model = get_model()
        model.fit(X_train, y_train)
        acc = accuracy_score(model.predict(X_test), y_test)
        assert acc >= 0.90

    @pytest.mark.parametrize("op", ["and", "or"])
    def test_accuracy_relaxed(self, op):
        X_train, X_test, y_train, y_test = get_data(op)
        model = get_model(num_epochs=500)
        model.fit(X_train, y_train)
        acc = accuracy_score(model.predict(X_test), y_test)
        assert acc >= 0.60

    def test_predict_shape(self):
        X_train, X_test, y_train, _ = get_data()
        model = get_model(num_epochs=5)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert len(preds) == len(X_test)

    def test_predict_values_are_valid_classes(self):
        X_train, X_test, y_train, _ = get_data()
        model = get_model(num_epochs=5)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert set(preds).issubset(set(model.classes_))

    def test_predict_before_fit_raises(self):
        model = get_model()
        X = np.random.rand(5, NUM_FEATURES)
        with pytest.raises(Exception):
            model.predict(X)


class TestPredictProba:
    def test_shape(self):
        X_train, X_test, y_train, _ = get_data()
        model = get_model(num_epochs=5)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)
        assert proba.shape == (len(X_test), len(model.classes_))

    def test_probabilities_sum_to_one(self):
        X_train, X_test, y_train, _ = get_data()
        model = get_model(num_epochs=5)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)
        sums = proba.sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-5)

    def test_probabilities_non_negative(self):
        X_train, X_test, y_train, _ = get_data()
        model = get_model(num_epochs=5)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)
        assert np.all(proba >= 0)


class TestGetTopRules:
    def test_returns_list(self):
        X_train, _, y_train, _ = get_data()
        model = get_model(num_epochs=20)
        model.fit(X_train, y_train)
        rules = model.get_top_rules(top_k=3)
        assert isinstance(rules, list)
        assert len(rules) == 3

    def test_rule_structure(self):
        X_train, _, y_train, _ = get_data()
        model = get_model(num_epochs=20)
        model.fit(X_train, y_train)
        rules = model.get_top_rules(top_k=1)
        assert "features" in rules[0]
        assert "score" in rules[0]


class TestReproducibility:
    def test_same_seed_same_result(self):
        X_train, X_test, y_train, _ = get_data()
        model1 = get_model(num_epochs=20, random_state=42)
        model1.fit(X_train, y_train)
        pred1 = model1.predict(X_test)

        model2 = get_model(num_epochs=20, random_state=42)
        model2.fit(X_train, y_train)
        pred2 = model2.predict(X_test)

        np.testing.assert_array_equal(pred1, pred2)


class TestEdgeCases:
    def test_single_sample_predict(self):
        X_train, _, y_train, _ = get_data()
        model = get_model(num_epochs=5)
        model.fit(X_train, y_train)
        single = X_train[:1]
        pred = model.predict(single)
        assert len(pred) == 1

    def test_large_batch_size(self):
        X_train, X_test, y_train, _ = get_data()
        model = get_model(num_epochs=5, batch_size=64)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        assert len(pred) == len(X_test)
