import pytest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from hornets import HorNetClassifier, generate_synthetic_data

NUM_INSTANCES = 128
NUM_FEATURES = 8
BATCH_SIZE = 16
NUM_RULES = 256
EPOCHS = 500


def get_data(problem):
    """
    Helper function for data generation.
    """
    X, y = generate_synthetic_data(
        num_instances=NUM_INSTANCES, num_features=NUM_FEATURES, operation=problem
    )
    return train_test_split(X, y, test_size=0.10, random_state=42)


def get_model():
    """
    Default model like in the paper.
    """
    model = HorNetClassifier(
        num_rules=NUM_RULES,
        exp_param=4,
        feature_names=[f"feature_{i}" for i in range(NUM_FEATURES)],
        activation="polyclip",
        order=5,
        learning_rate=0.1,
        batch_size=BATCH_SIZE,
        stopping_crit=100,
        comb_samples_fp=None,
        num_epochs=300,
        device="cpu",
        verbose=True,
    )
    return model


def test_xor():
    """
    Test the HorNetClassifier on the XOR operation.
    Ensures that the classifier achieves near-perfect accuracy.
    """
    X_train, X_test, y_train, y_test = get_data("xor")
    model = get_model()
    model.fit(X_train, y_train)
    acc = accuracy_score(model.predict(X_test), y_test)
    assert acc == pytest.approx(1.0)


def test_not():
    """
    Test the HorNetClassifier on the NOT operation.
    Ensures that the classifier achieves near-perfect accuracy.
    """
    X_train, X_test, y_train, y_test = get_data("not")
    model = get_model()
    model.fit(X_train, y_train)
    acc = accuracy_score(model.predict(X_test), y_test)

    assert acc == pytest.approx(1.0)


def test_and():
    """
    Test the HorNetClassifier on the AND operation.
    Ensures that the classifier achieves near-perfect accuracy.
    """
    X_train, X_test, y_train, y_test = get_data("and")
    model = get_model()
    model.fit(X_train, y_train)
    acc = accuracy_score(model.predict(X_test), y_test)

    assert acc == pytest.approx(1.0)
