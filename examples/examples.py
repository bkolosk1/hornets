import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from hornets import HorNetClassifier, generate_synthetic_data


def main():
    num_features = 3
    num_instances = 128
    operation = "xor"
    X, y = generate_synthetic_data(
        num_features=num_features, num_instances=num_instances, operation=operation
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    classifier = HorNetClassifier(
        num_rules=256,
        exp_param=4,
        feature_names=[f"feature_{i}" for i in range(X.shape[1])],
        activation="polyclip",
        order=5,
        learning_rate=0.1,
        batch_size=10,
        stopping_crit=100,
        num_epochs=300,
        verbose=True,
    )

    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
