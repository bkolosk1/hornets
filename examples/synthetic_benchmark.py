from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import trange

from hornets import HorNetClassifier, generate_synthetic_data


def synthetic_run_setup():
    probs = ["xor", "or", "and", "not", "xnor"]
    nft = [64, 128]  # 2, 3, 4, 8, 16, 32,
    num_instances = 128
    reps = 10
    synthetic_samples_results = []
    for num_features in nft:
        for iteration in trange(reps):
            for problem in probs:
                X, y = generate_synthetic_data(
                    num_features, num_instances, operation=problem
                )
                problem = problem + "(dim=" + str(num_features) + ")"

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.10, random_state=42
                )

                classifier = HorNetClassifier(
                    num_rules=256,
                    exp_param=4,
                    feature_names=[f"feature_{i}" for i in range(X.shape[1])],
                    activation="polyclip",
                    order=5,
                    learning_rate=0.1,
                    batch_size=16,
                    stopping_crit=100,
                    num_epochs=300,
                    verbose=True,
                )

                classifier.fit(X_train, y_train)

                y_pred = classifier.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                synthetic_samples_results.append(
                    {
                        "num_features": num_features,
                        "iter": iteration,
                        "problem": problem,
                        "score": acc,
                    }
                )

                print(
                    "Num Features",
                    num_features,
                    "Num Instances",
                    num_instances,
                    "Problem",
                    problem,
                    ":",
                    acc,
                )


synthetic_run_setup()
