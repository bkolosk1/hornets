# HorNets: Learning from Discrete and Continuous Signals with Routing Neural Networks

![build](https://github.com/bkolosk1/hornets/actions/workflows/python-install.yml/badge.svg)  ![lint](https://github.com/bkolosk1/hornets/actions/workflows/lint.yml/badge.svg) ![test](https://github.com/bkolosk1/hornets/actions/workflows/pytest.yml/badge.svg) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HxBQRLPa-j54WYU6nxj9VgwLv1ywfzJ-?usp=sharing)




**HorNets** is a Python package implementing the HorNets architecture of the ``HorNets: Learning from Discrete and Continuous Signals with Routing Neural Networks`` paper.


![alt text](image.png)


## Installation

Follow the instructions below to install **HorNets** using your preferred method.

### Using PyPI ``stable version``

You can install **HorNets** directly from PyPI using `pip`. This is the simplest method if you just want to use the package.

```bash
pip install hornets
```

### Using PyPI ``latest version``

```bash
pip install git+https://github.com/bkolosk1/hornets.git
```

### Local Development Installation

To install locally with `poetry` follow the following steps:

1. Install Poetry: `pip install poetry`  
2. Clone the repo: `git clone git@github.com:bkolosk1/hornets.git && cd hornets`  
3. Run: `poetry install`
4. Test the installation with: `poetry run python examples/examples.py`


## Usage


You can run `examples/examples.py` 

Or you can run the minimal code snippet:


```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from hornets import HorNetClassifier, generate_synthetic_data

# Generate synthetic data
X, y = generate_synthetic_data(
    num_features=64,
    num_instances=128,
    operation="xor"
)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize and train HorNetClassifier
classifier = HorNetClassifier(
    num_rules=256,
    exp_param=4,
    activation="polyclip",
    order=5,
    learning_rate=0.1,
    batch_size=10,
    stopping_crit=100,
    num_epochs=500,
    verbose=True
)
classifier.fit(X_train, y_train)

# Predict on test data
y_pred = classifier.predict(X_test)

# Evaluate
print("Classification Report:")
print(classification_report(y_test, y_pred))
```




## Hyperparameters


| **Parameter**     | **Type**                | **Default** | **Description**                                                                                                                                                                                                                                                                                  | **Suggested HPO Range**              |
|-------------------|-------------------------|------------:|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------|
| `batch_size`      | `int`                  | 4           | Number of samples per training batch.                                                                                                                                             | 4, 8, 16, 32                         |
| `num_epochs`      | `int`                  | 1000        | Maximum number of epochs (complete passes through the training data). Large values allow more training time but risk overfitting.                                                                                                 | 100, 300, 500, 1000                  |
| `learning_rate`   | `float`                | 1e-4        | Learning rate for the optimizer. Higher values speed up training but risk overshooting minima; lower values may stabilize training but at the cost of slower convergence.                                                                                                                        | 1e-5 to 1e-2                         |
| `stopping_crit`   | `int`                  | 10          | Number of consecutive epochs without improvement on validation metric to wait before stopping training early.                                                                                              | 5, 10, 15                            |
| `feature_names`   | `Optional[List[str]]`  | None        | Names of input features.                                                                                                                                                                                    | Not applicable                       |
| `num_rules`       | `int`                  | 256         | Number of rules (feature combinations) in the HorNet model. Increasing this can capture more complex patterns but may lead to higher risk of overfitting or longer training times.                                                                                                               | 64, 128, 256, 512                    |
| `activation`      | `str`                  | "polyclip"  | Activation function used in the model. Options include `"polyclip"` or `"relu"` functions.                                                 | "polyclip", "relu", "sigmoid", etc.  |
| `comb_samples_fp` | `int`                  | 48          | Number of combination samples for feature processing. Larger values may capture more interaction patterns but at higher computational cost.                                                                                                              | 16, 32, 48, 64                       |
| `exp_param`       | `int`                  | 1           | Expansion parameter for polynomial clipping within the activation. Primarily relevant for `"polyclip"` activations, controlling how inputs are processed.                                                                                                                                        | 1, 2, 3                              |
| `order`           | `int`                  | 5           | Order of feature combinations. Higher orders enable capturing more complex relationships but can drastically increase computational demands.                                                                                                                                                     | 2, 3, 5                              |
| `device`          | `str`                  | "cpu"       | Device for model training and inference. Using `"cuda"` (if available) can speed up training significantly, but `"cpu"` is a safer default.                                                                                                                                                      | "cpu", "cuda"                        |
| `random_state`    | `Optional[int]`        | None        | Seed for random number generation, ensuring reproducible results. Not usually tuned as a hyperparameter, but important for experiments where repeatability is required.                                                                                                                          | Not applicable                       |
| `verbose`         | `bool`                 | False       | If True, prints progress and debug messages. Can be helpful during development or troubleshooting but is not generally part of hyperparameter optimization.                                                                                                                                       | True or False                        |



## Citation



Cite this work as:

```
@misc{koloski2025hornetslearningdiscretecontinuous,
      title={HorNets: Learning from Discrete and Continuous Signals with Routing Neural Networks}, 
      author={Boshko koloski and Nada Lavrač and Blaž Škrlj},
      year={2025},
      eprint={2501.14346},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2501.14346}, 
}
```
