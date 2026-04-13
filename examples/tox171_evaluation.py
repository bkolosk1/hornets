"""
Realistic evaluation of HorNets on the TOX_171 dataset.

TOX_171: 171 samples, 5748 gene expression features, 4 toxicity classes.
Source: https://jundongl.github.io/scikit-feature/datasets.html

Runs 5-fold cross-validation, reports per-fold accuracy,
and visualizes train+test embeddings with t-SNE, UMAP, and PCA for each fold.
"""

import os
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import umap

from hornets import HorNetClassifier

DATA_URL = "https://jundongl.github.io/scikit-feature/files/datasets/TOX_171.mat"
DATA_PATH = os.path.join(os.path.dirname(__file__), "TOX_171.mat")


def load_tox171():
    if not os.path.exists(DATA_PATH):
        print("Downloading TOX_171.mat ...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    mat = scipy.io.loadmat(DATA_PATH)
    X = np.asarray(mat["X"], dtype=np.float64)
    y = mat["Y"].ravel().astype(int)
    return X, y


def make_classifier():
    return HorNetClassifier(
        num_rules=256,
        exp_param=2,
        activation="polyclip",
        order=3,
        learning_rate=0.01,
        batch_size=16,
        stopping_crit=30,
        comb_samples_fp=None,
        num_epochs=300,
        device="cpu",
        verbose=False,
    )


def reduce_2d(embeddings, method):
    if method == "t-SNE":
        return TSNE(
            n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1)
        ).fit_transform(embeddings)
    elif method == "UMAP":
        return umap.UMAP(
            n_components=2, random_state=42, n_neighbors=min(15, len(embeddings) - 1)
        ).fit_transform(embeddings)
    elif method == "PCA":
        return PCA(n_components=2, random_state=42).fit_transform(embeddings)


def plot_fold(ax, coords, y_vals, split_sizes, classes, fold, method):
    cmap = plt.cm.Set1
    n_train = split_sizes[0]
    train_coords, test_coords = coords[:n_train], coords[n_train:]
    y_train, y_test = y_vals[:n_train], y_vals[n_train:]

    for cls in classes:
        # train points
        mask_tr = y_train == cls
        ax.scatter(
            train_coords[mask_tr, 0], train_coords[mask_tr, 1],
            c=[cmap(cls / max(classes))], marker="o", alpha=0.6, s=30,
            label=f"Cls {cls} train",
        )
        # test points (black edge)
        mask_te = y_test == cls
        ax.scatter(
            test_coords[mask_te, 0], test_coords[mask_te, 1],
            c=[cmap(cls / max(classes))], marker="^", alpha=0.9, s=50,
            edgecolors="black", linewidths=0.6,
            label=f"Cls {cls} test",
        )

    ax.set_title(f"Fold {fold + 1} — {method}", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    X, y = load_tox171()
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accs = []
    classes = np.unique(y)
    methods = ["t-SNE", "UMAP", "PCA"]

    fig, axes = plt.subplots(5, 3, figsize=(14, 20))
    fig.suptitle("HorNets Embeddings on TOX_171", fontsize=14, y=1.0)

    # Column headers
    for j, method in enumerate(methods):
        axes[0, j].set_title(f"Fold 1 — {method}", fontsize=9)

    all_y_true, all_y_pred = [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = make_classifier()
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        fold_accs.append(acc)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        emb_train = clf.transform(X_train)
        emb_test = clf.transform(X_test)
        all_emb = np.vstack([emb_train, emb_test])
        all_y = np.concatenate([y_train, y_test])

        for j, method in enumerate(methods):
            coords = reduce_2d(all_emb, method)
            plot_fold(
                axes[fold, j], coords, all_y,
                (len(emb_train),), classes, fold, method
            )

        print(f"Fold {fold + 1}: accuracy = {acc:.4f}")

    # Shared legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(classes) * 2, fontsize=7)

    print(f"\nMean accuracy: {np.mean(fold_accs):.4f} +/- {np.std(fold_accs):.4f}")
    print("\nOverall classification report:")
    print(classification_report(all_y_true, all_y_pred, target_names=[f"Class {c}" for c in classes]))

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    out_path = os.path.join(os.path.dirname(__file__), "tox171_embeddings.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved embedding visualization to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
