import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from pathlib import Path


COMPONENT_COUNTS = [10, 20, 30, 50, 75, 100, 150, 200, 250, 300]
K_VALUES = [1, 3, 5]


def save_figure(fig, output_dir, filename):
    fig.savefig(output_dir / filename)
    plt.close(fig)


def prepare_data(seed, output_dir):
    people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
    rng = np.random.default_rng(seed)
    selected = []
    for person_id in np.unique(people.target):
        person_indices = np.flatnonzero(people.target == person_id)
        selected.extend(rng.choice(person_indices, size=20, replace=False))

    selected = np.asarray(selected)
    images = people.images[selected]
    y = people.target[selected]
    class_count = len(people.target_names)
    before = np.bincount(people.target, minlength=class_count)
    after = np.bincount(y, minlength=class_count)

    fig, axes = plt.subplots(2, 1, figsize=(16, 11), sharex=True)
    positions = np.arange(class_count)
    axes[0].bar(positions, before)
    axes[0].set_title("Images per person before balancing")
    axes[1].bar(positions, after)
    axes[1].set_title("Images per person after balancing (20 each)")
    axes[1].set_xticks(positions, people.target_names, rotation=90, fontsize=7)
    for ax in axes:
        ax.set_ylabel("Number of images")
    fig.tight_layout()
    save_figure(fig, output_dir, "01_class_counts.png")

    train_indices, test_indices = train_test_split(
        np.arange(len(images)), test_size=0.25, stratify=y, random_state=seed
    )
    return people, images, y, train_indices, test_indices


def run_pca(X_train):
    plain = PCA(svd_solver="full", whiten=False)
    white = PCA(svd_solver="full", whiten=True)
    plain.fit(X_train)
    white.fit(X_train)
    return plain, white


def pca_result(plain, white, X_train, X_test, y_train, y_test,
               target_names, image_shape, output_dir):
    plain_scores = plain.transform(X_train)
    white_scores = white.transform(X_train)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, scores, title in zip(
        axes, [plain_scores, white_scores], ["Without whitening", "With whitening"]
    ):
        ax.scatter(scores[:, 0], scores[:, 1], c=y_train, s=12)
        ax.set_title(title)
        ax.set_xlabel("Principal component 1")
        ax.set_ylabel("Principal component 2")
        ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    save_figure(fig, output_dir, "02_whitening_scatter.png")

    fig, axes = plt.subplots(3, 4, figsize=(9, 9))
    for index, ax in enumerate(axes.flat):
        ax.imshow(plain.components_[index].reshape(image_shape), cmap="gray")
        ax.set_title(f"Eigenface {index + 1}")
        ax.axis("off")
    fig.tight_layout()
    save_figure(fig, output_dir, "03_eigenfaces.png")

    reconstruction_counts = [10, 30, 50, 100, 200, 300]
    example_indices = np.unique(y_test, return_index=True)[1][:4]
    examples = X_test[example_indices]
    example_scores = plain.transform(examples)
    fig, axes = plt.subplots(len(examples), len(reconstruction_counts) + 1,
                             figsize=(15, 9))
    for row, example_index in enumerate(example_indices):
        axes[row, 0].imshow(examples[row].reshape(image_shape), cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title("Original\n" + target_names[y_test[example_index]], fontsize=8)
        axes[row, 0].axis("off")
        for col, count in enumerate(reconstruction_counts, start=1):
            restored = example_scores[row, :count] @ plain.components_[:count] + plain.mean_
            axes[row, col].imshow(restored.reshape(image_shape), cmap="gray", vmin=0, vmax=1)
            axes[row, col].set_title(f"{count} components")
            axes[row, col].axis("off")
    fig.tight_layout()
    save_figure(fig, output_dir, "04_reconstructions.png")

    cumulative = np.cumsum(plain.explained_variance_ratio_)
    n90 = int(np.searchsorted(cumulative, 0.90) + 1)
    n95 = int(np.searchsorted(cumulative, 0.95) + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(np.arange(1, len(cumulative) + 1), cumulative)
    for threshold, count in [(0.90, n90), (0.95, n95)]:
        ax.axhline(threshold, linestyle="--")
        ax.axvline(count, linestyle=":", label=f"{threshold:.0%}: {count} components")
    ax.set(xlabel="Number of principal components", ylabel="Cumulative explained variance",
           title="PCA explained variance", ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    save_figure(fig, output_dir, "05_cumulative_variance.png")


def knn_sweep(pca_models, X_train, X_test, y_train, y_test):
    rows = []
    for whitening, pca in zip([False, True], pca_models):
        train_scores = pca.transform(X_train)
        test_scores = pca.transform(X_test)
        for count in COMPONENT_COUNTS:
            for k in K_VALUES:
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(train_scores[:, :count], y_train)
                predictions = knn.predict(test_scores[:, :count])
                accuracy = accuracy_score(y_test, predictions)
                rows.append({"components": count, "whitening": whitening,
                             "k": k, "accuracy": accuracy})
    return rows


def run_knn(pca_models, X, y, train_indices, test_indices, output_dir):
    rows = knn_sweep(pca_models, X[train_indices], X[test_indices],
                     y[train_indices], y[test_indices])
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, k in zip(axes, K_VALUES):
        for whitening in [False, True]:
            results = [row for row in rows if row["k"] == k and row["whitening"] == whitening]
            ax.plot([row["components"] for row in results], [row["accuracy"] for row in results],
                    label="Whitening" if whitening else "No whitening")
        ax.set(title=f"KNN: k={k}", xlabel="Number of principal components", ylim=(0, 1))
        ax.legend()
    axes[0].set_ylabel("Test accuracy")
    fig.tight_layout()
    save_figure(fig, output_dir, "06_knn_accuracy.png")
    return rows


def run_cnn(images, y, train_indices, test_indices, seed, epochs, batch_size, output_dir):
    from tensorflow import keras
    keras.utils.set_random_seed(seed)
    cnn_images = images[..., np.newaxis]
    if cnn_images.max() > 1.0:
        cnn_images = cnn_images / 255.0
    class_count = len(np.unique(y))
    fit_indices, val_indices = train_test_split(
        train_indices, test_size=0.20, stratify=y[train_indices], random_state=seed
    )
    layers = keras.layers
    model_layers = [layers.Input(shape=cnn_images.shape[1:])]
    for filters in [16, 32, 64]:
        model_layers.extend([
            layers.Conv2D(filters, (3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D((2, 2))
        ])
    model_layers.extend([
        layers.Flatten(), layers.Dense(128, activation="relu"),
        layers.Dense(class_count, activation="softmax")
    ])
    model = keras.Sequential(model_layers)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(
        cnn_images[fit_indices], y[fit_indices],
        validation_data=(cnn_images[val_indices], y[val_indices]),
        epochs=epochs, batch_size=batch_size
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    epoch_numbers = np.arange(1, epochs + 1)
    for ax, metric, label in zip(axes, ["accuracy", "loss"], ["Accuracy", "Loss"]):
        ax.plot(epoch_numbers, history.history[metric], label="Training")
        ax.plot(epoch_numbers, history.history["val_" + metric], label="Validation")
        ax.set(title=f"CNN {label.lower()}", xlabel="Epoch", ylabel=label)
        ax.legend()
    fig.tight_layout()
    save_figure(fig, output_dir, "07_cnn_learning_curves.png")

    _, test_accuracy = model.evaluate(cnn_images[test_indices], y[test_indices])
    return test_accuracy

def main():
    seed = 1557
    epochs = 40
    batch_size = 32
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    people, images, y, train_indices, test_indices = prepare_data(seed, output_dir)
    X = images.reshape(len(images), -1)
    pca_models = run_pca(X[train_indices])
    pca_result(
        *pca_models, X[train_indices], X[test_indices], y[train_indices], y[test_indices],
        people.target_names, images.shape[1:], output_dir
    )
    knn_results = run_knn(pca_models, X, y, train_indices, test_indices, output_dir)
    cnn_accuracy = run_cnn(images, y, train_indices, test_indices,
                           seed, epochs, batch_size, output_dir)

    knn_result = next(row for row in knn_results
                      if row["components"] == 50 and row["k"] == 1 and row["whitening"])
    accuracies = [knn_result["accuracy"], cnn_accuracy]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(["PCA + KNN", "CNN"], accuracies,
                  color=["tab:blue", "tab:orange"])
    ax.bar_label(bars, labels=[f"{accuracy:.2%}" for accuracy in accuracies], padding=4)
    ax.set(ylabel="Test accuracy", ylim=(0, 1), title="Same test split: PCA + KNN vs. CNN")
    ax.set_xlabel(
        f"KNN: {knn_result['components']} components, k={knn_result['k']}, whitening=yes\n"
        f"CNN: {epochs} epochs", fontsize=9
    )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, output_dir, "08_final_accuracy_comparison.png")


if __name__ == "__main__":
    main()
