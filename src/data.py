import numpy as np
import jax


def normalize_and_reshape(imgs):
    normalized = ((imgs / 255.0) - 0.5) / 0.5
    return jax.image.resize(normalized, shape=(len(normalized), 3, 224, 224), method="bilinear")


def import_data_efficient_mask():
    train_images = np.load("numpy_cifar100/train_images.npy")
    train_labels = np.load("numpy_cifar100/train_labels.npy")

    train_images = jax.device_put(train_images, device=jax.devices("cpu")[0])
    train_labels = jax.device_put(train_labels, device=jax.devices("cpu")[0])

    # Load test data

    test_images = np.load("numpy_cifar100/test_images.npy")
    test_labels = np.load("numpy_cifar100/test_labels.npy")

    test_images = jax.device_put(test_images, device=jax.devices("cpu")[0])
    test_labels = jax.device_put(test_labels, device=jax.devices("cpu")[0])

    return train_images, train_labels, test_images, test_labels
