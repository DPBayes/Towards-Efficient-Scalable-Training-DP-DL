import numpy as np
import jax

def import_data_efficient_mask():
    train_images = np.load(
        "numpy_cifar100/train_images.npy"
    )  # .to_device(device=jax.devices("cpu")[0]) Use this with JAX > 0.4.40, with 0.4.30 still doesn't exist
    train_labels = np.load("numpy_cifar100/train_labels.npy")  # .to_device(device=jax.devices("cpu")[0])

    train_images = jax.device_put(train_images, device=jax.devices("cpu")[0])
    train_labels = jax.device_put(train_labels, device=jax.devices("cpu")[0])

    # Load test data

    test_images = np.load("numpy_cifar100/test_images.npy")  # .to_device(device=jax.devices("cpu")[0])
    test_labels = np.load("numpy_cifar100/test_labels.npy")  # .to_device(device=jax.devices("cpu")[0])

    test_images = jax.device_put(test_images, device=jax.devices("cpu")[0])
    test_labels = jax.device_put(test_labels, device=jax.devices("cpu")[0])

    return train_images, train_labels, test_images, test_labels