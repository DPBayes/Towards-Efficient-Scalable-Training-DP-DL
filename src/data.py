import jax
from datasets import load_dataset

def normalize_and_reshape(imgs):
    normalized = ((imgs / 255.0) - 0.5) / 0.5
    return jax.image.resize(normalized, shape=(len(normalized), 3, 224, 224), method="bilinear")

def load_from_huggingface(dataset_name : str, cache_dir : str):
    """Load a dataset from huggingface.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset to be loaded.
    cache_dir : str
        The directory for caching the dataset.

    Returns
    -------
    train_images: jax.typing.ArrayLike
        The training images.
    train_labels: jax.typing.ArrayLike
        The training labels.
    test_images: jax.typing.ArrayLike
        The test images.
    test_labels: jax.typing.ArrayLike
        The training labels.
    """
    ds = load_dataset(dataset_name, cache_dir=cache_dir)
    ds = ds.with_format("jax")

    train_images = ds["train"]["img"]
    train_labels = ds["train"]["fine_label"]
    train_images = jax.device_put(train_images, device=jax.devices("cpu")[0])
    train_labels = jax.device_put(train_labels, device=jax.devices("cpu")[0])

    test_images = ds["test"]["img"]
    test_labels = ds["test"]["fine_label"]
    test_images = jax.device_put(test_images, device=jax.devices("cpu")[0])
    test_labels = jax.device_put(test_labels, device=jax.devices("cpu")[0])
    return train_images, train_labels, test_images, test_labels