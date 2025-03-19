import jax
from datasets import load_dataset


def normalize_and_reshape(imgs):
    normalized = ((imgs / 255.0) - 0.5) / 0.5
    return jax.image.resize(normalized, shape=(len(normalized), 3, 224, 224), method="bilinear")

def normalize_and_reshape_(imgs,scale,mean,std,shape):
    normalized_data = ((imgs/scale) - mean)/std
    return jax.image.resize(normalized_data, shape = shape, method = 'bilinear')


def load_from_huggingface(dataset_name: str, cache_dir: str, feature_name="img", label_name="label"):
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

    train_images = ds["train"][feature_name]
    train_labels = ds["train"][label_name]
    train_images = jax.device_put(train_images, device=jax.devices("cpu")[0])
    train_labels = jax.device_put(train_labels, device=jax.devices("cpu")[0])

    test_images = ds["test"][feature_name]
    test_labels = ds["test"][label_name]
    test_images = jax.device_put(test_images, device=jax.devices("cpu")[0])
    test_labels = jax.device_put(test_labels, device=jax.devices("cpu")[0])
    return train_images, train_labels, test_images, test_labels


def prepare_sharding():
    """Prepare the JAX sharding objects to distribute the data and the model.

    Returns
    -------
    mesh:   jax.sharding.Mesh
        Describes how the devices are arranged. It is returned since it is necessary
        for the main sharding map.
    data_shard:  jax.sharding.NamedSharding
        A named sharding object, describes how to shard the data across devices.
        Data will be sharded without the need of explicitly dividing it. It will
        shard over the first dimension, which in general is the batch dimension.
    model_shard:  jax.sharding.NamedSharding
        A named sharding object, describes how to shard the model across devices
        The model will be replicated on each device. So far we don't have a feature
        of sharding parts of the model. To shard across devices, the PartitionSpec
        object must have an axis. If it is empty, it will replicate over the mesh.
    """
    mesh = jax.sharding.Mesh(jax.devices(), "devices")

    data_shard = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("devices"))
    model_shard = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    return mesh, data_shard, model_shard
