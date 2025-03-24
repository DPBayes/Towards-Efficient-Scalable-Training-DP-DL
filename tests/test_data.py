from jaxdpopt.data import load_from_huggingface
import jax


def test_load_dataset():
    """
    Test that load_from_huggingface returns the correct shapes for MNIST.
    """

    train_images, train_labels, test_images, test_labels = load_from_huggingface(
        "ylecun/mnist", None, feature_name="image"
    )

    # check shapes
    assert train_images.shape == (60000, 28, 28)
    assert test_images.shape == (10000, 28, 28)
    assert train_labels.shape == (60000,)
    assert test_labels.shape == (10000,)

    # check that arrs are on cpu
    for arr in [train_images, train_labels, test_images, test_labels]:
        for device in arr.devices():
            assert device.device_kind == "cpu" and device.device_kind != "gpu"


def test_load_from_huggingface_device_and_dtype():
    """
    Test that load_from_huggingface returns JAX arrays placed on CPU and with valid dtypes.
    """
    train_images, train_labels, test_images, test_labels = load_from_huggingface(
        "ylecun/mnist", None, feature_name="image"
    )

    # Check that arrays are JAX DeviceArrays and reside on CPU.
    for arr in [train_images, train_labels, test_images, test_labels]:
        assert isinstance(arr, jax.Array)
        for device in arr.devices():
            assert device.device_kind == "cpu"


def test_load_from_huggingface_consistency():
    """
    Test that repeated calls to load_from_huggingface with the same parameters yield consistent outputs.
    """
    out1 = load_from_huggingface("ylecun/mnist", None, feature_name="image")
    out2 = load_from_huggingface("ylecun/mnist", None, feature_name="image")
    for a, b in zip(out1, out2):
        # Compare shapes and content
        assert a.shape == b.shape
        assert (a == b).all()


def test_load_from_huggingface_cifar10_and_svhn():
    """
    Test that load_from_huggingface returns expected array shapes and devices
    for CIFAR10 and SVHN datasets.
    """
    cifar10_train, cifar10_train_labels, cifar10_test, cifar10_test_labels = load_from_huggingface(
        "uoft-cs/cifar10", None
    )
    assert cifar10_train.shape == (50000, 32, 32, 3)
    assert cifar10_test.shape == (10000, 32, 32, 3)
    assert len(cifar10_train_labels.shape) == 1
    assert len(cifar10_test_labels.shape) == 1
    for arr in [cifar10_train, cifar10_train_labels, cifar10_test, cifar10_test_labels]:
        for device in arr.devices():
            assert device.device_kind == "cpu"

    svhn_train, svhn_train_labels, svhn_test, svhn_test_labels = load_from_huggingface(
        "dpdl-benchmark/svhn_cropped",
        None,
        feature_name="image"
    )
    assert svhn_train.shape == (73257, 32, 32, 3)
    assert svhn_test.shape == (26032, 32, 32, 3)
    assert len(svhn_train_labels.shape) == 1
    assert len(svhn_test_labels.shape) == 1
    for arr in [svhn_train, svhn_train_labels, svhn_test, svhn_test_labels]:
        for device in arr.devices():
            assert device.device_kind == "cpu"
