from data import load_from_huggingface


def test_load_dataset():
    train_images, train_labels, test_images, test_labels = load_from_huggingface("mnist", None, feature_name="image")

    # check shapes
    assert train_images.shape == (60000, 28, 28)
    assert test_images.shape == (10000, 28, 28)
    assert train_labels.shape == (60000,)
    assert test_labels.shape == (10000,)

    # check that arrs are on cpu
    for arr in [train_images, train_labels, test_images, test_labels]:
        for device in arr.devices():
            assert device.device_kind == "cpu" and device.device_kind != "gpu"
