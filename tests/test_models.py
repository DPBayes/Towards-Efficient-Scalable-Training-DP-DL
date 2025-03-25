from collections import namedtuple

import jax
import jax.numpy as jnp
import optax
import pytest
from flax.training import train_state
from jaxdpopt.data import normalize_and_reshape, load_dataset, load_from_huggingface

from jaxdpopt.models import create_train_state, load_model


def test_create_train_state_small():
    """
    Tests that create_train_state creates a valid TrainState for the 'small' model,
    that the forward pass returns output with shape (batch_size, num_classes),
    and that the total number of parameters is correct.
    """
    optimizer_config = namedtuple("OptimizerConfig", ["learning_rate"])(learning_rate=0.001)
    image_dimension = 32
    num_classes = 10
    batch_size = 2

    # Check that the function returns a valid TrainState
    state = create_train_state("small", num_classes, image_dimension, optimizer_config)
    assert isinstance(state, train_state.TrainState)

    # Number of parameters
    total_params = sum(p.size for p in jax.tree_util.tree_leaves(state.params))
    expected_params = 234314
    assert total_params == expected_params, f"Expected {expected_params}, got {total_params}"

    # test model apply function
    rng = jax.random.PRNGKey(42)
    dummy_input = jax.random.normal(rng, (batch_size, 3, image_dimension, image_dimension))
    logits = state.apply_fn(dummy_input, state.params)[0]
    assert logits.shape == (batch_size, num_classes)


def test_load_model_small():
    """
    Tests that load_model returns valid model parameters
    and that the forward pass returns output with shape (batch_size, num_classes).
    """

    rng = jax.random.PRNGKey(0)
    image_dimension = 32
    num_classes = 10
    batch_size = 2

    main_rng, model, params, from_flax = load_model(rng, "small", image_dimension, num_classes)

    # check that it is not from Flax
    assert not from_flax

    # Number of parameters
    total_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    expected_params = 234314
    assert total_params == expected_params, f"Expected {expected_params}, got {total_params}"

    # test model apply function
    dummy_input = jax.random.normal(rng, (batch_size, 3, image_dimension, image_dimension))
    logits = model.apply({"params": params}, dummy_input)[0]
    assert logits.shape == (batch_size, num_classes)


def test_create_train_state_vit():
    """Tests that create_train_state can create a valid TrainState for a ViT model, the number of params is correct,
    and that the forward pass returns logits with shape (batch_size, num_classes).
    """
    optimizer_config = namedtuple("OptimizerConfig", ["learning_rate"])(learning_rate=0.001)
    image_dimension = 224
    num_classes = 10
    batch_size = 2

    # Check that the function returns a valid TrainState
    state = create_train_state("google/vit-base-patch16-224", num_classes, image_dimension, optimizer_config)
    assert isinstance(state, train_state.TrainState)

    # Total number of parameters
    num_params = sum(p.size for p in jax.tree_util.tree_leaves(state.params))
    assert num_params == 85806346

    # Check that the forward pass returns logits with the correct shape
    dummy_input = jax.random.normal(jax.random.PRNGKey(42), (batch_size, 3, image_dimension, image_dimension))
    logits = state.apply_fn(dummy_input, state.params)[0]
    assert logits.shape == (batch_size, num_classes)


def test_create_train_state_freeze_layers():
    """
    Tests that parameters in layers specified in layers_to_freeze remain unchanged after an update,
    while other layers are modified.
    """

    optimizer_config = namedtuple("OptimizerConfig", ["learning_rate"])(learning_rate=0.1)
    image_dimension = 32
    num_classes = 10
    frozen_layers = ["Conv"]
    state = create_train_state("small", num_classes, image_dimension, optimizer_config, layers_to_freeze=frozen_layers)

    dummy_grads = jax.tree.map(lambda x: jnp.ones_like(x), state.params)
    state_after_update = state.apply_gradients(grads=dummy_grads)

    for layer, params in state.params.items():
        if any(freeze_key in layer for freeze_key in frozen_layers):
            for subkey in params.keys():
                assert jnp.allclose(
                    state_after_update.params[layer][subkey], params[subkey]
                ), f"Layer {layer} was updated although it should be frozen."
        else:
            for subkey in params.keys():
                assert not jnp.allclose(
                    state_after_update.params[layer][subkey], params[subkey]
                ), f"Layer {layer} was not updated although it should be trainable."


def test_models_outputs():

    lr = 1e-3
    num_steps = 120

    train_images, train_labels, test_images, test_labels = load_from_huggingface(
        "uoft-cs/cifar10", cache_dir=None, feature_name="img"
    )
    ORIG_IMAGE_DIMENSION, RESIZED_IMAGE_DIMENSION = 32, 32
    train_images = (
        train_images[train_labels < 2]
        .transpose(0, 3, 1, 2)
        .reshape(-1, 1, 3, ORIG_IMAGE_DIMENSION, ORIG_IMAGE_DIMENSION)
    )
    train_labels = train_labels[train_labels < 2]
    test_images = test_images[test_labels < 2].transpose(0, 3, 1, 2)
    test_labels = test_labels[test_labels < 2]
    batch_size = 100

    num_classes = 2
    dataset_size = len(train_labels)

    optimizer_config = namedtuple("Config", ["learning_rate"])
    optimizer_config.learning_rate = lr

    state_small = create_train_state(
        model_name="small",
        num_classes=num_classes,
        image_dimension=RESIZED_IMAGE_DIMENSION,
        optimizer_config=optimizer_config,
    )

    state_vit = create_train_state(
        model_name='google/vit-base-patch16-224-in21k',
        num_classes=num_classes,
        image_dimension=RESIZED_IMAGE_DIMENSION,
        optimizer_config=optimizer_config,
    )


    logits = state_small.apply_fn(test_images[:10], params=state_small.params)
    print(logits)
    
    logits_vit = state_vit.apply_fn(normalize_and_reshape(test_images[:10]), params=state_vit.params)
    print(logits_vit)

    assert isinstance(logits,tuple) and isinstance(logits_vit,tuple)
    assert len(logits[0]) == len(logits_vit[0])
    