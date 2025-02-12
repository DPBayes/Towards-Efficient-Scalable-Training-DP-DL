from collections import namedtuple

import jax
import jax.numpy as jnp
import optax
import pytest
from flax.training import train_state

from src.models import create_train_state, load_model


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
    logits = state.apply_fn(dummy_input, state.params)
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
    logits = model.apply({"params": params}, dummy_input)
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
