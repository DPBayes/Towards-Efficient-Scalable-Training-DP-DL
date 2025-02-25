import jax
import optax
from flax.core.frozen_dict import freeze
from flax import traverse_util
import flax.linen as nn
from transformers import FlaxViTForImageClassification
from collections import namedtuple
from flax.training import train_state


def create_train_state(
    model_name: str, num_classes: int, image_dimension: int, optimizer_config: namedtuple, layers_to_freeze=None
):
    """Creates initial `TrainState`."""
    rng, model, params, from_flax = load_model(jax.random.key(0), model_name, image_dimension, num_classes)

    if layers_to_freeze is None:
        tx = optax.adam(optimizer_config.learning_rate)
    else:
        # https://flax.readthedocs.io/en/v0.6.11/guides/transfer_learning.html
        params = freeze(params)
        partition_optimizers = {"trainable": optax.adam(optimizer_config.learning_rate), "frozen": optax.set_to_zero()}
        
        # the paths are joined by "." here so that we can just check if the layer_to_freeze is in the path (substring match)
        param_partitions = freeze(
            traverse_util.path_aware_map(
                lambda path, v: (
                    "frozen"
                    if any([layer_to_freeze in ".".join(path) for layer_to_freeze in layers_to_freeze])
                    else "trainable"
                ),
                params,
            )
        )
        tx = optax.multi_transform(partition_optimizers, param_partitions)

    if from_flax:
        return train_state.TrainState.create(apply_fn=model.__call__, params=params, tx=tx)
    else:
        return train_state.TrainState.create(
            apply_fn=lambda x, params: model.apply({"params": params}, x), params=params, tx=tx
        )


def load_model(rng, model_name, dimension, num_classes):
    print("load model name", model_name, flush=True)
    main_key, params_key = jax.random.split(key=rng, num=2)
    if model_name == "small":

        class CNN(nn.Module):
            """A simple CNN model."""

            @nn.compact
            def __call__(self, x):
                x = nn.Conv(features=64, kernel_size=(7, 7), strides=2)(x)
                x = nn.relu(x)
                x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
                x = x.reshape((x.shape[0], -1))
                x = nn.Dense(features=256)(x)
                x = nn.relu(x)
                x = nn.Dense(features=num_classes)(x)
                return (x,)

        model = CNN()
        input_shape = (1, 3, dimension, dimension)
        # But then, we need to split it in order to get random numbers

        # The init function needs an example of the correct dimensions, to infer the dimensions.
        # They are not explicitly writen in the module, instead, the model infer them with the first example.
        x = jax.random.normal(params_key, input_shape)

        main_rng, init_rng, dropout_init_rng = jax.random.split(main_key, 3)
        # Initialize the model
        variables = model.init({"params": init_rng}, x)
        # variables = model.init({'params':main_key}, batch)
        model.apply(variables, x)
        return main_rng, model, variables["params"], False

    elif "vit" in model_name:
        model = FlaxViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            return_dict=False,
            ignore_mismatched_sizes=True,
        )
        return main_key, model, model.params, True
