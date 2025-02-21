import jax
import flax.linen as nn
from transformers import FlaxViTForImageClassification


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
                x = nn.Dense(features=100)(x)
                return x

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
        return main_rng, model, variables["params"]

    elif "vit" in model_name:
        model = FlaxViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            return_dict=False,
            ignore_mismatched_sizes=True,
        )
        return main_key, model, model.params
