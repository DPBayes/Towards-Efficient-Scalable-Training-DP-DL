import os
import numpy as np
import re
import jax
import flax.linen as nn
from transformers import FlaxViTForImageClassification
import models_flax


def transform_params(params, params_tf, num_classes):
    # BiT and JAX models have different naming conventions, so we need to
    # properly map TF weights to JAX weights
    params['root_block']['conv_root']['kernel'] = (
    params_tf['resnet/root_block/standardized_conv2d/kernel'])

    for block in ['block1', 'block2', 'block3', 'block4']:
        units = set([re.findall(r'unit\d+', p)[0] for p in params_tf.keys()
                        if p.find(block) >= 0])
        for unit in units:
            for i, group in enumerate(['a', 'b', 'c']):
                params[block][unit][f'conv{i+1}']['kernel'] = (
                    params_tf[f'resnet/{block}/{unit}/{group}/'
                            'standardized_conv2d/kernel'])
                params[block][unit][f'gn{i+1}']['bias'] = (
                    params_tf[f'resnet/{block}/{unit}/{group}/'
                            'group_norm/beta'][None, None, None])
                params[block][unit][f'gn{i+1}']['scale'] = (
                    params_tf[f'resnet/{block}/{unit}/{group}/'
                            'group_norm/gamma'][None, None, None])

            projs = [p for p in params_tf.keys()
                    if p.find(f'{block}/{unit}/a/proj') >= 0]
            assert len(projs) <= 1
            if projs:
                params[block][unit]['conv_proj']['kernel'] = params_tf[projs[0]]

    params['norm-pre-head']['bias'] = (
        params_tf['resnet/group_norm/beta'][None, None, None])
    params['norm-pre-head']['scale'] = (
        params_tf['resnet/group_norm/gamma'][None, None, None])

    params['conv_head']['kernel'] = np.zeros(
        (params['conv_head']['kernel'].shape[0], num_classes), dtype=np.float32)
    params['conv_head']['bias'] = np.zeros(num_classes, dtype=np.float32)


def load_model(rng,model_name,dimension,num_classes):
    print('load model name',model_name,flush=True)
    main_key, params_key= jax.random.split(key=rng,num=2)
    if model_name == 'small':
        class CNN(nn.Module):
            """A simple CNN model."""

            @nn.compact
            def __call__(self, x):
                x = nn.Conv(features=64, kernel_size=(7, 7),strides=2)(x)
                x = nn.relu(x)
                x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
                x = x.reshape((x.shape[0], -1))
                x = nn.Dense(features=256)(x)
                x = nn.relu(x)
                x = nn.Dense(features=100)(x)
                return x

        model = CNN()
        input_shape = (1,3,dimension,dimension)
        #But then, we need to split it in order to get random numbers
        
        #The init function needs an example of the correct dimensions, to infer the dimensions.
        #They are not explicitly writen in the module, instead, the model infer them with the first example.
        x = jax.random.normal(params_key, input_shape)

        main_rng, init_rng, dropout_init_rng = jax.random.split(main_key, 3)
        #Initialize the model
        variables = model.init({'params':init_rng},x)
        #variables = model.init({'params':main_key}, batch)
        model.apply(variables, x)
        return main_rng,model,variables['params']
    
    elif 'vit' in model_name:
        model = FlaxViTForImageClassification.from_pretrained(model_name, num_labels=num_classes, return_dict=False, ignore_mismatched_sizes=True)
        return main_key,model, model.params

    else:
        model = models_flax.KNOWN_MODELS['BiT-M-R50x1']
        bit_pretrained_dir = '/models_files/' # Change this with your directory. It might need the whole path, not the relative one.
        
        # Load weigths of a BiT model
        bit_model_file = os.path.join(bit_pretrained_dir, f'{model_name}.npz')
        if not os.path.exists(bit_model_file):
            raise FileNotFoundError(
            f'Model file is not found in "{bit_pretrained_dir}" directory.')
        with open(bit_model_file, 'rb') as f:
            params_tf = np.load(f)
            params_tf = dict(zip(params_tf.keys(), params_tf.values()))

        # Build ResNet architecture
        ResNet = model(num_classes = num_classes)

        x = jax.random.normal(params_key, (1, dimension, dimension, 3))

        main_rng, init_rng, dropout_init_rng = jax.random.split(main_key, 3)

        #Initialize the model
        variables = ResNet.init({'params':init_rng},x)

        params = variables['params']

        transform_params(params, params_tf,
            num_classes=num_classes)
        
        ResNet.apply({'params':params},x)
        
        return main_rng,ResNet,params