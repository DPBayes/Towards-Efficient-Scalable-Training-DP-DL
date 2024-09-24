import jax
import optax
import jax.numpy as jnp
import numpy as np
from flax.training import train_state
import torch
from typing import List
from torch.utils.data import Sampler, Dataset
from scipy.stats import binom
import torchvision
import math
import flax.linen as nn
from transformers import FlaxViTModel
from private_vit import ViTModelHead


@jax.jit
def compute_per_example_gradients(state, batch_X, batch_y):
    """Computes gradients, loss and accuracy for a single batch."""

    def loss_fn(params, X, y):
        logits = state.apply_fn({'params': params}, X)
        one_hot = jax.nn.one_hot(y, 10)
        loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot).flatten()
        assert len(loss) == 1
        return loss.sum()
    
    grad_fn = lambda X, y: jax.grad(loss_fn)(state.params, X, y)
    px_grads = jax.vmap(grad_fn, in_axes=(0, 0))(batch_X, batch_y)
  
    return px_grads


def compute_gradients_non_private(state, batch_X, batch_y, mask):
    """Computes gradients, loss and accuracy for a single batch."""

    def loss_fn(params, X, y):
        logits = state.apply_fn({'params': params}, X)
        one_hot = jax.nn.one_hot(y, 10)
        loss = jnp.sum(mask * optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        return loss
    
    summed_grad = jax.grad(loss_fn)(state.params, batch_X, batch_y)
  
    return summed_grad

@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)

@jax.jit
def eval(state, batch_X, batch_y,num_classes):
    """Computes gradients, loss and accuracy for a single batch."""

    logits = state.apply_fn({'params': state.params}, batch_X)
    one_hot = jax.nn.one_hot(batch_y, num_classes)
    loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot).flatten()
    predicted_class = jnp.argmax(logits,axis=-1)
    
    acc = jnp.mean(predicted_class == batch_y)

    return acc

def eval_model(data_loader,state,num_classes):
    # Test model on all images of a data loader and return avg loss
    accs = []
    losses = []
    test_loss = 0
    total_test = 0
    correct_test = 0
    batch_idx = 0
    for batch_idx,batch in enumerate(data_loader):
        loss, acc,cor = eval(state,batch[0],batch[1],num_classes)
        test_loss += loss
        correct_test += cor
        total_test += len(batch[1])
        accs.append(cor/len(batch[1]))
        losses.append(float(loss))
        del batch
    eval_acc = jnp.mean(jnp.array(accs))
    eval_loss = jnp.mean(jnp.array(losses))
    
    return test_loss/len(data_loader),eval_acc,correct_test,total_test

def create_train_state(rng, lr,model,params):
    """Creates initial `TrainState`."""
    #cnn = CNN()
    #params = cnn.init(rng, jnp.ones([1, 28, 28, 1]))['params']
    tx = optax.adam(lr)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def process_a_physical_batch(px_grads, mask, C):

    def clip_mask_and_sum(x, mask, clipping_multiplier):

        new_shape = (-1,) + (1,) * (x.ndim - 1)
        mask = mask.reshape(new_shape)
        clipping_multiplier = clipping_multiplier.reshape(new_shape)

        return jnp.sum(x * mask * clipping_multiplier, axis=0)

    px_per_param_sq_norms = jax.tree_map(lambda x: jnp.linalg.norm(x.reshape(x.shape[0], -1), axis=-1)**2, px_grads)
    flattened_px_per_param_sq_norms, tree_def = jax.tree_util.tree_flatten(px_per_param_sq_norms)

    px_grad_norms = jnp.sqrt(jnp.sum(jnp.array(flattened_px_per_param_sq_norms), axis=0))

    clipping_multiplier = jnp.minimum(1., C/px_grad_norms)

    return jax.tree_map(lambda x: clip_mask_and_sum(x, mask, clipping_multiplier), px_grads)

def private_iteration_v2(logical_batch, state, k, q, t, noise_std, C, full_data_size):
    params = state.params
    
    sampling_rng = jax.random.PRNGKey(t + 1)
    batch_rng, binomial_rng = jax.random.split(sampling_rng, 2) 

    x, y = logical_batch

    logical_batch_size = len(x)
    physical_batches = jnp.array(jnp.split(x, k)) # k x pbs x dim
    physical_labels = jnp.array(jnp.split(y, k))
    # poisson subsample
    actual_batch_size = jax.random.bernoulli(binomial_rng, shape=(full_data_size,), p=q).sum()    
    n_masked_elements = logical_batch_size - actual_batch_size
    masks = jnp.concatenate([jnp.ones(actual_batch_size), jnp.zeros(n_masked_elements)])

    masks = jnp.array(jnp.split(masks, k))

    @jax.jit
    def noise_addition(rng_key, accumulated_clipped_grads, noise_std, C):
        num_vars = len(jax.tree_util.tree_leaves(accumulated_clipped_grads))
        treedef = jax.tree_util.tree_structure(accumulated_clipped_grads)
        new_key, *all_keys = jax.random.split(rng_key, num=num_vars + 1)
        noise = jax.tree_util.tree_map(
            lambda g, k: jax.random.normal(k, shape=g.shape, dtype=g.dtype),
            accumulated_clipped_grads, jax.tree_util.tree_unflatten(treedef, all_keys))
        updates = jax.tree_util.tree_map(
            lambda g, n: g + noise_std * C * n,
            accumulated_clipped_grads, noise)
        return updates

    ### gradient accumulation
    total_iter = 0
    correct_iter = 0
    accumulated_clipped_grads = jax.tree_map(lambda x: 0. * x, params)
    for pb, yb, mask in zip(physical_batches, physical_labels, masks):
        per_example_gradients = compute_per_example_gradients(state, pb, yb)
        sum_of_clipped_grads_from_pb = process_a_physical_batch(per_example_gradients, mask, C)
        accumulated_clipped_grads = jax.tree_map(lambda x,y: x+y, 
                                                accumulated_clipped_grads, 
                                                sum_of_clipped_grads_from_pb
                                                )

    noisy_grad = noise_addition(jax.random.PRNGKey(t), accumulated_clipped_grads, noise_std, C)

    ### update
    new_state = update_model(state, noisy_grad)
    return new_state, noisy_grad, actual_batch_size

def private_iteration_fori_loop(logical_batch, state, k, q, t, noise_std, C, full_data_size):
    params = state.params
    
    sampling_rng = jax.random.PRNGKey(t + 1)
    batch_rng, binomial_rng = jax.random.split(sampling_rng, 2) 

    x, y = logical_batch

    logical_batch_size = len(x)
    physical_batches = jnp.array(jnp.split(x, k)) # k x pbs x dim
    physical_labels = jnp.array(jnp.split(y, k))
    # poisson subsample
    actual_batch_size = jax.random.bernoulli(binomial_rng, shape=(full_data_size,), p=q).sum()    
    n_masked_elements = logical_batch_size - actual_batch_size
    masks = jnp.concatenate([jnp.ones(actual_batch_size), jnp.zeros(n_masked_elements)])
    masks = jnp.array(jnp.split(masks, k))

    @jax.jit
    def noise_addition(rng_key, accumulated_clipped_grads, noise_std, C):
        num_vars = len(jax.tree_util.tree_leaves(accumulated_clipped_grads))
        treedef = jax.tree_util.tree_structure(accumulated_clipped_grads)
        new_key, *all_keys = jax.random.split(rng_key, num=num_vars + 1)
        noise = jax.tree_util.tree_map(
            lambda g, k: jax.random.normal(k, shape=g.shape, dtype=g.dtype),
            accumulated_clipped_grads, jax.tree_util.tree_unflatten(treedef, all_keys))
        updates = jax.tree_util.tree_map(
            lambda g, n: g + noise_std * C * n,
            accumulated_clipped_grads, noise)
        return updates

    ### gradient accumulation
    
    @jax.jit
    def body_fun(t, accumulated_clipped_grads):
        pb = physical_batches[t]
        yb = physical_labels[t]
        mask = masks[t]
        per_example_gradients = compute_per_example_gradients(state, pb, yb)
        sum_of_clipped_grads_from_pb = process_a_physical_batch(per_example_gradients, mask, C)
        accumulated_clipped_grads = jax.tree_map(lambda x,y: x+y, 
                                                accumulated_clipped_grads, 
                                                sum_of_clipped_grads_from_pb
                                                )
        return accumulated_clipped_grads

    
    accumulated_clipped_grads0 = jax.tree_map(lambda x: 0. * x, params)
    accumulated_clipped_grads = jax.lax.fori_loop(0, k, body_fun, accumulated_clipped_grads0)

    noisy_grad = noise_addition(jax.random.PRNGKey(t), accumulated_clipped_grads, noise_std, C)

    ### update
    new_state = update_model(state, noisy_grad)
    return new_state, noisy_grad, actual_batch_size

def non_private_iteration(logical_batch, state, k, q, t, full_data_size):
    params = state.params
    
    sampling_rng = jax.random.PRNGKey(t + 1)
    batch_rng, binomial_rng = jax.random.split(sampling_rng, 2) 

    x, y = logical_batch

    logical_batch_size = len(x)
    physical_batches = jnp.array(jnp.split(x, k)) # k x pbs x dim
    physical_labels = jnp.array(jnp.split(y, k))
    # poisson subsample
    actual_batch_size = jax.random.bernoulli(binomial_rng, shape=(full_data_size,), p=q).sum()    
    n_masked_elements = logical_batch_size - actual_batch_size
    masks = jnp.concatenate([jnp.ones(actual_batch_size), jnp.zeros(n_masked_elements)])
    masks = jnp.array(jnp.split(masks, k))

    ### gradient accumulation
    
    @jax.jit
    def body_fun(t, accumulated_grads):
        pb = physical_batches[t]
        yb = physical_labels[t]
        mask = masks[t]
        summed_grads_from_pb = compute_gradients_non_private(state, pb, yb, mask)
        
        accumulated_grads = jax.tree_map(lambda x,y: x+y, 
                                                accumulated_grads, 
                                                summed_grads_from_pb
                                                )
        return accumulated_grads

    
    accumulated_grads0 = jax.tree_map(lambda x: 0. * x, params)
    accumulated_grads = jax.lax.fori_loop(0, k, body_fun, accumulated_grads0)


    ### update
    new_state = update_model(state, accumulated_grads)
    return new_state, accumulated_grads, actual_batch_size

class FixedBatchsizeSampler(Sampler[List[int]]):

    def __init__(
        self, *, num_samples_total: int, batch_size: int, steps: int, generator=None
    ):

        self.num_samples_total = num_samples_total
        self.batch_size = batch_size
        self.steps = steps
        
        self.generator = generator

        if self.num_samples_total <= 0:
            raise ValueError(
                "num_samples should be a positive integer "
                "value, but got num_samples={}".format(self.num_samples)
            )

    def __len__(self):
        return self.steps

    def __iter__(self):
        num_batches = self.steps
        while num_batches > 0:
            indices = torch.randperm(self.num_samples_total, generator=self.generator)[:self.batch_size]
            yield indices

            num_batches -= 1
    
def image_to_numpy_wo_t(img):
    img = np.array(img, dtype=np.float32)
    img = ((img / 255.) -  np.array([0.5, 0.5, 0.5])) /  np.array([0.5, 0.5, 0.5])
    img = np.transpose(img,[2,0,1])
    return img
    
def load_dataset(dimension):

    transformation = torchvision.transforms.Compose([
        torchvision.transforms.Resize(dimension),
        image_to_numpy_wo_t,
    ])
    trainset = torchvision.datasets.CIFAR100(root='../data_cifar100/', train=True, download=True, transform=transformation)
    testset = torchvision.datasets.CIFAR100(root='../data_cifar100/', train=False, download=True, transform=transformation)
    
    return trainset,testset

def numpy_collate(batch):
    if isinstance(batch[0],np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0],(tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)
    
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
                #x = nn.Conv(features=64, kernel_size=(3, 3))(x)
                #x = nn.relu(x)
                #x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
                x = x.reshape((x.shape[0], -1))  # flatten
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
        model = model
        params = variables['params']
    
    elif 'vit' in model_name:
        model_name = model_name
        #model = FlaxViTForImageClassification.from_pretrained(model_name)
        model = FlaxViTModel.from_pretrained(model_name,add_pooling_layer=False)
        module = model.module # Extract the Flax Module
        vars = {'params': model.params} # Extract the parameters
        #config = module.config
        model = ViTModelHead(num_classes=num_classes,pretrained_model=model)

        input_shape = (1,3,dimension,dimension)
        #But then, we need to split it in order to get random numbers
        

        #The init function needs an example of the correct dimensions, to infer the dimensions.
        #They are not explicitly writen in the module, instead, the model infer them with the first example.
        x = jax.random.normal(params_key, input_shape)

        main_rng, init_rng, dropout_init_rng = jax.random.split(main_key, 3)
        #Initialize the model
        variables = model.init({'params':init_rng},x)

        #So far, the parameters are initialized randomly, so we need to unfreeze them and add the pre loaded parameters.
        params = variables['params']
        params['vit'] = vars['params']
        #params = unfreeze(params)
        #print_param_shapes(params)
        #print(params)
        #model.apply({'params':params},x)
        model = model
        params = params
    return main_rng,model,params

def main(args):
    print(args)
    rng = jax.random.PRNGKey(0)

    trainset,testset = load_dataset(args.dimension)

    print('dataset loaded')

    rng,model,params = load_model(rng,args.model_name,args.dimension,args.ten)
    
    print('model loaded')
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, args.lr,model,params)
    q = q = 1/math.ceil(len(trainset)/args.bs)
    n = len(trainset)
    physical_bs = args.phy_bs

    alpha = 1e-9 # failure prob.

    from scipy.stats import binom
    k = 1
    binom_dist = binom(n, q)
    while True:
        right_prob = binom_dist.sf(k * physical_bs)
        if right_prob < alpha:
            break
        k += 1
        
    max_logical_batch_size = k*physical_bs

    print('n',n,'q',q,'k',k,'max logical batch size',max_logical_batch_size)

    steps = args.epochs * math.ceil(len(trainset)/args.bs)

    fbs = FixedBatchsizeSampler(num_samples_total=n, batch_size=max_logical_batch_size, steps=steps)

    #dataset = CustomImageDataset(train_images, train_labels)
    trainloader = torch.utils.data.DataLoader(trainset, batch_sampler=fbs,collate_fn=numpy_collate)
    testloader = torch.utils.data.DataLoader(testset, batch_size=80, shuffle=False,collate_fn=numpy_collate)

    clipping_mode = args.clipping_mode

    epochs = args.epochs

    t = 0
    for e in range(epochs):
        for batch_X, batch_y in trainloader:
            batch_X = jnp.array(batch_X).reshape(-1, 1,3, args.dimension, args.dimension)
            batch_y = jnp.array(batch_y)
            if clipping_mode == 'non-private':
                state, non_private_grad, actual_batch_size = non_private_iteration((batch_X, batch_y), state, k, q, t, n)
            elif clipping_mode == 'private':
                state, noisy_grad, actual_batch_size = private_iteration_v2((batch_X, batch_y), state, k, q, t, 10.0, 1.0, n)
            t = t+1
        print('after',e,'epoch','iteration',t)
        #eval(state,)

#main(dict({'dimension':224,'epochs':2,'clipping_mode':'private','num_classes':100,'model_name':'google/vit-base-patch16-224','lr':0.00031,'bs':25000,'pbs':50}))

    # _, barely_clipped_grad, actual_batch_size = private_iteration_v2((batch_X, batch_y), state, k, q, t, 0.0, 1000.0, n)
    # _, just_clipped_grad, actual_batch_size = private_iteration_v2((batch_X, batch_y), state, k, q, t, 0.0, 1.0, n)
    # _, noisy_grad, actual_batch_size = private_iteration_v2((batch_X, batch_y), state, k, q, t, 10.0, 1.0, n)