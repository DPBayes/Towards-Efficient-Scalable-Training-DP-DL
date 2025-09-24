import argparse
import os
import math
import time
import warnings
import jax
import optax

import numpy as np

from functools import partial

from datasets import load_dataset
from collections import namedtuple


from transformers import FlaxBertForSequenceClassification, BertTokenizer, FlaxRobertaForSequenceClassification, RobertaTokenizer,FlaxDistilBertForSequenceClassification, DistilBertTokenizer
#from peft import LoraConfig, get_peft_model, TaskType
from flax.training import train_state

from jaxdpopt.dp_accounting_utils import calculate_noise

from jaxdpopt.jax_mask_efficient import (
    compute_per_example_gradients_physical_batch,
    add_trees,
    clip_physical_batch,
    accumulate_physical_batch,
    LossFunction,
    CrossEntropyLoss
)

from jaxdpopt.dp_accounting_utils import compute_epsilon
from jaxdpopt.jax_mask_efficient import (
    get_padded_logical_batch,
    add_Gaussian_noise,
    poisson_sample_logical_batch_size,
    setup_physical_batches,
    update_model,
)
import jax.numpy as jnp

from flax.core.frozen_dict import freeze
from flax import traverse_util

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


def count_params(pytree):
    total = 0
    for leaf in jax.tree_util.tree_leaves(pytree):
        total += leaf.size
    return total

def main(args):

    print("Used args:", args, flush=True)

    jax.clear_caches()

    if args.precision == 'fp32':
        jax.config.update("jax_default_matmul_precision", jax.lax.Precision.HIGHEST)

    print('loading dataset',flush=True)

    # ## 0.2 Use GPU or CPU?


    USE_GPU = True


    dataset = load_dataset(args.dataset)
    train_set = dataset['train']
    test_set = dataset['test']
       

    dataset_size = train_set.num_rows


    # ## 1.2 Model

    optimizer_config = namedtuple("Config", ["learning_rate","max_length"])
    optimizer_config.learning_rate = args.lr
    max_length = args.max_length

    if args.model == 'distilbert/distilbert-base-uncased':
       model = FlaxDistilBertForSequenceClassification.from_pretrained(args.model, num_labels=args.num_classes)
       tokenizer = DistilBertTokenizer.from_pretrained(args.model)
       params = model.params
       print('distilbert',params['classifier'])

       params['classifier']['kernel'] = jnp.zeros_like(params['classifier']['kernel'])
       params['classifier']['bias'] = jnp.zeros_like(params['classifier']['bias'])
       print('distilbert after zeroing',params['classifier'])
    elif args.model == 'FacebookAI/roberta-base':
       model = FlaxRobertaForSequenceClassification.from_pretrained(
             args.model,
             num_labels=args.num_classes
       )

       tokenizer = RobertaTokenizer.from_pretrained(args.model, do_lower_case=False)
       params = model.params
       print('roberta params',params['classifier'])
       params["classifier"]['dense']["kernel"] = jnp.zeros_like(params["classifier"]['dense']["kernel"])
       params["classifier"]['dense']["bias"] = jnp.zeros_like(params["classifier"]['dense']["bias"])
       print('roberta params',params['classifier'])
    else:
       model = FlaxBertForSequenceClassification.from_pretrained(
             args.model,
             num_labels=args.num_classes
       )

       tokenizer = BertTokenizer.from_pretrained(args.model, do_lower_case=False)

       params = model.params

       print('bert params', params['classifier'])
       params["classifier"]["kernel"] = jnp.zeros_like(params["classifier"]["kernel"])
       params["classifier"]["bias"] = jnp.zeros_like(params["classifier"]["bias"])
       print('bert params', params['classifier'])

    layers_to_freeze = ['embeddings']
    params = freeze(params)
    partition_optimizers = {"trainable": optax.adam(optimizer_config.learning_rate), "frozen": optax.set_to_zero()}
    
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

    flat = list(traverse_util.flatten_dict(param_partitions).items())
    print(freeze(traverse_util.unflatten_dict(dict(flat))))

    flat_params = traverse_util.flatten_dict(params, sep=".")
    flat_partitions = traverse_util.flatten_dict(param_partitions, sep=".")

    
    trainable, frozen = 0, 0

    for k, v in flat_params.items():
        n = v.size  # number of elements in this array
        label = flat_partitions[k]
        if label == "trainable":
           trainable += n
        else:
           frozen += n

    print(f"Trainable parameters: {trainable:,}")
    print(f"Frozen parameters:    {frozen:,}")
    print(f"Total parameters:     {trainable+frozen:,}",flush=True)
    
    print(count_params(params))

    state = train_state.TrainState.create(apply_fn=model.__call__, params=params, tx=tx)

    args.logical_bs = dataset_size // 2

    # ## 1.3 DP accounting


    if dataset_size * args.target_delta > 1.0:
        warnings.warn("Your delta might be too high.")
   
    subsampling_ratio = 1 / math.ceil(dataset_size / args.logical_bs)

    noise_std = calculate_noise(
            sample_rate=subsampling_ratio,
            target_epsilon=args.target_epsilon,
            target_delta=args.target_delta,
            steps=args.num_steps,
            accountant=args.accountant,
    )


    # 

        # # 2. Function to process one physical batch
    # 
    # First we define the function that computes per example gradients (`compute_per_example_gradients_physical_batch`) and clips them (`clip_and_accumulate_physical_batch`). This function can be jit compiled and then used in the full training loop later (see 3.).

    class CrossEntropyLossText(LossFunction):

        def __init___(
                self,
                state,
                num_classes,
                resizer_fn
        ):
            """
            Initialize cross entropy loss

            Parameters
            ----------
            state : train_state.TrainState
                The train state that contains the model, parameters and optimizer    
            num_classes : int
                For classification tasks, the loss needs the number of classes
            resizer_fn : Callable:
                Optional callable function, that resizes the inputs before loss computation
            """
            super().__init__(state, num_classes, resizer_fn)
            if resizer_fn is None:
                self.resizer_fn = lambda x:x
        
        def __call__(self, params, X, y):
            """
            Compute cross entropy loss

            Return
            ----------
            Scalar loss value, as the sum of cross entropy loss between prediction and target

            """
            loss = self.state.apply_fn(input_ids=X[:, 0, :],attention_mask=X[:, 1, :], params=params)
            logits = loss[0]
            print('train loss',loss,flush=True)
            one_hot = jax.nn.one_hot(y, num_classes=self.num_classes)
            loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot).flatten()

            return loss.sum()


    def dict_for_input_train(x):
        return {'input_ids':x[:, 0, :],'attention_mask':x[:, 1, :]}

    def dict_for_input_test(x):
        return {'input_ids':x[:, 0, :],'attention_mask':x[:, 1, :]}

    loss_fn = CrossEntropyLossText(state=state,num_classes=args.num_classes,resizer_fn=lambda x:x)


    # 2 - first dimension: input_ids, second dimension: attention mask
    # max_length = max length for the sequence

    TEXT_SHAPE = (2,max_length,)

    # # 2. Function to process one physical batch
    # 
    # First we define the function that computes per example gradients (`compute_per_example_gradients_physical_batch`) and clips them (`clip_and_accumulate_physical_batch`). This function can be jit compiled and then used in the full training loop later (see 3.).




    @jax.jit
    def process_physical_batch(t, params):
        (
            state,
            accumulated_clipped_grads,
            logical_batch_X,
            logical_batch_y,
            masks,
        ) = params
        # slice
        start_idx = t * args.physical_bs

        start_shape = (start_idx,0,) + (0,)*len(TEXT_SHAPE)

        batch_shape = (args.physical_bs,1,) + TEXT_SHAPE

        pb = jax.lax.dynamic_slice(
            logical_batch_X,
            start_shape,
            batch_shape,
        )
        yb = jax.lax.dynamic_slice(logical_batch_y, (start_idx,), (args.physical_bs,))
        mask = jax.lax.dynamic_slice(masks, (start_idx,), (args.physical_bs,))

        # compute grads and clip
        per_example_gradients = compute_per_example_gradients_physical_batch(state, pb, yb, loss_fn)
        clipped_grads_from_pb = clip_physical_batch(per_example_gradients, args.clipping_norm)
        sum_of_clipped_grads_from_pb = accumulate_physical_batch(clipped_grads_from_pb, mask)
        accumulated_clipped_grads = add_trees(accumulated_clipped_grads, sum_of_clipped_grads_from_pb)

        return (
            state,
            accumulated_clipped_grads,
            logical_batch_X,
            logical_batch_y,
            masks,
        )

    @jax.jit
    def process_physical_batch_2(
        state,
        pb,
        yb,
        mask
    ):
        
        # compute grads and clip
        per_example_gradients = compute_per_example_gradients_physical_batch(state, pb, yb, loss_fn)
        #print('per_example_gradients')
        clipped_grads_from_pb = clip_physical_batch(per_example_gradients, args.clipping_norm)
        #print('clipped_grads')
        sum_of_clipped_grads_from_pb = accumulate_physical_batch(clipped_grads_from_pb, mask)
    
        return sum_of_clipped_grads_from_pb



    # 


    # # 3. Full training loop
    # 
    # The below cell executes the main training loop. It consists of the following parts at every step:
    # 
    # - Poission sampling of the logical batch size (`poisson_sample_logical_batch_size`): This gives us the logical batch size using Poisson subsampling.
    # - Rounding up of the logical batch size so that there are full physical batches (`setup_physical_batches`): This rounds up the logical batch size so that it is divisible in full physical batches
    # - Padding of the logical batches (`get_padded_logical_batch`): Here we load the actual images and labels.
    # - Computation of the per sample gradients (`jax.lax.fori_loop` using the previously defined `process_physical_batch`): This efficiently computes the per-example gradients of the logical batch.
    # - Addition of noise (`add_Gaussian_noise`): Add the required noise to the accumulated gradients of the logical batch. 
    # - Update of the model (`update_model`): Apply the gradient update to the model weights.
    # 
    # At the end of a step the following things are executed:
    # - Computation of the throughput: Compute the number of processed examples divided by the time spent.
    # - Evaluate the model (`model_evaluation`): Compute the test accuracy.
    # - Compute the spent privacy budget (`compute_epsilon`): Compute the spent privacy budget using a privacy accountant.


    def tokenize_data(data, tokenizer):
        """Tokenize the dataset"""
        def tokenize_function(examples):
            return tokenizer(
                examples['text'],
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='jax'
            )
        
        tokenized = data.map(tokenize_function, batched=True, remove_columns=['text'])
        return tokenized

    #Tokenize data

    train_tokens = tokenize_data(train_set,tokenizer)
    test_tokens = tokenize_data(test_set,tokenizer)
    train_inputs_ids = jnp.array(train_tokens['input_ids'])
    train_attention_masks = jnp.array(train_tokens['attention_mask'])
    train_x = jnp.stack([train_inputs_ids,train_attention_masks],axis=1)
    train_y = jnp.array(train_tokens['label'])

    test_inputs_ids = jnp.array(test_tokens['input_ids'])
    test_attention_masks = jnp.array(test_tokens['attention_mask'])
    test_x = jnp.stack([test_inputs_ids,test_attention_masks],axis=1)
    test_y = jnp.array(test_tokens['label'])

    print('test shape',test_x.shape,test_y.shape)
    
    @partial(jax.jit,static_argnames=["resizer_fn"])
    def compute_accuracy_for_batch(
        state: train_state.TrainState, batch_X: jax.typing.ArrayLike, batch_y: jax.typing.ArrayLike, resizer_fn=None
    ):
        """Computes accuracy for a single batch."""
        if resizer_fn is None:
            resizer_fn = lambda x: x

        if batch_X.size == 0:
            return 0

        logits = state.apply_fn(input_ids=batch_X[:, 0, :],attention_mask=batch_X[:, 1, :], params = state.params).logits
        if type(logits) is tuple:
            logits = logits[0]
        predicted_class = jnp.argmax(logits, axis=-1)
        correct = jnp.sum(predicted_class == batch_y)

        return correct


    @partial(jax.jit, static_argnames=["test_batch_size", "orig_img_shape","resizer_fn"])
    def test_body_fun(t, params, test_batch_size, orig_img_shape, resizer_fn=None):
        (state, accumulated_corrects, test_X, test_y) = params
        # slice
        start_idx = t * test_batch_size

        start_shape = (start_idx,) + (0,)*len(orig_img_shape)

        test_batch_shape = (test_batch_size,) + orig_img_shape

        pb = jax.lax.dynamic_slice(
            test_X,
            start_shape,
            test_batch_shape,
        )
        yb = jax.lax.dynamic_slice(test_y, (start_idx,), (test_batch_size,))

        n_corrects = compute_accuracy_for_batch(state, pb, yb, resizer_fn)

        accumulated_corrects += n_corrects

        return (state, accumulated_corrects, test_X, test_y)


    def model_evaluation(
        state: train_state.TrainState,
        test_images: jax.typing.ArrayLike,
        test_labels: jax.typing.ArrayLike,
        orig_img_shape: tuple,
        batch_size: int = 50,
        use_gpu=True,
        resizer_fn=None
    ):

        accumulated_corrects = 0
        n_test_batches = len(test_images) // batch_size

        if use_gpu:
            test_images = jax.device_put(test_images, jax.devices("gpu")[0])
            test_labels = jax.device_put(test_labels, jax.devices("gpu")[0])
            state = jax.device_put(state, jax.devices("gpu")[0])
        
        _, accumulated_corrects, *_ = jax.lax.fori_loop(
            0,
            n_test_batches,
            lambda t, params: test_body_fun(
                t, params, test_batch_size=batch_size, orig_img_shape=orig_img_shape,resizer_fn=resizer_fn
            ),
            (state, accumulated_corrects, test_images, test_labels),
        )

        # last remaining samples (basically the part that isn't a full batch)
        processed_samples = n_test_batches * batch_size

        n_remaining = len(test_images) % batch_size
        if n_remaining > 0:
            pb = test_images[-n_remaining:]
            yb = test_labels[-n_remaining:]
            n_corrects = compute_accuracy_for_batch(state, pb, yb)
            accumulated_corrects += n_corrects
            processed_samples += n_remaining

        return accumulated_corrects / processed_samples

    times = []
    logical_batch_sizes = []
    print('test before training')

    acc_f = model_evaluation(state, test_x, test_y, batch_size=10, use_gpu=USE_GPU, orig_img_shape=TEXT_SHAPE,resizer_fn=lambda x:x)

    print(acc_f)
    print('start training')

    for t in range(args.num_steps):
        print('starting iteration',t,flush=True)
        sampling_rng = jax.random.key(t + 1)
        batch_rng, binomial_rng, noise_rng = jax.random.split(sampling_rng, 3)

        #######
        # poisson subsample
        actual_batch_size = poisson_sample_logical_batch_size(
            binomial_rng=binomial_rng, dataset_size=dataset_size, q=subsampling_ratio
        )

        # determine padded_logical_bs so that there are full physical batches
        # and create appropriate masks to mask out unnessary elements later
        masks, n_physical_batches = setup_physical_batches(
            actual_logical_batch_size=actual_batch_size,
            physical_bs=args.physical_bs,
        )

        # get random padded logical batches that are slighly larger actual batch size
        padded_logical_batch_X, padded_logical_batch_y = get_padded_logical_batch(
            batch_rng=batch_rng,
            padded_logical_batch_size=len(masks),
            train_X=train_x,
            train_y=train_y,
        )

        padded_logical_batch_X = padded_logical_batch_X.reshape(-1, 1, 2, max_length)
        
        #padded_logical_batch_X = padded_logical_batch_X.reshape(n_physical_batches,args.physical_bs, 1, 2, max_length)
        #padded_logical_batch_y = padded_logical_batch_y.reshape(n_physical_batches,args.physical_bs)
        #masks = masks.reshape(n_physical_batches,args.physical_bs)

        padded_logical_batch_X = jax.device_put(padded_logical_batch_X, jax.devices("gpu")[0])
        padded_logical_batch_y = jax.device_put(padded_logical_batch_y, jax.devices("gpu")[0])
        masks = jax.device_put(masks, jax.devices("gpu")[0])
    
        print(padded_logical_batch_X.shape)      

        params = state.params

        accumulated_clipped_grads0 = jax.tree.map(lambda x: 0.0 * x, params)

        start = time.time()

        #process_logical_fn = jax.vmap(process_physical_batch_2, in_axes=(None, 0, 0, 0))

        #clipped_grads_from_pb = process_logical_fn(state,padded_logical_batch_X,padded_logical_batch_y,masks)

        #print(clipped_grads_from_pb)
        #accumulated_clipped_grads = jax.tree_util.tree_map(lambda x: x.sum(axis=0), clipped_grads_from_pb)
        #with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True,create_perfetto_trace=True):
        # Main loop
        _, accumulated_clipped_grads, *_ = jax.lax.fori_loop(
            0,
            n_physical_batches,
            process_physical_batch,
            (
                state,
                accumulated_clipped_grads0,
                padded_logical_batch_X,
                padded_logical_batch_y,
                masks,
            ),
        )  

        noisy_grad = add_Gaussian_noise(noise_rng, accumulated_clipped_grads, noise_std, args.clipping_norm)

        # update
        state = jax.block_until_ready(update_model(state, noisy_grad))

        end = time.time()
        duration = end - start

        times.append(duration)
        logical_batch_sizes.append(actual_batch_size)

        print(f"throughput at iteration {t}: {actual_batch_size / duration}", flush=True)

        acc_iter = jax.block_until_ready(model_evaluation(
            state, test_x, test_y, batch_size=100, orig_img_shape=TEXT_SHAPE, use_gpu=USE_GPU,resizer_fn=lambda x:x
            )
        )
        print(f"accuracy at iteration {t}: {acc_iter}", flush=True)

        # Compute privacy guarantees
        epsilon, delta = compute_epsilon(
            noise_multiplier=noise_std,
            sample_rate=subsampling_ratio,
            steps=t + 1,
            target_delta=args.target_delta,
            accountant=args.accountant,
        )
        privacy_results = {"accountant": args.accountant, "epsilon": epsilon, "delta": delta}
        print(privacy_results, flush=True)

    acc_last = model_evaluation(state, test_x, test_y, batch_size=10, use_gpu=USE_GPU, orig_img_shape=TEXT_SHAPE,resizer_fn=lambda x:x)

    print("times \n", times, flush=True)

    print("batch sizes \n ", logical_batch_sizes, flush=True)

    print("accuracy at end of training", acc_last, flush=True)
    thr = np.mean(np.array(logical_batch_sizes) / np.array(times))
    print("throughput", thr)

    thr_wocom = np.mean(np.array(logical_batch_sizes[1:])/np.array(times[1:]))
    comp_time = times[0]
    batch_size_comp = logical_batch_sizes[0]
    print("throughput", thr)

    return thr,thr_wocom,comp_time,batch_size_comp,acc_last





