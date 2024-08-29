import os
from typing import Any
import tensorflow as tf
tf.config.experimental.set_visible_devices([], 'GPU')

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".65"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"

#import pdb
#import itertools
from datetime import datetime
import warnings
#from collections import defaultdict
#import argparse

#from typing import Sequence
import functools

import re
import numpy as np

import math


from scipy.stats import binom

#JAX
import jax
from jax import jit
import jax.numpy as jnp
#from jax.nn import log_softmax
import jax.profiler
from jax.lax import fori_loop


#Flax
import flax.linen as nn
#from flax import jax_utils
#from flax.training import train_state
#from flax.core.frozen_dict import freeze,FrozenDict

#Optimizer library for JAX. From it we got the dpsgd optimizer
import optax
from optax._src import clipping,base

#DP-Accounting - JAX/Flax doesn't have their own as Opacus
from dp_accounting import dp_event
from dp_accounting import rdp

#Torch libraries, mainly for data loading
import torch
import torchvision
import torch.utils.data as data
import torch.backends.cudnn

#Import the modules of the models. For the ResNet models, they came from the private_resnet file
#from private_resnet import FlaxResNetModelClassifier,ResNetModelHeadModule
from transformers import FlaxViTModel
from private_vit import ViTModelHead
from functools import partial

## tqdm for progress bars
from tqdm.auto import tqdm

# Noise multiplier from Opacus. To calculate the sigma and ensure the epsilon, the privacy budget
from opacus.accountants.utils import get_noise_multiplier
from opacus.data_loader import DPDataLoader

#Logging
from torch.utils.tensorboard.writer import SummaryWriter
from nvitop.callbacks.tensorboard import add_scalar_dict
from nvitop import CudaDevice,ResourceMetricCollector

import models_flax
import time
from MyOwnBatchManager import MyBatchMemoryManager,EndingLogicalBatchSignal
from transform_functions import add_noise,AddNoiseStateC

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


class TrainerModule:

    def __init__(self,model_name,lr=0.0005,epochs = 20,seed=1234,max_grad = 0.1,accountant_method='rdp',
                 batch_size=20,physical_bs = 10,target_epsilon=2,target_delta=1e-5,num_classes = 10,test='train',dimension=224,clipping_mode='private',dataset_size = 50000,k = 100,q=1/2) -> None:
        self.lr = lr
        self.seed = seed
        self.epochs = epochs
        self.max_grad_norm = max_grad
        self.rng = jax.random.PRNGKey(self.seed)
        self.accountant = accountant_method
        self.batch_size = batch_size
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.model_name = model_name
        self.dimension = dimension
        self.num_classes = num_classes
        self.acc_steps = batch_size//physical_bs
        self.physical_bs = physical_bs
        self.dataset_size = dataset_size

        timestamp = datetime.now().strftime('%Y%m%d%M')
        print('model at time: ',timestamp,flush=True)
        self.logger = SummaryWriter('runs_{}/{}_{}_cifar_{}_epsilon_{}_model_{}_{}_{}'.format(target_epsilon,test,clipping_mode,num_classes,target_epsilon,model_name,timestamp,epochs),flush_secs=30)
        self.collector = ResourceMetricCollector(devices=CudaDevice.all(),
                                            root_pids={os.getpid()},
                                            interval=1.0)
        

        #self.create_functions()

        self.k = k
        self.q = q
        self.state = None
        self.load_model()
        print(self.model_name,self.num_classes,self.target_epsilon,'acc steps',self.acc_steps)

    def compute_epsilon(self,steps,batch_size, num_examples=60000, target_delta=1e-5,noise_multiplier=0.1):
        if num_examples * target_delta > 1.:
            warnings.warn('Your delta might be too high.')

        print('steps',steps,flush=True)

        print('noise multiplier',noise_multiplier,flush=True)

        q = batch_size / float(num_examples)
        orders = list(jnp.linspace(1.1, 10.9, 99)) + list(range(11, 64))
        accountant = rdp.rdp_privacy_accountant.RdpAccountant(orders) # type: ignore
        accountant.compose(
            dp_event.PoissonSampledDpEvent(
                q, dp_event.GaussianDpEvent(noise_multiplier)), steps)
        
        epsilon = accountant.get_epsilon(target_delta)
        delta = accountant.get_delta(epsilon)

        return epsilon,delta
    
    def set_loaders(self,train_loader,test_loader):
        self.train_loader = train_loader
        self.test_loader = test_loader

    def calculate_noise(self,size):
        noise_multiplier = get_noise_multiplier(
            target_epsilon=self.target_epsilon,
            target_delta=self.target_delta,
            sample_rate=1/size,
            epochs=self.epochs,
            accountant=self.accountant
        )

        self.noise_multiplier = noise_multiplier
    
    def init_non_optimizer(self):
        self.optimizer = optax.adam(learning_rate=self.lr)
        self.opt_state = self.optimizer.init(self.params)

    def init_with_chain(self,size,sample_rate):
        print('init optimizer, size ',size)

        expected_bs = (size * sample_rate)

        total_steps = int(size//expected_bs)
        print('total steps',total_steps)

        expected_acc_steps = expected_bs // self.physical_bs

        print('expected acc steps',expected_acc_steps)

        optimizer = optax.chain(
            add_noise(self.noise_multiplier*self.max_grad_norm,expected_bs,self.seed),
            optax.adam(learning_rate=self.lr)
        )
        
        self.optimizer = optax.MultiSteps(optimizer,every_k_schedule=int(expected_acc_steps),use_grad_mean=False)

        self.opt_state  = self.optimizer.init(self.params)

        #print('self opt after init',self.opt_state)

    def init_with_chain2(self,size,sample_rate):
        print('init optimizer, size ',size)

        expected_bs = (size * sample_rate)

        #total_steps = int(size//expected_bs)
        #print('total steps',total_steps)

        #expected_acc_steps = expected_bs // self.physical_bs

        print('expected batch size',expected_bs)

        print('noise multiplier',self.noise_multiplier,'max grad norm',self.max_grad_norm,'noise',self.noise_multiplier*self.max_grad_norm)

        #noise_state = AddNoiseStateC(self.seed)

        self.optimizer = optax.chain(
            add_noise(self.noise_multiplier*self.max_grad_norm,expected_bs,self.seed),
            optax.adam(learning_rate=self.lr)
        )
        
        #self.optimizer = optax.MultiSteps(optimizer,every_k_schedule=int(expected_acc_steps),use_grad_mean=False)

        self.opt_state  = self.optimizer.init(self.params)

        #print('self opt after init',self.opt_state)
        
    
    def calculate_metrics(self,params,batch):
        inputs,targets = batch
        logits = self.model.apply({'params':params},inputs)
        predicted_class = jnp.argmax(logits,axis=-1)
        acc = jnp.mean(predicted_class==targets)
        return acc
    
    def loss(self,params,batch):
        inputs,targets = batch
        logits = self.model.apply({'params':params},inputs)
        predicted_class = jnp.argmax(logits,axis=-1)

        cross_loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()

        vals = predicted_class == targets
        acc = jnp.mean(vals)
        cor = jnp.sum(vals)

        return cross_loss,(acc,cor)

    def loss_eval(self,params,batch):
        inputs,targets = batch
        logits = self.model.apply({'params':params},inputs)
        #print('logits shape',logits.shape)
        predicted_class = jnp.argmax(logits,axis=-1)
        cross_losses = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        #print('cross_losses:',cross_losses.shape)
        

        cross_loss = jnp.mean(cross_losses)
        #vals = (predicted_class > 0.5) == (targets > 0.5) 
        vals = predicted_class == targets
        acc = jnp.mean(vals)
        cor = jnp.sum(vals)
        #print('targets',targets)
        #print('predicted class',predicted_class)

        #jax.debug.breakpoint()

        return cross_loss,acc,cor
    
    #@partial(jit,static_argnums=0)
    def eval_step_non(self, params, batch):
        # Return the accuracy for a single batch
        #batch = jax.tree_map(lambda x: x[:, None], batch)
        loss,acc,cor =self.loss_eval(params,batch)
        #loss, acc= self.loss_2(self.params, batch)
        return loss, acc,cor
    
    @partial(jit, static_argnums=0)
    def mini_batch_dif_clip2(self,batch,params,l2_norm_clip):
        
        batch = jax.tree_map(lambda x: x[:, None], batch)
        
        (loss_val,(acc,cor)), per_example_grads = jax.vmap(jax.value_and_grad(self.loss,has_aux=True),in_axes=(None,0))(params,batch)
        
        grads_flat, grads_treedef = jax.tree_util.tree_flatten(per_example_grads)

        clipped, num_clipped = clipping.per_example_global_norm_clip(grads_flat, l2_norm_clip)

        grads_unflat = jax.tree_util.tree_unflatten(grads_treedef,clipped)

        return grads_unflat,jnp.mean(loss_val),jnp.mean(acc),jnp.sum(cor),num_clipped

    @partial(jit, static_argnums=0)
    def grad_acc_update(self,grads,opt_state,params):
        updates,opt_state = self.optimizer.update(grads,opt_state,params)
        params = optax.apply_updates(params,updates)
        return params,opt_state
    
    @partial(jit, static_argnums=0)
    def non_private_update(self,params,batch):
        (loss_val,(acc,cor)), grads = jax.value_and_grad(self.loss,has_aux=True)(params,batch)
        return grads,loss_val,acc,cor
    
    def print_param_change(self,old_params, new_params):
        for (old_k, old_v), (new_k, new_v) in zip(old_params.items(), new_params.items()):
            if isinstance(old_v, dict):
                self.print_param_change(old_v, new_v)
            else:
                diff = jnp.abs(new_v - old_v).mean()
                print(f"Param {old_k} mean absolute change: {diff}")
    
    @partial(jit,static_argnums=0)
    def add_noise_fn(self,noise_std,expected_bs,rng_key,updates):
        
        #jax.debug.print('inside update function:',noise_std,'expected_bs',expected_bs,'PRNG key',rng_key,flush=True)
        num_vars = len(jax.tree_util.tree_leaves(updates))
        treedef = jax.tree_util.tree_structure(updates)
        new_key,*all_keys = jax.random.split(rng_key, num=num_vars + 1)
        #print('num_vars',num_vars,flush=True)
        noise = jax.tree_util.tree_map(
            lambda g, k: jax.random.normal(k, shape=g.shape, dtype=g.dtype),
            updates, jax.tree_util.tree_unflatten(treedef, all_keys))
        updates = jax.tree_util.tree_map(
            lambda g, n: (g + noise_std * n)/expected_bs,
            updates, noise)

        return updates, new_key

    def iter_loop(self,train_loader,acc_function,params,opt_state,k,q,n):
        #_acc_update = lambda grad, acc : grad + acc
        for batch_idx,(x,y) in enumerate(train_loader): #logical

            print('batch idx',batch_idx)
            #print(type(x),x[0].shape)
            x  = jnp.array(x)
            y = jnp.array(y)
            #print(type(x),len(x),x.shape)
            #print(type(y),len(y),y.shape)

            diff = len(y) % k

            if diff > 0:

                x = jnp.pad(x, ((0, k - diff), (0, 0), (0, 0), (0, 0)), mode='constant')
                y = jnp.pad(y, ((0, k - diff)), mode='constant')
                print('new shape',x.shape,y.shape)
            
            batch_size = len(x)

            choice_rng, binom_rng = jax.random.split(jax.random.PRNGKey(batch_idx), 2)

            physical_batches = jnp.array(jnp.split(x, k))
            physical_labels = jnp.array(jnp.split(y,k))
            actual_logical_bs = jax.random.bernoulli(binom_rng, q, shape=(n,)).sum()
            masks = jnp.array(jnp.split((jnp.arange(batch_size) < actual_logical_bs), k))
            acc_grads = jax.tree_util.tree_map(jnp.zeros_like,params)
            def foo(t, args):
                acc_grad = args
                mask = masks[t]
                data_x = (physical_batches[t] * mask.reshape(-1, 1, 1, 1))
                data_y = (physical_labels[t] * mask)
                grads,loss,acc,cor = self.non_private_update(params,(data_x,data_y))
                return jax.tree_util.tree_map(
                                    functools.partial(acc_function),
                                    grads, acc_grad)

            accumulated_gradients = fori_loop(0, k, foo, acc_grads)
            #print('update?',accumulated_gradients)
            print('batch_idx',batch_idx,'update step')
            params,opt_state = self.grad_acc_update(accumulated_gradients,opt_state,params)
        
        return params,opt_state
    
    def iter_loop_private(self,train_loader,acc_function,params,opt_state,k,q,n,expected_bs):
        #_acc_update = lambda grad, acc : grad + acc
        for batch_idx,(x,y) in enumerate(train_loader): #logical

            print('batch idx',batch_idx)
            #print(type(x),x[0].shape)
            x  = jnp.array(x)
            y = jnp.array(y)
            #print(type(x),len(x),x.shape)
            #print(type(y),len(y),y.shape)

            diff = len(y) % k

            if diff > 0:

                x = jnp.pad(x, ((0, k - diff), (0, 0), (0, 0), (0, 0)), mode='constant')
                y = jnp.pad(y, ((0, k - diff)), mode='constant')
                print('new shape',x.shape,y.shape)
            
            batch_size = len(x)

            choice_rng, binom_rng = jax.random.split(jax.random.PRNGKey(batch_idx), 2)

            physical_batches = jnp.array(jnp.split(x, k))
            physical_labels = jnp.array(jnp.split(y,k))
            actual_logical_bs = jax.random.bernoulli(binom_rng, q, shape=(n,)).sum()
            masks = jnp.array(jnp.split((jnp.arange(batch_size) < actual_logical_bs), k))
            acc_grads = jax.tree_util.tree_map(jnp.zeros_like,params)
            def foo(t, args):
                acc_grad = args
                mask = masks[t]
                data_x = (physical_batches[t] * mask.reshape(-1, 1, 1, 1))
                data_y = (physical_labels[t] * mask)

                grads,loss,acc,cor,num_clipped = self.mini_batch_dif_clip2((data_x,data_y),params,self.max_grad_norm)
                #grads,loss,acc,cor = self.non_private_update(params,(data_x,data_y))
                return jax.tree_util.tree_map(
                                    functools.partial(acc_function),
                                    grads, acc_grad)

            accumulated_gradients = fori_loop(0, k, foo, acc_grads)
            updates,self.rng = self.add_noise_fn(self.noise_multiplier*self.max_grad_norm,expected_bs,self.rng,accumulated_gradients)
            #print('update?',accumulated_gradients)
            print('batch_idx',batch_idx,'update step')
            params,opt_state = self.grad_acc_update(updates,opt_state,params)
        
        return params,opt_state
    
    def train_epochs_dp(self,trainloader,testloader):
        expected_bs = len(trainloader.dataset)/len(trainloader)
        self.calculate_noise(len(trainloader))
        _acc_update = lambda grad, acc : grad + acc
        for i in range(self.epochs):
            print('epoch',i)
            self.params,self.opt_state = self.iter_loop_private(trainloader,_acc_update,self.params,self.opt_state,self.k,self.q,self.dataset_size,expected_bs)
            _,acc,_,_ = self.eval_model(testloader)
            print('end epoch',i,'acc',acc)
        return self.eval_model(testloader)


    def train_epochs(self,trainloader,testloader):
        
        _acc_update = lambda grad, acc : grad + acc
        for i in range(self.epochs):
            print('epoch',i)
            self.params,self.opt_state = self.iter_loop(trainloader,_acc_update,self.params,self.opt_state,self.k,self.q,self.dataset_size)
            _,acc,_,_ = self.eval_model(testloader)
            print('end epoch',i,'acc',acc)
        return self.eval_model(testloader)

    def eval_model(self, data_loader):
        # Test model on all images of a data loader and return avg loss
        accs = []
        losses = []
        test_loss = 0
        total_test = 0
        correct_test = 0
        batch_idx = 0
        for batch_idx,batch in enumerate(data_loader):
            loss, acc,cor = self.eval_step_non(self.params,batch)
            test_loss += loss
            correct_test += cor
            total_test += len(batch[1])
            accs.append(cor/len(batch[1]))
            losses.append(float(loss))
            del batch
        eval_acc = jnp.mean(jnp.array(accs))
        eval_loss = jnp.mean(jnp.array(losses))
        
        return test_loss/len(data_loader),eval_acc,correct_test,total_test
    
    def print_param_shapes(self,params, prefix=''):
        for key, value in params.items():
            if isinstance(value, dict):
                print(f"{prefix}{key}:")
                self.print_param_shapes(value, prefix + '  ')
            else:
                print(f"{prefix}{key}: {value.shape}")
      
    def load_model(self):
        print('load model name',self.model_name,flush=True)
        main_key, params_key= jax.random.split(key=self.rng,num=2)
        if self.model_name == 'small':
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
            input_shape = (1,3,self.dimension,self.dimension)
            #But then, we need to split it in order to get random numbers
            

            #The init function needs an example of the correct dimensions, to infer the dimensions.
            #They are not explicitly writen in the module, instead, the model infer them with the first example.
            x = jax.random.normal(params_key, input_shape)

            main_rng, init_rng, dropout_init_rng = jax.random.split(main_key, 3)
            #Initialize the model
            variables = model.init({'params':init_rng},x)
            #variables = model.init({'params':main_key}, batch)
            model.apply(variables, x)
            self.model = model
            self.params = variables['params']
        
        elif 'vit' in self.model_name:
            model_name = self.model_name
            model = FlaxViTModel.from_pretrained(model_name)
            module = model.module # Extract the Flax Module
            vars = {'params': model.params} # Extract the parameters
            config = module.config
            model = ViTModelHead(num_classes=self.num_classes,pretrained_model=model)

            input_shape = (1,3,self.dimension,self.dimension)
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
            self.print_param_shapes(params)
            #print(params)
            model.apply({'params':params},x)
            self.model = model
            self.params = params

        else:
            crop_size = self.dimension
            model = models_flax.KNOWN_MODELS['BiT-M-R50x1']
            bit_pretrained_dir = '/models_files/' # Change this with your directory. It might need the whole path, not the relative one.
            
            # Load weigths of a BiT model
            bit_model_file = os.path.join(bit_pretrained_dir, f'{self.model_name}.npz')
            if not os.path.exists(bit_model_file):
                raise FileNotFoundError(
                f'Model file is not found in "{bit_pretrained_dir}" directory.')
            with open(bit_model_file, 'rb') as f:
                params_tf = np.load(f)
                params_tf = dict(zip(params_tf.keys(), params_tf.values()))

            # Build ResNet architecture
            ResNet = model(num_classes = self.num_classes)

            x = jax.random.normal(params_key, (1, crop_size, crop_size, 3))

            main_rng, init_rng, dropout_init_rng = jax.random.split(main_key, 3)

            #Initialize the model
            variables = ResNet.init({'params':init_rng},x)

            params = variables['params']

            transform_params(params, params_tf,
                num_classes=self.num_classes)
            
            ResNet.apply({'params':params},x)
            
            self.model = ResNet
            self.params = params

        print('finish loading',flush=True)
        print('model loaded')
        print(jax.tree_util.tree_map(jnp.shape, self.params))
    
    def __str__(self) -> str:
        return f"Trainer with seed: {self.seed} and model"
    
DATA_MEANS = np.array([0.5, 0.5, 0.5])
DATA_STD = np.array([0.5,0.5, 0.5])

DATA_MEANS2 = (0.485, 0.456, 0.406)
DATA_STD2 =  (0.229, 0.224, 0.225)
def image_to_numpy(img):
    img = np.array(img, dtype=np.float32)
    img = (img / 255. - DATA_MEANS) / DATA_STD
    #There is the need of transposing the image. The image has the right dimension, but inside the ViT, it has a transpose where they move the number of channels to the last dim. So here I inverse 
    #that operation, so it works later during the pass
    return np.transpose(img)

def image_to_numpy_wo_t(img):
    img = np.array(img, dtype=np.float32)
    img = ((img / 255.) - DATA_MEANS) / DATA_STD
    img = np.transpose(img,[2,0,1])
    return img

def image_to_numpy_wo_t2(img):
    img = np.array(img, dtype=np.float32)
    img = ((img / 255.) - DATA_MEANS2) / DATA_STD2
    img = np.transpose(img,[2,0,1])
    return img

def numpy_collate(batch):
    if isinstance(batch[0],np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0],(tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)
    
def numpy_collate2(batch):
  return jax.tree_util.tree_map(jnp.asarray, data.default_collate(batch))
    
#Defines each worker seed. Since each worker needs a different seed.
#The worker_id is a parameter given by the loader, but it is not used inside the method
def seed_worker(worker_id):

    #print(torch.initial_seed(),flush=True)

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    #random.seed(worker_seed)

#Set seeds.
#Returns the generator, that will be used for the data loader
def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #random.seed(seed)

    g_cpu = torch.Generator('cpu')

    g_cpu.manual_seed(seed)

    np.random.seed(seed)

    return g_cpu


#Load CIFAR data
def load_data_cifar(ten,dimension,batch_size_train,physical_batch_size,num_workers,generator,norm):

    print('load_data_cifar',batch_size_train,physical_batch_size,num_workers)

    w_batch = batch_size_train

    if norm == 'True':
        fn = image_to_numpy_wo_t
    else:
        fn = image_to_numpy_wo_t2


    transformation = torchvision.transforms.Compose([
        torchvision.transforms.Resize(dimension),
        fn,
        #torchvision.transforms.ToTensor(),
        #torchvision.transforms.Normalize(DATA_MEANS,DATA_STD),
    ])
    
    if ten==10:
        trainset = torchvision.datasets.CIFAR10(root='../data_cifar10/', train=True, download=True, transform=transformation)
        testset = torchvision.datasets.CIFAR10(root='../data_cifar10/', train=False, download=True, transform=transformation)
    else:
        trainset = torchvision.datasets.CIFAR100(root='../data_cifar100/', train=True, download=True, transform=transformation)
        testset = torchvision.datasets.CIFAR100(root='../data_cifar100/', train=False, download=True, transform=transformation)

    trainloader = data.DataLoader(
        trainset, batch_size=w_batch, shuffle=True,collate_fn=numpy_collate, num_workers=num_workers,generator=generator,worker_init_fn=seed_worker)

    testloader = data.DataLoader(
        testset, batch_size=80, shuffle=False,collate_fn=numpy_collate, num_workers=num_workers,generator=generator,worker_init_fn=seed_worker)

    return trainloader,testloader

def calculate_sampling(dataset_size,physical_bs,sampling_rate):

    # n = 1 * int(1e4)
    # pbs = 32
    # q = 0.5

    alpha = 1e-9 # failure prob.


    k = 1
    binom_dist = binom(dataset_size, sampling_rate)
    while True:
        right_prob = binom_dist.sf(k * physical_bs)
        if right_prob < alpha:
            break
        k += 1
    
    return k,k*physical_bs

def privatize_dataloader(data_loader):
    return DPDataLoader.from_data_loader(data_loader)

def main(args):
    print(args,flush=True)
    generator = set_seeds(args.seed)

    q = 1/math.ceil(50000/args.bs)

    k,mlbs = calculate_sampling(50000,args.phy_bs,q)
    print('k:',k,'mlbs:',mlbs,'sample rate',q)
    #Load data
    trainloader,testloader = load_data_cifar(args.ten,args.dimension,mlbs,args.phy_bs,args.n_workers,generator,args.normalization)
    #if args.clipping_mode == 'mini':
    #    trainloader = privatize_dataloader(trainloader)
    #trainloader = privatize_dataloader(trainloader)
    print('data loaded',len(trainloader),flush=True)
    #Create Trainer Module, that loads the model and train it
    trainer = TrainerModule(model_name=args.model,lr=args.lr,seed=args.seed,epochs=args.epochs,max_grad=args.grad_norm,accountant_method=args.accountant,batch_size=args.bs,physical_bs=args.phy_bs,target_epsilon=args.epsilon,target_delta=args.target_delta,num_classes=args.ten,test=args.test,dimension=args.dimension,clipping_mode=args.clipping_mode,k=k,q=q)
    tloss,tacc,cor_eval,tot_eval = trainer.eval_model(testloader)
    print('Without trainig test loss',tloss)
    print('Without training test accuracy',tacc,'(',cor_eval,'/',tot_eval,')')
    trainer.init_non_optimizer()

    if args.clipping_mode == 'non-private':
        vals = trainer.train_epochs(trainloader,testloader)
    else:
        vals = trainer.train_epochs_dp(trainloader,testloader)

    print(vals)

    return 0,0,0,vals[1]

    # if args.clipping_mode == 'non-private':
    #     throughputs,throughputs_t,comp_time = trainer.non_private_training_mini_batch_2(trainloader,testloader)
    # elif args.clipping_mode == 'mini':
    #     throughputs,throughputs_t,comp_time,privacy_measures = trainer.private_training_mini_batch_2(trainloader,testloader)
    # tloss,tacc,cor_eval,tot_eval = trainer.eval_model(testloader)
    # print('throughputs',throughputs,'mean throughput', np.mean(throughputs))
    # print('compiling time',comp_time)
    # print('test loss',tloss)
    # print('test accuracy',tacc)
    # print('(',cor_eval,'/',tot_eval,')')
    # return np.mean(throughputs),np.mean(throughputs_t),comp_time,tacc