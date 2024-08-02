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

#JAX
import jax
from jax import jit
import jax.numpy as jnp
#from jax.nn import log_softmax
import jax.profiler

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
from transform_functions import add_noise

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
                 batch_size=20,physical_bs = 10,target_epsilon=2,target_delta=1e-5,num_classes = 10,test='train',dimension=224,clipping_mode='private',dataset_size = 50000) -> None:
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

        timestamp = datetime.now().strftime('%Y%m%d')
        print('model at time: ',timestamp,flush=True)
        self.logger = SummaryWriter('runs/{}_{}_cifar_{}_model_{}_{}'.format(test,clipping_mode,num_classes,model_name,timestamp),flush_secs=30)
        self.collector = ResourceMetricCollector(devices=CudaDevice.all(),
                                            root_pids={os.getpid()},
                                            interval=1.0)
        

        #self.create_functions()
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
        return accountant.get_epsilon(target_delta)

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
        self.opt_state  = self.optimizer.init(self.params)

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
        #print('init optimizer, size ',size)

        expected_bs = (size * sample_rate)

        #total_steps = int(size//expected_bs)
        #print('total steps',total_steps)

        #expected_acc_steps = expected_bs // self.physical_bs

        #print('expected acc steps',expected_acc_steps)

        self.optimizer = optax.chain(
            add_noise(self.noise_multiplier*self.max_grad_norm,expected_bs,self.seed),
            optax.adam(learning_rate=self.lr)
        )
        
        #self.optimizer = optax.MultiSteps(optimizer,every_k_schedule=int(expected_acc_steps),use_grad_mean=False)

        self.opt_state  = self.optimizer.init(self.params)

        #print('self opt after init',self.opt_state)
    
    def loss(self,params,batch):
        inputs,targets = batch
        logits = self.model.apply({'params':params},inputs)
        predicted_class = jnp.argmax(logits,axis=-1)

        cross_loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()

        acc = (predicted_class==targets).mean()

        #jax.debug.breakpoint()

        return cross_loss,acc

    def loss_eval(self,params,batch):
        inputs,targets = batch
        logits = self.model.apply({'params':params},inputs)
        predicted_class = jnp.argmax(logits,axis=-1)

        cross_loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()

        acc = (predicted_class==targets).mean()

        #jax.debug.breakpoint()

        return cross_loss,acc
    
    #@partial(jit,static_argnums=0)
    def eval_step_non(self, params, batch):
        # Return the accuracy for a single batch
        #batch = jax.tree_map(lambda x: x[:, None], batch)
        loss,acc =self.loss_eval(params,batch)
        #loss, acc= self.loss_2(self.params, batch)
        return loss, acc
    
    @partial(jit, static_argnums=0)
    def mini_batch_dif_clip2(self,batch,params,l2_norm_clip):
        
        batch = jax.tree_map(lambda x: x[:, None], batch)
        
        (loss_val,acc), per_example_grads = jax.vmap(jax.value_and_grad(self.loss,has_aux=True),in_axes=(None,0))(params,batch)
        
        grads_flat, grads_treedef = jax.tree_util.tree_flatten(per_example_grads)

        clipped, num_clipped = clipping.per_example_global_norm_clip(grads_flat, l2_norm_clip)

        grads_unflat = jax.tree_util.tree_unflatten(grads_treedef,clipped)

        return grads_unflat,jnp.mean(loss_val),jnp.mean(acc),num_clipped

    @partial(jit, static_argnums=0)
    def grad_acc_update(self,grads,opt_state,params):
        updates,opt_state = self.optimizer.update(grads,opt_state,params)
        params = optax.apply_updates(params,updates)
        return params,opt_state
    
    @partial(jit, static_argnums=0)
    def non_private_update(self,params,batch):
        (loss_val,acc), grads = jax.value_and_grad(self.loss,has_aux=True)(params,batch)
        return grads,loss_val,acc
    
    def private_training_mini_batch_2(self,trainloader,testloader):

        #Training
        print('private learning',flush=True)
        
        _acc_update = lambda grad, acc : grad + acc

        self.calculate_noise(len(trainloader))
        self.init_with_chain2(len(trainloader.dataset),1/len(trainloader))
        print('noise multiplier',self.noise_multiplier)
        throughputs = np.zeros(self.epochs)
        throughputs_t = np.zeros(self.epochs)
        expected_bs = len(trainloader.dataset)/len(trainloader)
        expected_acc_steps = expected_bs // self.physical_bs
        print('expected accumulation steps',expected_acc_steps)
        acc_grads = jax.tree_util.tree_map(jnp.zeros_like, self.params)
        comp_time = 0
        gradient_step_ac = 0
        for epoch in range(1,self.epochs+1):
            flag = EndingLogicalBatchSignal()
            batch_idx = 0
            metrics = {}
            metrics['loss'] = jnp.array([])
            metrics['acc'] = jnp.array([])
            
            total_time_epoch = 0
            samples_used = 0 
            start_time_epoch = time.time()
            batch_times = []
            sample_sizes = []

            steps = int(epoch * expected_acc_steps)

            print('steps',steps,flush=True)
            with MyBatchMemoryManager(
                data_loader=trainloader, 
                max_physical_batch_size=self.physical_bs, 
                signaler=flag
                ) as memory_safe_data_loader:
                print('memory safe data loader len ',len(memory_safe_data_loader.batch_sampler))
                for batch_idx, batch in enumerate(memory_safe_data_loader): 
                    with self.collector(tag='batch'):
                        print(batch[0][0].shape)
                        samples_used += len(batch[0])
                        sample_sizes.append(len(batch[0]))
                        start_time = time.time()
                        grads,loss,accu,num_clipped = jax.block_until_ready(self.mini_batch_dif_clip2(batch,self.params,self.max_grad_norm))
                        acc_grads = jax.tree_util.tree_map(
                            functools.partial(_acc_update),
                            grads, acc_grads)
                        if not flag._check_skip_next_step():
                            self.params,self.opt_state = jax.block_until_ready(self.grad_acc_update(acc_grads,self.opt_state,self.params))
                            gradient_step_ac += 1
                            print('flag queue',flag.skip_queue)
                            #print('here the step should be taken, the opt state:',self.opt_state.gradient_step,'count',gradient_step_ac)
                            print('batch_idx',batch_idx)
                            print(type(self.opt_state))
                            acc_grads = jax.tree_util.tree_map(jnp.zeros_like, self.params)
                            

                        batch_time = time.time() - start_time

                        add_scalar_dict(self.logger, #type: ignore
                                        'train_batch_memorystats',
                                        torch.cuda.memory_stats(),
                                        global_step=len(memory_safe_data_loader)*epoch + batch_idx)
                        add_scalar_dict(self.logger, 
                                    'resources',      # tag='resources/train/batch/...'
                                    self.collector.collect(),
                                    global_step=len(memory_safe_data_loader)*epoch + batch_idx)

                        metrics['loss'] = jnp.append(metrics['loss'],float(loss))
                        metrics['acc'] = jnp.append(metrics['acc'],(float(accu)))

                        batch_times.append(batch_time)
                        total_time_epoch += batch_time

                    if batch_idx % 100 == 99 or ((batch_idx + 1) == len(memory_safe_data_loader)):
                        
                        avg_loss = float(jnp.mean(metrics['loss']))
                        avg_acc = float(jnp.mean(metrics['acc']))
                        print(f'Epoch {epoch} Batch idx {batch_idx + 1} acc: {avg_acc} loss: {avg_loss}')
                        print('Accuracy values',metrics['acc'])
                        print('Loss values',metrics['loss'])
                        add_scalar_dict(self.logger,f'train_batch_stats',{'acc':avg_acc,'loss':avg_loss},global_step=len(memory_safe_data_loader)*epoch + batch_idx)
                        metrics['loss'] = jnp.array([])
                        metrics['acc'] = jnp.array([])
                        add_scalar_dict(self.logger,f'time batch',{f'batch time':batch_time},global_step=len(memory_safe_data_loader)*epoch + batch_idx)
            
            if epoch == 1:
                print('First Batch time \n',batch_times[0],'Second batch time',batch_times[1])

            epoch_time = time.time() - start_time_epoch

            print('Finish epoch',epoch,' batch_idx',batch_idx+1,'batch',len(batch),flush=True)

            eval_loss, eval_acc = self.eval_model(testloader)
            print('Epoch',epoch,'eval acc',eval_acc,'eval loss',eval_loss)
            add_scalar_dict(self.logger,'test_accuracy',{'accuracy eval':float(eval_acc),'loss eval':float(eval_loss)},global_step=epoch)


        
            epsilon = self.compute_epsilon(steps=int(gradient_step_ac),batch_size=expected_bs,target_delta=self.target_delta,noise_multiplier=self.noise_multiplier)
            
            privacy_results = {'eps_rdp':epsilon}
            add_scalar_dict(self.logger,'train_epoch_privacy',privacy_results,global_step=epoch)
            print('privacy results',privacy_results)

            throughput_t = (samples_used)/epoch_time
            throughput = (samples_used)/total_time_epoch
            print('total time epoch - epoch time',np.abs(total_time_epoch - epoch_time),'total time epoch',total_time_epoch,'epoch time',epoch_time)
            init_v = sample_sizes[0]
            for i in range(len(sample_sizes)):
                if sample_sizes[i] != init_v:
                    if i != 0:
                        print('before',sample_sizes[i-1],batch_times[i-1])
                    print('after',sample_sizes[i],batch_times[i])
                    init_v = sample_sizes[i]
            print('End of Epoch ', ' number of batches ',len(sample_sizes),np.column_stack((sample_sizes,batch_times)))

            if epoch == 1:
                throughput_wout_comp = (samples_used - self.physical_bs)/(total_time_epoch - batch_times[0])
                throughput_wout_t_comp = (samples_used - self.physical_bs)/(epoch_time - batch_times[0])
                print('throughput',throughput,'throughput minus the first time',throughput_wout_comp)
                throughput = throughput_wout_comp
                throughput_t = throughput_wout_t_comp
            throughputs[epoch-1] = throughput
            throughputs_t[epoch-1] = throughput_t
            if epoch == 1:
                comp_time = batch_times[0]
            print('Epoch {} Total time {} Throughput {} Samples Used {}'.format(epoch,total_time_epoch,throughput,samples_used),flush=True)  
        
        
        epsilon = self.compute_epsilon(steps=int(gradient_step_ac),batch_size=expected_bs,target_delta=self.target_delta,noise_multiplier=self.noise_multiplier)
        
        privacy_results = {'eps_rdp':epsilon}
        print('privacy results',privacy_results)
        print('Finish training',flush=True)
        return throughputs,throughputs_t,comp_time
    
    def non_private_training_mini_batch_2(self,trainloader,testloader):

        #Training
        print('Non private learning')
        
        _acc_update = lambda grad, acc : grad + acc

        self.calculate_noise(len(trainloader))
        self.init_non_optimizer()
        print('noise multiplier',self.noise_multiplier)
        throughputs = np.zeros(self.epochs)
        throughputs_t = np.zeros(self.epochs)
        expected_bs = len(trainloader.dataset)/len(trainloader)
        expected_acc_steps = expected_bs // self.physical_bs
        print('expected accumulation steps',expected_acc_steps)
        acc_grads = jax.tree_util.tree_map(jnp.zeros_like, self.params)
        comp_time = 0
        gradient_step_ac = 0
        for epoch in range(1,self.epochs+1):
            flag = EndingLogicalBatchSignal()
            batch_idx = 0
            metrics = {}
            metrics['loss'] = jnp.array([])
            metrics['acc'] = jnp.array([])
            
            total_time_epoch = 0
            samples_used = 0 
            start_time_epoch = time.time()
            batch_times = []

            steps = int(epoch * expected_acc_steps)

            print('steps',steps,flush=True)
            with MyBatchMemoryManager(
                data_loader=trainloader, 
                max_physical_batch_size=self.physical_bs, 
                signaler=flag
                ) as memory_safe_data_loader:
                print('memory safe data loader len ',len(memory_safe_data_loader.batch_sampler))
                for batch_idx, batch in enumerate(memory_safe_data_loader): 
                    with self.collector(tag='batch'):
                        samples_used += len(batch[0])
                        #print(samples_used)
                        start_time = time.time()
                        grads,loss,accu = jax.block_until_ready(self.non_private_update(self.params,batch))
                        acc_grads = jax.tree_util.tree_map(
                            functools.partial(_acc_update),
                            grads, acc_grads)
                        if not flag._check_skip_next_step():
                            self.params,self.opt_state = jax.block_until_ready(self.grad_acc_update(acc_grads,self.opt_state,self.params))  
                            gradient_step_ac += 1
                            print('flag queue',flag.skip_queue)
                            #print('here the step should be taken, the opt state:',self.opt_state.gradient_step,'count',gradient_step_ac)
                            print('batch_idx',batch_idx)
                            acc_grads = jax.tree_util.tree_map(jnp.zeros_like, self.params)
                                                        
                        batch_time = time.time() - start_time
                        

                        add_scalar_dict(self.logger, #type: ignore
                                        'train_batch_memorystats',
                                        torch.cuda.memory_stats(),
                                        global_step=len(memory_safe_data_loader)*epoch + batch_idx)
                        add_scalar_dict(self.logger, 
                                    'resources',      # tag='resources/train/batch/...'
                                    self.collector.collect(),
                                    global_step=len(memory_safe_data_loader)*epoch + batch_idx)
                        metrics['loss'] = jnp.append(metrics['loss'],float(loss))
                        metrics['acc'] = jnp.append(metrics['acc'],(float(accu)))
                        #batch_idx += 1
                        batch_times.append(batch_time)
                        total_time_epoch += batch_time

                    if batch_idx % 100 == 99 or ((batch_idx + 1) == len(memory_safe_data_loader)):
                        
                        avg_loss = float(jnp.mean(metrics['loss']))
                        avg_acc = float(jnp.mean(metrics['acc']))
                        print(f'Epoch {epoch} Batch idx {batch_idx + 1} acc: {avg_acc} loss: {avg_loss}')
                        add_scalar_dict(self.logger,f'train_batch_stats',{'acc':avg_acc,'loss':avg_loss},global_step=len(memory_safe_data_loader)*epoch + batch_idx)
                        metrics['loss'] = jnp.array([])
                        metrics['acc'] = jnp.array([])
                        add_scalar_dict(self.logger,f'time batch',{f'batch time':batch_time},global_step=len(memory_safe_data_loader)*epoch + batch_idx)            
            if epoch == 1:
                print('First Batch time \n',batch_times[0],'Second batch time',batch_times[1])

            epoch_time = time.time() - start_time_epoch

            print('Finish epoch',epoch,' batch_idx',batch_idx+1,'batch',len(batch),flush=True)

            eval_loss, eval_acc = self.eval_model(testloader)
            print('Epoch',epoch,'eval acc',eval_acc,'eval loss',eval_loss)
            add_scalar_dict(self.logger,'test_accuracy',{'accuracy eval':float(eval_acc),'loss eval':float(eval_loss)},global_step=epoch)

            throughput_t = (samples_used)/epoch_time
            throughput = (samples_used)/total_time_epoch
            print('total time epoch - epoch time',np.abs(total_time_epoch - epoch_time),'total time epoch',total_time_epoch,'epoch time',epoch_time)

            if epoch == 1:
                throughput_wout_comp = (samples_used - self.physical_bs)/(total_time_epoch - batch_times[0])
                throughput_wout_t_comp = (samples_used - self.physical_bs)/(epoch_time - batch_times[0])
                print('throughput',throughput,'throughput minus the first time',throughput_wout_comp)
                throughput = throughput_wout_comp
                throughput_t = throughput_wout_t_comp
            throughputs[epoch-1] = throughput
            throughputs_t[epoch-1] = throughput_t
            if epoch == 1:
                comp_time = batch_times[0]
            print('Epoch {} Total time {} Throughput {} Samples Used {}'.format(epoch,total_time_epoch,throughput,samples_used),flush=True)  
        
        print('Finish training',flush=True)
        return throughputs,throughputs_t,comp_time

    def eval_model(self, data_loader):
        # Test model on all images of a data loader and return avg loss
        accs = []
        losses = []
        for batch in data_loader:
            loss, acc = self.eval_step_non(self.params,batch)
            accs.append(float(acc))
            losses.append(float(loss))
        eval_acc = np.mean(accs)
        eval_loss = np.mean(losses)
        return eval_loss,eval_acc
      
    def load_model(self):
        print('load model name',self.model_name,flush=True)
        main_key, params_key= jax.random.split(key=self.rng,num=2)
        if self.model_name == 'small':
            class CNN(nn.Module):
                """A simple CNN model."""

                @nn.compact
                def __call__(self, x):
                    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
                    x = nn.relu(x)
                    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
                    #x = nn.Conv(features=64, kernel_size=(3, 3))(x)
                    #x = nn.relu(x)
                    #x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
                    x = x.reshape((x.shape[0], -1))  # flatten
                    x = nn.Dense(features=256)(x)
                    x = nn.relu(x)
                    x = nn.Dense(features=10)(x)
                    return x

            model = CNN()
            batch = jnp.ones((1,self.dimension,self.dimension,3))
            variables = model.init({'params':main_key}, batch)
            output = model.apply(variables, batch)
            self.model = model
            self.params = variables['params']
        
        elif 'vit' in self.model_name:
            model_name = self.model_name
            model = FlaxViTModel.from_pretrained(model_name)
            module = model.module # Extract the Flax Module
            vars = {'params': model.params} # Extract the parameters
            config = module.config
            model = ViTModelHead(num_classes=self.num_classes,vit=module)

            input_shape = (1,self.dimension,self.dimension,3)
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
def image_to_numpy(img):
    img = np.array(img, dtype=np.float32)
    img = (img / 255. - DATA_MEANS) / DATA_STD
    #There is the need of transposing the image. The image has the right dimension, but inside the ViT, it has a transpose where they move the number of channels to the last dim. So here I inverse 
    #that operation, so it works later during the pass
    return np.transpose(img)

def image_to_numpy_wo_t(img):
    img = np.array(img, dtype=np.float32)
    img = (img / 255. - DATA_MEANS) / DATA_STD
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
def load_data_cifar(ten,dimension,batch_size_train,physical_batch_size,num_workers,generator):

    print('load_data_cifar',batch_size_train,physical_batch_size,num_workers)

    w_batch = batch_size_train

    transformation = torchvision.transforms.Compose([
        torchvision.transforms.Resize(dimension),
        image_to_numpy_wo_t,
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
        testset, batch_size=100, shuffle=False,collate_fn=numpy_collate, num_workers=num_workers,generator=generator,worker_init_fn=seed_worker)

    return trainloader,testloader

def privatize_dataloader(data_loader):
    return DPDataLoader.from_data_loader(data_loader)

def main(args):
    print(args,flush=True)
    generator = set_seeds(args.seed)
    #Load data
    trainloader,testloader = load_data_cifar(args.ten,args.dimension,args.bs,args.phy_bs,args.n_workers,generator)
    if args.clipping_mode == 'mini':
        trainloader = privatize_dataloader(trainloader)
    print('data loaded',flush=True)
    #Create Trainer Module, that loads the model and train it
    trainer = TrainerModule(model_name=args.model,lr=args.lr,seed=args.seed,epochs=args.epochs,max_grad=args.grad_norm,accountant_method=args.accountant,batch_size=args.bs,physical_bs=args.phy_bs,target_epsilon=args.epsilon,target_delta=args.target_delta,num_classes=args.ten,test=args.test,dimension=args.dimension,clipping_mode=args.clipping_mode)
    if args.clipping_mode == 'non-private':
        throughputs,throughputs_t,comp_time = trainer.non_private_training_mini_batch_2(trainloader,testloader)
    elif args.clipping_mode == 'mini':
        throughputs,throughputs_t,comp_time = trainer.private_training_mini_batch_2(trainloader,testloader)
    tloss,tacc = trainer.eval_model(testloader)
    print('throughputs',throughputs,'mean throughput', np.mean(throughputs))
    print('compiling time',comp_time)
    print('test loss',tloss)
    print('test accuracy',tacc)
    return np.mean(throughputs),np.mean(throughputs_t),comp_time,tacc