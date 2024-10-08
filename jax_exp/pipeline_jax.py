import os
from typing import Any
import tensorflow as tf
tf.config.experimental.set_visible_devices([], 'GPU')

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".75"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"

# os.environ['XLA_FLAGS'] = (
#     '--xla_gpu_enable_triton_softmax_fusion=true '
#     '--xla_gpu_triton_gemm_any=True '
#     '--xla_gpu_enable_async_collectives=true '
#     '--xla_gpu_enable_latency_hiding_scheduler=true '
#     '--xla_gpu_enable_highest_priority_async_stream=true '
# )

from datetime import datetime
import warnings
import functools

import re
import numpy as np

#JAX
import jax
from jax import jit
import jax.numpy as jnp
import jax.profiler

#Flax
import flax.linen as nn
from flax.core.frozen_dict import unfreeze,freeze,FrozenDict

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
from transformers import FlaxViTModel,FlaxViTForImageClassification
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

@jit
def mini_batch_dif_clip2(per_example_grads,l2_norm_clip):
    
    grads_flat, grads_treedef = jax.tree_util.tree_flatten(per_example_grads)

    clipped, num_clipped = clipping.per_example_global_norm_clip(grads_flat, l2_norm_clip)

    grads_unflat = jax.tree_util.tree_unflatten(grads_treedef,clipped)

    return grads_unflat,num_clipped

@jit
def add_noise_fn(noise_std,rng_key,updates):
    
    num_vars = len(jax.tree_util.tree_leaves(updates))
    treedef = jax.tree_util.tree_structure(updates)
    new_key,*all_keys = jax.random.split(rng_key, num=num_vars + 1)
    noise = jax.tree_util.tree_map(
        lambda g, k: jax.random.normal(k, shape=g.shape, dtype=g.dtype),
        updates, jax.tree_util.tree_unflatten(treedef, all_keys))
    updates = jax.tree_util.tree_map(
        lambda g, n: (g + noise_std * n),
        updates, noise)

    return updates, new_key

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

        timestamp = datetime.now().strftime('%Y%m%d%M')
        print('model at time: ',timestamp,flush=True)
        self.logger = SummaryWriter('runs_{}/{}_{}_cifar_{}_epsilon_{}_model_{}_{}_{}'.format(target_epsilon,test,clipping_mode,num_classes,target_epsilon,model_name,timestamp,epochs),flush_secs=30)
        self.collector = ResourceMetricCollector(devices=CudaDevice.all(),
                                            root_pids={os.getpid()},
                                            interval=1.0)
        

        #self.create_functions()
        self.state = None
        self.rng = self.load_model()
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
    
    def init_optimizer(self):
        self.optimizer = optax.adam(learning_rate=self.lr)
        self.opt_state = self.optimizer.init(self.params)
    
    def calculate_metrics(self,params,batch):
        inputs,targets = batch
        logits = self.model(inputs,params=params)[0]
        predicted_class = jnp.argmax(logits,axis=-1)
        acc = jnp.mean(predicted_class==targets)
        return acc
    
    def loss(self,params,batch):
        inputs,targets = batch
        logits = self.model(inputs,params=params)[0]
        predicted_class = jnp.argmax(logits,axis=-1)

        cross_loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).sum()

        vals = predicted_class == targets
        acc = jnp.mean(vals)
        cor = jnp.sum(vals)

        return cross_loss,(acc,cor)

    def loss_eval(self,params,batch):
        inputs,targets = batch
        logits = self.model(inputs,params=params)[0]
        predicted_class = jnp.argmax(logits,axis=-1)
        cross_losses = optax.softmax_cross_entropy_with_integer_labels(logits, targets)

        cross_loss = jnp.mean(cross_losses)
        vals = predicted_class == targets
        acc = jnp.mean(vals)
        cor = jnp.sum(vals)

        return cross_loss,acc,cor
    
    #@partial(jit,static_argnums=0)
    def eval_step_non(self, params, batch):
        # Return the accuracy for a single batch
        loss,acc,cor =self.loss_eval(params,batch)
        return loss, acc,cor
    
    @partial(jit,static_argnums=0)
    def per_example_gradients(self,params,batch):
        batch = jax.tree_map(lambda x: x[:, None], batch)
        
        (loss_val,(acc,cor)), per_example_grads = jax.vmap(jax.value_and_grad(self.loss,has_aux=True),in_axes=(None,0))(params,batch)

        return per_example_grads,jnp.sum(loss_val),jnp.mean(acc),jnp.sum(cor)    
    
    @partial(jit, static_argnums=0)
    def mini_batch_dif_clip2(self,per_example_grads,l2_norm_clip):
        
        grads_flat, grads_treedef = jax.tree_util.tree_flatten(per_example_grads)

        clipped, num_clipped = clipping.per_example_global_norm_clip(grads_flat, l2_norm_clip)

        grads_unflat = jax.tree_util.tree_unflatten(grads_treedef,clipped)

        return grads_unflat,num_clipped

    #@partial(jit, static_argnums=0)
    def grad_acc_update(self,grads,opt_state,params):
        updates,new_opt_state = self.optimizer.update(grads,opt_state,params)
        new_params = optax.apply_updates(params,updates)
        return new_params,new_opt_state
    
    #@partial(jit, static_argnums=0)
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
    def add_noise_fn(self,noise_std,rng_key,updates):
        
        num_vars = len(jax.tree_util.tree_leaves(updates))
        treedef = jax.tree_util.tree_structure(updates)
        new_key,*all_keys = jax.random.split(rng_key, num=num_vars + 1)
        noise = jax.tree_util.tree_map(
            lambda g, k: jax.random.normal(k, shape=g.shape, dtype=g.dtype),
            updates, jax.tree_util.tree_unflatten(treedef, all_keys))
        updates = jax.tree_util.tree_map(
            lambda g, n: (g + noise_std * n),
            updates, noise)

        return updates, new_key
        
    
    def private_training_mini_batch(self,trainloader,testloader):

        #Training
        print('private learning',flush=True)
        
        _acc_update = lambda grad, acc : grad + acc

        self.calculate_noise(len(trainloader))
        self.init_optimizer()
        throughputs = np.zeros(self.epochs)
        throughputs_t = np.zeros(self.epochs)
        expected_bs = len(trainloader.dataset)/len(trainloader)
        expected_acc_steps = expected_bs // self.physical_bs
        print('expected accumulation steps',expected_acc_steps)

        @jit
        def add_noise_fn(noise_std,rng_key,updates):
            
            num_vars = len(jax.tree_util.tree_leaves(updates))
            treedef = jax.tree_util.tree_structure(updates)
            new_key,*all_keys = jax.random.split(rng_key, num=num_vars + 1)
            noise = jax.tree_util.tree_map(
                lambda g, k: jax.random.normal(k, shape=g.shape, dtype=g.dtype),
                updates, jax.tree_util.tree_unflatten(treedef, all_keys))
            updates = jax.tree_util.tree_map(
                lambda g, n: (g + noise_std * n),
                updates, noise)

            return updates, new_key
        
        comp_time = 0
        gradient_step_ac = 0
        for epoch in range(1,self.epochs+1):
            flag = EndingLogicalBatchSignal()
            batch_idx = 0
            metrics = {}
            metrics['loss'] = np.array([])
            metrics['acc'] = np.array([])
            
            total_time_epoch = 0
            samples_used = 0 
            start_time_epoch = time.time()
            batch_times = []
            sample_sizes = []

            steps = int(epoch * expected_acc_steps)

            accumulated_iterations = 0
            
            train_loss = 0
            correct = 0
            total = 0
            total_batch = 0
            correct_batch = 0
            batch_idx = 0

            acc_grads = jax.tree_util.tree_map(jnp.zeros_like, self.params)

            with MyBatchMemoryManager(
                data_loader=trainloader, 
                max_physical_batch_size=self.physical_bs, 
                signaler=flag
                ) as memory_safe_data_loader:
                for batch_idx, batch in enumerate(memory_safe_data_loader): 
                    samples_used += len(batch[0])
                    sample_sizes.append(len(batch[0]))
                    start_time = time.perf_counter()
                    per_grads,loss,accu,cor = jax.block_until_ready(self.per_example_gradients(self.params,batch))
                    grads,num_clipped = jax.block_until_ready(mini_batch_dif_clip2(per_grads,self.max_grad_norm))
                    acc_grads = add_trees(grads,acc_grads)
                    # acc_grads = jax.tree_map(lambda x,y: x+y, 
                    #                             grads, 
                    #                             acc_grads
                    #                             )
                    accumulated_iterations += 1
                    if not flag._check_skip_next_step():
                        print('about to update:')
                        updates,self.rng = add_noise_fn(self.noise_multiplier*self.max_grad_norm,self.rng,acc_grads)

                        self.params,self.opt_state = jax.block_until_ready(self.grad_acc_update(updates,self.opt_state,self.params))
                        
                        gradient_step_ac += 1
                        print('batch_idx',batch_idx)
                        print('count',gradient_step_ac)
                        acc_grads = jax.tree_util.tree_map(jnp.zeros_like, self.params)
                        accumulated_iterations = 0
                    batch_time = time.perf_counter() - start_time

                    train_loss += loss
                    total_batch += len(batch[1])
                    correct_batch += cor
                    metrics['loss'] = jnp.append(metrics['loss'],float(loss))
                    metrics['acc'] = jnp.append(metrics['acc'],(float(accu)))

                    batch_times.append(batch_time)
                    total_time_epoch += batch_time

                    if batch_idx % 100 == 99 or ((batch_idx + 1) == len(memory_safe_data_loader)):
                        
                        avg_loss = float(jnp.mean(metrics['loss']))
                        avg_acc = float(jnp.mean(metrics['acc']))
                        total += total_batch
                        correct += correct_batch
                        
                        print('(New)Accuracy values',100.*(correct_batch/total_batch))
                        print('(New)Loss values',train_loss)
                        #avg_acc = 100.*(correct/total)
                        #avg_loss = train_loss/total
                        print(f'Epoch {epoch} Batch idx {batch_idx + 1} acc: {avg_acc} loss: {avg_loss}')
                        print(f'Epoch {epoch} Batch idx {batch_idx + 1} acc: {100.*correct_batch/total_batch}')

                        metrics['loss'] = np.array([])
                        metrics['acc'] = np.array([])
                        
                        total_batch = 0
                        correct_batch = 0
                        
                        eval_loss, eval_acc,cor_eval,tot_eval = self.eval_model(testloader)
                        print('Epoch',epoch,'eval acc',eval_acc,cor_eval,'/',tot_eval,'eval loss',eval_loss,flush=True)

            print('-------------End Epoch---------------',flush=True)
            print('Finish epoch',epoch,' batch_idx',batch_idx+1,'batch',len(batch),flush=True)
            print('steps',steps,'gradient acc steps',gradient_step_ac,flush=True)
            print('Epoch: ', epoch, len(trainloader), 'Train Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total),flush=True)

            if epoch == 1:
                print('First Batch time \n',batch_times[0],'Second batch time',batch_times[1])

            epoch_time = time.time() - start_time_epoch
            eval_loss, eval_acc,cor_eval,tot_eval = self.eval_model(testloader)
            print('Epoch',epoch,'eval acc',eval_acc,cor_eval,'/',tot_eval,'eval loss',eval_loss,flush=True)

            epsilon,delta = self.compute_epsilon(steps=int(gradient_step_ac),batch_size=expected_bs,target_delta=self.target_delta,noise_multiplier=self.noise_multiplier)
            
            privacy_results = {'eps_rdp':epsilon,'delta_rdp':delta}
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
                throughput = throughput_wout_comp
                throughput_t = throughput_wout_t_comp
            throughputs[epoch-1] = throughput
            throughputs_t[epoch-1] = throughput_t
            if epoch == 1:
                comp_time = batch_times[0]
            print('Epoch {} Total time {} Throughput {} Samples Used {}'.format(epoch,total_time_epoch,throughput,samples_used),flush=True)  
        
        
        epsilon,delta = self.compute_epsilon(steps=int(gradient_step_ac),batch_size=expected_bs,target_delta=self.target_delta,noise_multiplier=self.noise_multiplier)
        
        privacy_results = {'eps_rdp':epsilon,'delta_rdp':delta}
        print('privacy results',privacy_results,flush=True)
        print('Finish training',flush=True)
        return throughputs,throughputs_t,comp_time,privacy_results

    def private_training(self,trainloader,testloader):

        #Training
        print('private learning',flush=True)
        
        self.calculate_noise(len(trainloader))
        self.init_optimizer()
        throughputs = np.zeros(self.epochs)
        throughputs_t = np.zeros(self.epochs)
        expected_bs = len(trainloader.dataset)/len(trainloader)
        expected_acc_steps = expected_bs // self.physical_bs
        print('expected accumulation steps',expected_acc_steps)
        
        comp_time = 0
        gradient_step_ac = 0
        for epoch in range(1,self.epochs+1):
            batch_idx = 0
            metrics = {}
            metrics['loss'] = np.array([])
            metrics['acc'] = np.array([])
            
            total_time_epoch = 0
            samples_used = 0 
            start_time_epoch = time.time()
            batch_times = []
            sample_sizes = []

            steps = int(epoch * expected_acc_steps)
            
            train_loss = 0
            correct = 0
            total = 0
            total_batch = 0
            correct_batch = 0
            batch_idx = 0

            #acc_grads = jax.tree_util.tree_map(jnp.zeros_like, self.params)

            for batch_idx, batch in enumerate(trainloader): 

                samples_used += len(batch[0])
                sample_sizes.append(len(batch[0]))
                start_time = time.perf_counter()
                grads,loss,accu,cor,num_clipped = jax.block_until_ready(self.mini_batch_dif_clip2(batch,self.params,self.max_grad_norm))

                    
                updates,self.rng = self.add_noise_fn(self.noise_multiplier*self.max_grad_norm,self.rng,grads)

                #old_params = self.params
                #self.params,self.opt_state = jax.block_until_ready(self.grad_acc_update(acc_grads,self.opt_state,self.params))
                self.params,self.opt_state = jax.block_until_ready(self.grad_acc_update(updates,self.opt_state,self.params))
                
                gradient_step_ac += 1
                print('batch idx',batch_idx,'with size',len(batch[0]))
                #self.print_param_change(old_params,self.params)
                #acc_grads = jax.tree_util.tree_map(jnp.zeros_like, self.params)

                batch_time = time.perf_counter() - start_time

                train_loss += loss
                total_batch += len(batch[1])
                correct_batch += cor
                metrics['loss'] = jnp.append(metrics['loss'],float(loss))
                metrics['acc'] = jnp.append(metrics['acc'],(float(accu)))

                batch_times.append(batch_time)
                total_time_epoch += batch_time

                if batch_idx % 100 == 99 or ((batch_idx + 1) == len(trainloader)):
                    
                    avg_loss = float(jnp.mean(metrics['loss']))
                    avg_acc = float(jnp.mean(metrics['acc']))
                    total += total_batch
                    correct += correct_batch
                    
                    print('(New)Accuracy values',100.*(correct_batch/total_batch))
                    print('(New)Loss values',train_loss)
                    #avg_acc = 100.*(correct/total)
                    #avg_loss = train_loss/total
                    print(f'Epoch {epoch} Batch idx {batch_idx + 1} acc: {avg_acc} loss: {avg_loss}')
                    print(f'Epoch {epoch} Batch idx {batch_idx + 1} acc: {100.*correct_batch/total_batch}')

                    metrics['loss'] = np.array([])
                    metrics['acc'] = np.array([])
                    
                    total_batch = 0
                    correct_batch = 0
                    
                    eval_loss, eval_acc,cor_eval,tot_eval = self.eval_model(testloader)
                    #eval_loss, eval_acc = self.eval_model(testloader)
                    print('Epoch',epoch,'eval acc',eval_acc,cor_eval,'/',tot_eval,'eval loss',eval_loss,flush=True)

            print('-------------End Epoch---------------',flush=True)
            print('Finish epoch',epoch,' batch_idx',batch_idx+1,'batch',len(batch),flush=True)
            print('steps',steps,'gradient acc steps',gradient_step_ac,flush=True)
            print('Epoch: ', epoch, len(trainloader), 'Train Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total),flush=True)

            if epoch == 1:
                print('First Batch time \n',batch_times[0],'Second batch time',batch_times[1])

            epoch_time = time.time() - start_time_epoch
            eval_loss, eval_acc,cor_eval,tot_eval = self.eval_model(testloader)
            #eval_loss, eval_acc = self.eval_model(testloader)
            print('Epoch',epoch,'eval acc',eval_acc,cor_eval,'/',tot_eval,'eval loss',eval_loss,flush=True)

            epsilon,delta = self.compute_epsilon(steps=int(gradient_step_ac),batch_size=expected_bs,target_delta=self.target_delta,noise_multiplier=self.noise_multiplier)
            
            privacy_results = {'eps_rdp':epsilon,'delta_rdp':delta}
            #add_scalar_dict(self.logger,'train_epoch_privacy',privacy_results,global_step=epoch)
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
                #print('throughput',throughput,'throughput minus the first time',throughput_wout_comp)
                throughput = throughput_wout_comp
                throughput_t = throughput_wout_t_comp
            throughputs[epoch-1] = throughput
            throughputs_t[epoch-1] = throughput_t
            if epoch == 1:
                comp_time = batch_times[0]
            print('Epoch {} Total time {} Throughput {} Samples Used {}'.format(epoch,total_time_epoch,throughput,samples_used),flush=True)  
        
        
        epsilon,delta = self.compute_epsilon(steps=int(gradient_step_ac),batch_size=expected_bs,target_delta=self.target_delta,noise_multiplier=self.noise_multiplier)
        
        privacy_results = {'eps_rdp':epsilon,'delta_rdp':delta}
        print('privacy results',privacy_results,flush=True)
        print('Finish training',flush=True)
        return throughputs,throughputs_t,comp_time,privacy_results

    def non_private_training_mini_batch(self,trainloader,testloader):

        #Training
        print('Non private learning virtual')
        
        #self.calculate_noise(len(trainloader))
        self.init_optimizer()
        print('self optimizer',self.optimizer)
        #print('self opt state',self.opt_state)
        #print('noise multiplier',self.noise_multiplier)
        throughputs = np.zeros(self.epochs)
        throughputs_t = np.zeros(self.epochs)
        expected_bs = len(trainloader.dataset)/len(trainloader)
        expected_acc_steps = expected_bs // self.physical_bs
        print('expected accumulation steps',expected_acc_steps,'len dataloader',len(trainloader),'expected_bs',expected_bs)
        _acc_update = lambda grad, acc : grad + acc / expected_acc_steps

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
            
            train_loss = 0
            correct = 0
            total = 0
            total_batch = 0
            correct_batch = 0
            batch_idx = 0
            accumulated_iterations = 0
            times_up = 0
            acc_grads = jax.tree_util.tree_map(jnp.zeros_like, self.params)

            with MyBatchMemoryManager(
                data_loader=trainloader, 
                max_physical_batch_size=self.physical_bs, 
                signaler=flag
                ) as memory_safe_data_loader:
                for batch_idx, batch in enumerate(memory_safe_data_loader): 
                    #with self.collector(tag='batch'):
                    batch = (jnp.array(batch[0]), jnp.array(batch[1]))
                    samples_used += len(batch[0])
                    #print(samples_used)
                    start_time = time.perf_counter()
                    grads,loss,accu,cor = jax.block_until_ready(self.non_private_update(self.params,batch))
                    acc_grads = add_trees(grads,acc_grads)

                    # acc_grads = jax.tree_util.tree_map(
                    #     lambda x,y: x+y,
                    #     grads, acc_grads)
                    accumulated_iterations += 1
                    if not flag._check_skip_next_step():
                        print('about to update:')
                        #acc_grads = jax.tree_util.tree_map(
                        #    lambda x: x/expected_bs*accumulated_iterations,
                        #    acc_grads)
                        #old_params = self.params
                        self.params,self.opt_state = jax.block_until_ready(self.grad_acc_update(acc_grads,self.opt_state,self.params))  
                        gradient_step_ac += 1
                        #print('flag queue',flag.skip_queue)
                        #print('here the step should be taken, the opt state:',self.opt_state.gradient_step,'count',gradient_step_ac)
                        print('batch_idx',batch_idx)
                        #self.print_param_change(old_params,self.params)
                        acc_grads = jax.tree_util.tree_map(jnp.zeros_like, self.params)
                        times_up += 1
                        accumulated_iterations = 0 

                    #jax.block_until_ready()
                                                    
                    batch_time = time.perf_counter() - start_time
                    train_loss += loss / expected_acc_steps
                    total_batch += len(batch[1])
                    correct_batch += cor
                    metrics['loss'] = jnp.append(metrics['loss'],float(loss))
                    metrics['acc'] = jnp.append(metrics['acc'],(float(accu)))
                    batch_times.append(batch_time)
                    total_time_epoch += batch_time

                    if batch_idx % 100 == 99 or ((batch_idx + 1) == len(memory_safe_data_loader)):
                        
                        avg_loss = float(jnp.mean(metrics['loss']))
                        avg_acc = float(jnp.mean(metrics['acc']))
                        total += total_batch
                        correct += correct_batch
                        new_loss = train_loss/len(metrics['loss'])
                        print('(New)Accuracy values',100.*(correct_batch/total_batch))
                        print('(New)Loss values',(new_loss))
                        print(f'Epoch {epoch} Batch idx {batch_idx + 1} acc: {avg_acc} loss: {new_loss}')
                        print(f'Epoch {epoch} Batch idx {batch_idx + 1} acc: {100.*correct_batch/total_batch}')
                        print('Update metrics')
                        metrics['loss'] = np.array([])
                        metrics['acc'] = np.array([])
                        
                        eval_loss, eval_acc,cor_eval,tot_eval = self.eval_model(testloader)
                        #eval_loss, eval_acc = self.eval_model(testloader)
                        print('Epoch',epoch,'eval acc',eval_acc,cor_eval,'/',tot_eval,'eval loss',eval_loss,flush=True)

                        total_batch = 0
                        correct_batch = 0
                    
        
            print('-------------End Epoch---------------',flush=True)
            print('Finish epoch',epoch,' batch_idx',batch_idx+1,'batch',len(batch),flush=True)
            print('steps',steps,'gradient acc steps',gradient_step_ac,'times updated',times_up,flush=True)
            print('Epoch: ', epoch, len(trainloader), 'Train Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (train_loss/(len(trainloader)), 100.*correct/total, correct, total),flush=True)
            
            if epoch == 1:
                print('First Batch time \n',batch_times[0],'Second batch time',batch_times[1])

            epoch_time = time.time() - start_time_epoch

            print('Finish epoch',epoch,' batch_idx',batch_idx+1,'batch',len(batch),flush=True)

            eval_loss, eval_acc,cor_eval,tot_eval = self.eval_model(testloader)
            print('Epoch',epoch,'eval acc',eval_acc,cor_eval,'/',tot_eval,'eval loss',eval_loss,flush=True)
            print('batch_idx',batch_idx,'samples used',samples_used,'samples used / batch_idx',samples_used/batch_idx,'physical batch size',self.physical_bs,flush=True)
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
    
    def non_private_training(self,trainloader,testloader):

        #Training
        print('Non private learning')
        
        #self.calculate_noise(len(trainloader))
        self.init_optimizer()
        print('self optimizer',self.optimizer)
        print('self opt state',self.opt_state)
        #print('noise multiplier',self.noise_multiplier)
        throughputs = np.zeros(self.epochs)
        throughputs_t = np.zeros(self.epochs)
        expected_bs = len(trainloader.dataset)/len(trainloader)
        expected_acc_steps = expected_bs // self.physical_bs
        print('expected accumulation steps',expected_acc_steps,'len dataloader',len(trainloader),'expected_bs',expected_bs)
        comp_time = 0
        gradient_step_ac = 0
        for epoch in range(1,self.epochs+1):
            batch_idx = 0
            metrics = {}
            metrics['loss'] = jnp.array([])
            metrics['acc'] = jnp.array([])
            
            total_time_epoch = 0
            samples_used = 0 
            start_time_epoch = time.time()
            batch_times = []

            steps = int(epoch * expected_acc_steps)
            
            train_loss = 0
            correct = 0
            total = 0
            total_batch = 0
            correct_batch = 0
            batch_idx = 0
            
            times_up = 0
            #acc_grads = jax.tree_util.tree_map(jnp.zeros_like, self.params)

            for batch_idx, batch in enumerate(trainloader): 
                batch = (jnp.array(batch[0]), jnp.array(batch[1]))
                #with self.collector(tag='batch'):
                samples_used += len(batch[0])
                print('batch idx',batch_idx,'with size',len(batch[0]))
                start_time = time.perf_counter()
                grads,loss,accu,cor = jax.block_until_ready(self.non_private_update(self.params,batch))
                acc_grads = jax.tree_util.tree_map(
                        lambda x: x/len(batch[0]),
                        grads)
                    #old_params = self.params
                self.params,self.opt_state = jax.block_until_ready(self.grad_acc_update(acc_grads,self.opt_state,self.params))  
                
                #jax.block_until_ready()
                                                
                batch_time = time.perf_counter() - start_time
                
                train_loss += loss / len(batch[0])
                total_batch += len(batch[1])
                correct_batch += cor
                metrics['loss'] = jnp.append(metrics['loss'],float(loss))
                metrics['acc'] = jnp.append(metrics['acc'],(float(accu)))
                batch_times.append(batch_time)
                total_time_epoch += batch_time

                if batch_idx % 100 == 99 or ((batch_idx + 1) == len(trainloader)):
                    
                    avg_loss = float(jnp.mean(metrics['loss']))
                    avg_acc = float(jnp.mean(metrics['acc']))
                    total += total_batch
                    correct += correct_batch
                    new_loss = train_loss/len(metrics['loss'])
                    print('(New)Accuracy values',100.*(correct_batch/total_batch))
                    print('(New)Loss values',(new_loss))
                    print(f'Epoch {epoch} Batch idx {batch_idx + 1} acc: {avg_acc} loss: {new_loss}')
                    print(f'Epoch {epoch} Batch idx {batch_idx + 1} acc: {100.*correct_batch/total_batch}')
                    print('Update metrics')
                    metrics['loss'] = np.array([])
                    metrics['acc'] = np.array([])
                    
                    eval_loss, eval_acc,cor_eval,tot_eval = self.eval_model(testloader)
                    #eval_loss, eval_acc = self.eval_model(testloader)
                    print('Epoch',epoch,'eval acc',eval_acc,cor_eval,'/',tot_eval,'eval loss',eval_loss,flush=True)

                    total_batch = 0
                    correct_batch = 0
                    
        
            print('-------------End Epoch---------------',flush=True)
            print('Finish epoch',epoch,' batch_idx',batch_idx+1,'batch',len(batch),flush=True)
            print('steps',steps,'gradient acc steps',gradient_step_ac,'times updated',times_up,flush=True)
            print('Epoch: ', epoch, len(trainloader), 'Train Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (train_loss/(len(trainloader)), 100.*correct/total, correct, total),flush=True)
            
            if epoch == 1:
                print('First Batch time \n',batch_times[0],'Second batch time',batch_times[1])

            epoch_time = time.time() - start_time_epoch

            print('Finish epoch',epoch,' batch_idx',batch_idx+1,'batch',len(batch),flush=True)

            eval_loss, eval_acc,cor_eval,tot_eval = self.eval_model(testloader)
            print('Epoch',epoch,'eval acc',eval_acc,cor_eval,'/',tot_eval,'eval loss',eval_loss,flush=True)
            print('batch_idx',batch_idx,'samples used',samples_used,'samples used / batch_idx',samples_used/batch_idx,'physical batch size',self.physical_bs,flush=True)
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

    def print_param_values(self,params):
        jax.tree_util.tree_map(lambda x: print(f"Shape: {x.shape}, Values: {x}"), params)
    
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
            model = FlaxViTForImageClassification.from_pretrained(model_name, num_labels=self.num_classes, return_dict=False, ignore_mismatched_sizes=True)
            self.model = model
            self.params = model.params
            #model = FlaxViTForImageClassification.from_pretrained(model_name)
            # model = FlaxViTModel.from_pretrained(model_name,add_pooling_layer=False)
            # module = model.module # Extract the Flax Module
            # vars = {'params': model.params} # Extract the parameters
            # #config = module.config
            # model = ViTModelHead(num_classes=self.num_classes,pretrained_model=model)

            # input_shape = (1,3,self.dimension,self.dimension)
            # #But then, we need to split it in order to get random numbers
            

            # #The init function needs an example of the correct dimensions, to infer the dimensions.
            # #They are not explicitly writen in the module, instead, the model infer them with the first example.
            # x = jax.random.normal(params_key, input_shape)

            # main_rng, init_rng, dropout_init_rng = jax.random.split(main_key, 3)
            # #Initialize the model
            # variables = jax.jit(model.init)({'params':init_rng},x)

            # #So far, the parameters are initialized randomly, so we need to unfreeze them and add the pre loaded parameters.
            # params = variables['params']
            # params['vit'] = vars['params']
            # #params = unfreeze(params)
            # #self.print_param_shapes(params)
            # #print(params)
            # #model.apply({'params':params},x)
            # self.model = model
            # self.params = freeze(params)

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
        self.print_param_shapes(self.params)
        #self.print_param_values(params)
        return main_key
        
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

def privatize_dataloader(data_loader):
    return DPDataLoader.from_data_loader(data_loader)

@jax.jit
def add_trees(x, y):
    #Helper function, add two tree objects
    return jax.tree_util.tree_map(lambda a, b: a + b, x, y)


def main(args):
    print(args,flush=True)
    print('devices ',jax.devices(),flush=True)
    generator = set_seeds(args.seed)
    
    #Load data
    trainloader,testloader = load_data_cifar(args.ten,args.dimension,args.bs,args.phy_bs,args.n_workers,generator,args.normalization)

    trainloader = privatize_dataloader(trainloader)
    print('data loaded',flush=True)
    
    #Create Trainer Module, that loads the model and train it
    trainer = TrainerModule(model_name=args.model,lr=args.lr,seed=args.seed,epochs=args.epochs,max_grad=args.grad_norm,accountant_method=args.accountant,batch_size=args.bs,physical_bs=args.phy_bs,target_epsilon=args.epsilon,target_delta=args.target_delta,num_classes=args.ten,test=args.test,dimension=args.dimension,clipping_mode=args.clipping_mode)
    
    #Test initial model without training
    tloss,tacc,cor_eval,tot_eval = trainer.eval_model(testloader)
    print('Without trainig test loss',tloss)
    print('Without training test accuracy',tacc,'(',cor_eval,'/',tot_eval,')')
    
    if args.clipping_mode == 'non-private':
        throughputs,throughputs_t,comp_time = trainer.non_private_training(trainloader,testloader)
    elif args.clipping_mode == 'non-private-virtual':
        throughputs,throughputs_t,comp_time = trainer.non_private_training_mini_batch(trainloader,testloader)
    elif args.clipping_mode == 'private-mini':
        throughputs,throughputs_t,comp_time,privacy_measures = trainer.private_training_mini_batch(trainloader,testloader)
        print(privacy_measures)
    elif args.clipping_mode == 'private':
        throughputs,throughputs_t,comp_time,privacy_measures = trainer.private_training(trainloader,testloader)
        print(privacy_measures)
    tloss,tacc,cor_eval,tot_eval = trainer.eval_model(testloader)
    print('throughputs',throughputs,'mean throughput', np.mean(throughputs))
    print('compiling time',comp_time)
    print('test loss',tloss)
    print('test accuracy',tacc)
    print('(',cor_eval,'/',tot_eval,')')
    return np.mean(throughputs),np.mean(throughputs_t),comp_time,tacc