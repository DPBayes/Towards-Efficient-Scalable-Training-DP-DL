import jax
import jax.numpy as jnp
import numpy as np
import torchvision
import torch.utils.data as data
from opacus.data_loader import DPDataLoader
import torch
import flax.linen as nn
from transformers import FlaxViTModel,FlaxViTForImageClassification
from private_vit import ViTModelHead
from MyOwnBatchManager import MyBatchMemoryManager,EndingLogicalBatchSignal
import optax
from jax import jit
#DP-Accounting - JAX/Flax doesn't have their own as Opacus
from dp_accounting import dp_event,rdp
import argparse
import time

from flax.training import train_state

import warnings

def image_to_numpy_wo_t(img):
    img = np.array(img, dtype=np.float32)
    img = ((img / 255.) - np.array([0.5, 0.5, 0.5])) / np.array([0.5,0.5, 0.5])
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

def seed_worker(worker_id):

    #print(torch.initial_seed(),flush=True)

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

def load_data_cifar(ten,dimension,batch_size_train,physical_batch_size,num_workers,generator,norm):

    print('load_data_cifar',batch_size_train,physical_batch_size,num_workers)

    w_batch = batch_size_train

    fn = image_to_numpy_wo_t

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

def print_param_shapes(params, prefix=''):
    for key, value in params.items():
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            print_param_shapes(value, prefix + '  ')
        else:
            print(f"{prefix}{key}: {value.shape}")


def compute_epsilon(steps,batch_size, num_examples=60000, target_delta=1e-5,noise_multiplier=0.1):
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

def create_train_state(model,lr):
    optimizer = optax.adam(lr)
    return train_state.TrainState.create(apply_fn=model.__call__,params=model.params,tx=optimizer)

def load_model(model_name,rng,dimension,num_classes):
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
        return model,params

    elif 'vit' in model_name:
        model_name = model_name
        # model = FlaxViTModel.from_pretrained(model_name,add_pooling_layer=False)
        # module = model.module # Extract the Flax Module
        # vars = {'params': model.params} # Extract the parameters
        # config = module.config
        # model = ViTModelHead(num_classes=num_classes,pretrained_model=model)

        # input_shape = (1,3,dimension,dimension)
        # #But then, we need to split it in order to get random numbers
        

        # #The init function needs an example of the correct dimensions, to infer the dimensions.
        # #They are not explicitly writen in the module, instead, the model infer them with the first example.
        # x = jax.random.normal(params_key, input_shape)

        # main_rng, init_rng, dropout_init_rng = jax.random.split(main_key, 3)
        # #Initialize the model
        # variables = model.init({'params':init_rng},x)

        # #So far, the parameters are initialized randomly, so we need to unfreeze them and add the pre loaded parameters.
        # params = variables['params']
        # params['vit'] = vars['params']
        # print_param_shapes(params)
        # #print(params)
        # model.apply({'params':params},x)
        #model = model
        #params = params

        model = FlaxViTForImageClassification.from_pretrained(model_name,num_labels=num_classes,return_dict=False)
        # Reinitialize the classification head
        #model.config.num_labels = num_classes
        #model = FlaxViTForImageClassification(model.config)
        #model = FlaxViTModel.from_pretrained(model_name,add_pooling_layer=False)
        # Initialize the model
        params = model.params
        #self.model = model
        #self.params = unfreeze(params)
        return model, params

def loss(params,model,batch):
    inputs,targets = batch
    logits = model.apply({'params':params},inputs)
    predicted_class = jnp.argmax(logits,axis=-1)

    cross_loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).sum()

    vals = predicted_class == targets
    acc = jnp.mean(vals)
    cor = jnp.sum(vals)

    return cross_loss,(acc,cor)
@jit
def loss_eval(params,model,batch):
    inputs,targets = batch
    logits = model.apply({'params':params},inputs)
    predicted_class = jnp.argmax(logits,axis=-1)
    cross_losses = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    cross_loss = jnp.mean(cross_losses)
    vals = predicted_class == targets
    acc = jnp.mean(vals)
    cor = jnp.sum(vals)
    return cross_loss,acc,cor
    
def eval_step_non(model,params, batch):
    loss,acc,cor = loss_eval(params,model,batch)
    return loss, acc,cor

@jit
def train_step(state,batch):
    def loss_fn(params):
        inputs,targets = batch
        logits  = state.apply_fn(inputs,params = params,train=True)[0]
        predicted_class = jnp.argmax(logits,axis=-1)

        cross_loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
        acc = jnp.mean(predicted_class ==targets)
        
        return cross_loss,acc
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, acc), grads = grad_fn(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss, acc


@jit
def train_step_private(state,batch):
    def loss_fn(params):
        inputs,targets = batch
        logits  = state.apply_fn(inputs,params = params,train=True)[0]
        predicted_class = jnp.argmax(logits,axis=-1)

        cross_loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
        acc = jnp.mean(predicted_class ==targets)
        
        return cross_loss,acc
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, acc), grads = grad_fn(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss, acc

def train_epoch(state,data_loader):

    #def body_fn(state, batch):
    #    new_state, loss, accuracy = train_step(state, batch)
    #    return new_state, (loss, accuracy)
    
    batch_idx = 0
    train_loss = 0
    train_acc = 0
    for batch_idx,batch in enumerate(data_loader):
        inputs,targets = jnp.array(batch[0]),jnp.array(batch[1])
        state, loss, accuracy = train_step(state, (inputs,targets))
        train_loss += loss
        train_acc += accuracy
        print('batch idx',batch_idx,loss/len(batch[0]),accuracy/len(batch[0]))
    

    return state, (train_loss/len(data_loader),train_acc//len(data_loader))

@jit
def eval_step(state,batch):
    inputs,targets = batch
    outputs = state.apply_fn(inputs,params = state.params,train=True)[0]
    predicted_class = jnp.argmax(outputs,axis=-1)

    #cross_losses = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    #cross_loss = jnp.mean(cross_losses)
    vals = predicted_class == targets
    return jnp.mean(vals)

def eval_model(state,data_loader):
    # Test model on all images of a data loader and return avg loss
    accuracies = 0
    batch_idx = 0
    for batch_idx,batch in enumerate(data_loader):
        #print('test loader batch idx',batch_idx,flush=True)
        acc = eval_step(state,batch)
        accuracies += acc
        del batch
    eval_acc = accuracies/len(data_loader)
    return eval_acc

def init_non_optimizer(lr,params):
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)
    return optimizer,opt_state
@jit
def grad_acc_update(grads,optimizer,opt_state,params):
    updates,opt_state = optimizer.update(grads,opt_state,params)
    params = optax.apply_updates(params,updates)
    return params,opt_state

@jit
def non_private_update(params,model,batch):
    (loss_val,(acc,cor)), grads = jax.value_and_grad(loss,has_aux=True)(params,model,batch)
    return grads,loss_val,acc,cor

def non_private_training_mini_batch_clean(trainloader,testloader,params,model,epochs,physical_bs,lr):

    #Training
    print('Non private learning with mini batches')
    
    #calculate_noise(len(trainloader))
    optimizer,opt_state = init_non_optimizer(lr,params)
    #print('noise multiplier',noise_multiplier)
    throughputs = np.zeros(epochs)
    throughputs_t = np.zeros(epochs)
    expected_bs = len(trainloader.dataset)/len(trainloader)
    expected_acc_steps = expected_bs // physical_bs
    print('expected accumulation steps',expected_acc_steps,'len dataloader',len(trainloader),'expected_bs',expected_bs)
    _acc_update = lambda grad, acc : grad + acc / expected_acc_steps

    comp_time = 0
    gradient_step_ac = 0
    for epoch in range(1,epochs+1):
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
        
        times_up = 0
        acc_grads = jax.tree_util.tree_map(jnp.zeros_like, params)
        samples_batch = 0

        with MyBatchMemoryManager(
            data_loader=trainloader, 
            max_physical_batch_size=physical_bs, 
            signaler=flag
            ) as memory_safe_data_loader:
            for batch_idx, batch in enumerate(memory_safe_data_loader): 
                #with collector(tag='batch'):
                samples_used += len(batch[0])
                samples_batch += len(batch[0])
                #print(samples_used)
                start_time = time.perf_counter()
                grads,loss,accu,cor = jax.block_until_ready(non_private_update(params,model,batch))
                acc_grads = jax.tree_util.tree_map(
                    lambda x,y: x+y,
                    grads, acc_grads)
                if not flag._check_skip_next_step():
                    print('about to update:')
                    acc_grads = jax.tree_util.tree_map(
                        lambda x: x/samples_batch,
                        acc_grads)
                    #old_params = params
                    params,opt_state = jax.block_until_ready(grad_acc_update(acc_grads,optimizer,opt_state,params))  
                    gradient_step_ac += 1
                    #print('flag queue',flag.skip_queue)
                    #print('here the step should be taken, the opt state:',opt_state.gradient_step,'count',gradient_step_ac)
                    print('batch_idx',batch_idx)
                    #print_param_change(old_params,params)
                    acc_grads = jax.tree_util.tree_map(jnp.zeros_like, params)
                    times_up += 1
                    print('samples used in logical batch size')
                    samples_batch = 0

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
                    
                    eval_loss, eval_acc,cor_eval,tot_eval = eval_model(testloader,model,params)
                    #eval_loss, eval_acc = eval_model(testloader)
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

        eval_loss, eval_acc,cor_eval,tot_eval = eval_model(testloader,model,params)
        print('Epoch',epoch,'eval acc',eval_acc,cor_eval,'/',tot_eval,'eval loss',eval_loss,flush=True)
        print('batch_idx',batch_idx,'samples used',samples_used,'samples used / batch_idx',samples_used/batch_idx,'physical batch size',physical_bs,flush=True)
        throughput_t = (samples_used)/epoch_time
        throughput = (samples_used)/total_time_epoch
        print('total time epoch - epoch time',np.abs(total_time_epoch - epoch_time),'total time epoch',total_time_epoch,'epoch time',epoch_time)

        if epoch == 1:
            throughput_wout_comp = (samples_used - physical_bs)/(total_time_epoch - batch_times[0])
            throughput_wout_t_comp = (samples_used - physical_bs)/(epoch_time - batch_times[0])
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


def non_private_training_clean(trainloader,testloader,epochs,physical_bs,model,lr):

    #Training
    print('Non private learning',flush=True)
    
    #calculate_noise(len(trainloader))
    optimizer,opt_state = init_non_optimizer(lr,params)
    #print('noise multiplier',noise_multiplier)
    throughputs = np.zeros(epochs)
    throughputs_t = np.zeros(epochs)
    expected_bs = len(trainloader.dataset)/len(trainloader)
    expected_acc_steps = expected_bs // physical_bs
    print('expected accumulation steps',expected_acc_steps,'len dataloader',len(trainloader),'expected_bs',expected_bs)
    comp_time = 0
    gradient_step_ac = 0
    for epoch in range(1,epochs+1):
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

        for batch_idx, batch in enumerate(trainloader): 

            samples_used += len(batch[0])

            start_time = time.perf_counter()
            grads,loss,accu,cor = jax.block_until_ready(non_private_update(params,model,batch))

            params,opt_state = jax.block_until_ready(grad_acc_update(grads,optimizer,opt_state,params))  
            times_up += 1                                
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
                new_loss = train_loss/len(metrics['loss'])
                print('(New)Accuracy values',100.*(correct_batch/total_batch))
                print('(New)Loss values',(new_loss))
                print(f'Epoch {epoch} Batch idx {batch_idx + 1} acc: {avg_acc} loss: {new_loss}')
                print(f'Epoch {epoch} Batch idx {batch_idx + 1} acc: {100.*correct_batch/total_batch}')
                print('Update metrics')
                metrics['loss'] = np.array([])
                metrics['acc'] = np.array([])
                
                eval_loss, eval_acc,cor_eval,tot_eval = eval_model(testloader,model,params)
                #eval_loss, eval_acc = eval_model(testloader)
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

        eval_loss, eval_acc,cor_eval,tot_eval = eval_model(testloader,model,params)
        print('Epoch',epoch,'eval acc',eval_acc,cor_eval,'/',tot_eval,'eval loss',eval_loss,flush=True)
        print('batch_idx',batch_idx,'samples used',samples_used,'samples used / batch_idx',samples_used/batch_idx,'physical batch size',physical_bs,flush=True)
        throughput_t = (samples_used)/epoch_time
        throughput = (samples_used)/total_time_epoch
        print('total time epoch - epoch time',np.abs(total_time_epoch - epoch_time),'total time epoch',total_time_epoch,'epoch time',epoch_time)

        if epoch == 1:
            throughput_wout_comp = (samples_used - physical_bs)/(total_time_epoch - batch_times[0])
            throughput_wout_t_comp = (samples_used - physical_bs)/(epoch_time - batch_times[0])
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


def main(args):
    generator = set_seeds(args.seed)
    trainloader,testloader = load_data_cifar(args.ten,args.dimension,args.bs,args.phy_bs,args.n_workers,generator,args.normalization)
    trainloader = privatize_dataloader(trainloader)
    rng = jax.random.PRNGKey(args.seed)
    
    model,params = load_model(args.model,rng,args.dimension,args.ten)

    state = create_train_state(model,args.lr)
    print('evaluating model before training',flush=True)

    test_acc = eval_model(state,testloader)


    #tloss,tacc,cor_eval,tot_eval = eval_model(testloader,model,params)
    print('Without trainig test acc',test_acc)

    for e in args.epochs:
        state, (train_loss, train_accuracy) = train_epoch(state,trainloader)
        test_acc = eval_model(state,testloader)
        print('epoch',e,'test_acc',test_acc)

    print(train_loss,train_accuracy)
    #print('Without training test accuracy',tacc,'(',cor_eval,'/',tot_eval,')',flush=True)
    #if args.clipping_mode == 'non-private-virtual':
    #    throughputs,throughputs_t,comp_time = non_private_training_mini_batch_clean(trainloader,testloader,params,model,args.epochs,args.phy_bs,args.lr)
    #elif args.clipping_mode == 'non-private':
    #    throughputs,throughputs_t,comp_time = non_private_training_clean(trainloader,testloader,params,model,args.epochs,args.phy_bs,args.lr)
    
    #elif args.clipping_mode == 'mini':
    #    throughputs,throughputs_t,comp_time,privacy_measures = private_training_mini_batch_clean(trainloader,testloader)
    test_acc = eval_model(state,testloader)
    #tloss,tacc,cor_eval,tot_eval = eval_model(testloader)
    #print('throughputs',throughputs,'mean throughput', np.mean(throughputs))
    #print('compiling time',comp_time)
    #print('test loss',tloss)
    print('test accuracy',test_acc)
    #print('(',cor_eval,'/',tot_eval,')')
    return test_acc
    #return np.mean(throughputs),np.mean(throughputs_t),comp_time,test_acc

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='JAX ViT CIFAR Training')
    parser.add_argument('--lr', default=0.0005, type=float, help='learning rate')
    parser.add_argument('--epochs', default=3, type=int,
                        help='numter of epochs')
    parser.add_argument('--bs', default=1000, type=int, help='batch size')
    parser.add_argument('--epsilon', default=8, type=float, help='target epsilon')
    parser.add_argument('--clipping_mode', default='non-private', type=str)
    parser.add_argument('--model', default='google/vit-base-patch16-224', type=str)
    parser.add_argument('--dimension', type=int,default=224)
    parser.add_argument('--origin_params', nargs='+', default=None)
    parser.add_argument('--ten',default=10,type=int)
    parser.add_argument('--n_workers',default=10,type=int)
    parser.add_argument('--phy_bs',default=100,type=int,help='Physical Batch')
    parser.add_argument('--accountant',default='rdp',type=str)
    parser.add_argument('--grad_norm', '-gn', default=1,
                        type=float, help='max grad norm')
    parser.add_argument('--target_delta',default=1e-5,type=float,help='target delta')
    parser.add_argument('--seed',default=1234,type=int)
    parser.add_argument('--normalization',default='True',type=str)
    parser.add_argument('--test',type=str,default='train')
    parser.add_argument('--file',type=str,default='thr_record.csv')

    args = parser.parse_args()

    main(args)