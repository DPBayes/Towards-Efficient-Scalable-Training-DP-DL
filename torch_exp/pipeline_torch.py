"""
So, I need multiple Privacy Engine. 

Opacus - Normal DP
from opacus import PrivacyEngine

Private Vision - Ghost Clipping Vision
from private_vision import PrivacyEngine

BK: 
from fastDP import PrivacyEngine
"""
import os
from opacus import PrivacyEngine as PrivacyEngineOpacus
from fastDP import PrivacyEngine as PrivacyEngineBK
from private_vision import PrivacyEngine as PrivacyEngineVision

#import pdb
import torch
import numpy as np
import random
#import torchvision
from torchvision import datasets,transforms

import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn
import timm
from opacus.validators import ModuleValidator
from opacus.accountants.utils import get_noise_multiplier
from opacus.utils.batch_memory_manager import BatchMemoryManager
from MyOwnBatchManager import MyBatchMemoryManager,EndingLogicalBatchSignal
from opacus.data_loader import DPDataLoader, switch_generator
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime
#from nvitop.callbacks.tensorboard import add_scalar_dict
#from nvitop import CudaDevice,ResourceMetricCollector
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from torch.nn.parallel import DistributedDataParallel as DDP

import warnings; warnings.filterwarnings("ignore")

import models
from functools import partial

import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group

import csv

#Defines each worker seed. Since each worker needs a different seed.
#The worker_id is a parameter given by the loader, but it is not used inside the method
def seed_worker(worker_id):

    #print(torch.initial_seed(),flush=True)

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

#Set seeds.
#Returns the generator, that will be used for the data loader
def set_seeds(seed,device):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)

    g_cuda = torch.Generator(device)

    g_cuda.manual_seed(seed)

    g_cpu = torch.Generator('cpu')

    g_cpu.manual_seed(seed)

    np.random.seed(seed)
    
    print('set seeds seed',seed,flush=True)

    print(torch.initial_seed(),flush=True)

    return g_cuda,g_cpu

def load_data_cifar(ten,dimension,batch_size_train,physical_batch_size,num_workers,normalization,lib,generator,world_size):

    print('load_data_cifar',lib,batch_size_train,physical_batch_size,num_workers)

    if normalization:
        means = (0.5,0.5,0.5)
        stds = (0.5,0.5,0.5)
    else:
        means = (0.485, 0.456, 0.406)
        stds =  (0.229, 0.224, 0.225)

    transformation = transforms.Compose([
        transforms.Resize(dimension),
        transforms.ToTensor(),
        transforms.Normalize(means, stds),
    ])
    
    if ten==10:
        trainset = datasets.CIFAR10(root='../data_cifar10/', train=True, download=True, transform=transformation)
        testset = datasets.CIFAR10(root='../data_cifar10/', train=False, download=True, transform=transformation)
    else:
        trainset = datasets.CIFAR100(root='../data_cifar100/', train=True, download=True, transform=transformation)
        testset = datasets.CIFAR100(root='../data_cifar100/', train=False, download=True, transform=transformation)

    if lib == 'non':
        trainloader = torch.utils.data.DataLoader(
            trainset, 
            #batch_size=batch_size_train if lib == 'opacus' else physical_batch_size,  #If it is opacus, it uses the normal batch size, because is the BatchMemoryManager the one that handles the phy and bs sizes
            batch_size=batch_size_train//world_size,
            shuffle=False, 
            num_workers=num_workers,
            generator=generator,
            worker_init_fn=seed_worker,
            sampler=torch.utils.data.DistributedSampler(trainset,drop_last=True),drop_last=True)
    else:
        trainloader = torch.utils.data.DataLoader(
                trainset, 
                #batch_size=batch_size_train if lib == 'opacus' else physical_batch_size,  #If it is opacus, it uses the normal batch size, because is the BatchMemoryManager the one that handles the phy and bs sizes
                batch_size=batch_size_train,
                shuffle=True, 
                num_workers=num_workers,
                generator=generator,
                worker_init_fn=seed_worker)
    
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=100, 
        shuffle=False, 
        num_workers=num_workers,
        generator=generator,
        worker_init_fn=seed_worker)

    return trainloader,testloader

#FastDP and private_vision libraries use a similar privacy engine. It will modify the internal methods for 
#training, like step and backward. 
#The privacy engine is returned, but it is actually never used, as the optimizer is attached to it.
#In the case of non private baseline, null is returned
def get_privacy_engine(model,loader,optimizer,lib,sample_rate,expected_batch_size,args):

    sigma = get_noise_multiplier(
            target_epsilon = args.epsilon,
            target_delta = args.target_delta,
            sample_rate = sample_rate,
            epochs = args.epochs,
            accountant = args.accountant
        )
    
    print('Noise multiplier', sigma,flush=True)

    if lib == 'fastDP':
        if 'BK' in args.clipping_mode:
            clipping_mode=args.clipping_mode[3:]
        else:
            clipping_mode='ghost'
        privacy_engine = PrivacyEngineBK(
            model,
            batch_size=expected_batch_size,
            sample_size=len(loader.dataset),
            noise_multiplier=sigma,
            epochs=args.epochs,
            clipping_mode=clipping_mode,
            origin_params=args.origin_params,
            accounting_mode=args.accountant
        )
        privacy_engine.attach(optimizer)
        return privacy_engine
    
    elif lib == 'private_vision':
        if 'ghost' in args.clipping_mode:

            privacy_engine = PrivacyEngineVision(
                model,
                batch_size=expected_batch_size,
                sample_size=len(loader.dataset),
                noise_multiplier=sigma,
                epochs=args.epochs,
                max_grad_norm=args.grad_norm,
                ghost_clipping='non' not in args.clipping_mode,
                mixed='mixed' in args.clipping_mode
            )
            privacy_engine.attach(optimizer)
            return privacy_engine
    
    return None

def get_privacy_engine_opacus(model,loader,optimizer,g,args):
    print('Opacus Engine')
    privacy_engine = PrivacyEngineOpacus(accountant=args.accountant)
    
    model, optimizer, loader = privacy_engine.make_private_with_epsilon(
        module = model,
        optimizer=optimizer,
        data_loader=loader,
        epochs=args.epochs,
        target_epsilon=args.epsilon,
        target_delta=args.target_delta,
        max_grad_norm=args.grad_norm,
        noise_generator=g
    )
    
    print('optimizer params',
    'noise multiplier',optimizer.noise_multiplier,
    'max grad norm',optimizer.max_grad_norm,
    'loss reduction',optimizer.loss_reduction,
    'expected batch size',optimizer.expected_batch_size,flush=True)
    

        
    return model,optimizer,loader,privacy_engine

def get_loss_function(lib):
    if lib == 'private_vision':
        criterion = nn.CrossEntropyLoss(reduction='none')
    else:
        criterion = nn.CrossEntropyLoss()
    return criterion

def privatize_dataloader(data_loader):
    return DPDataLoader.from_data_loader(data_loader,distributed=True)
    
def prepare_vision_model(model,model_name):

    pre_total, pre_train = count_params(model)

    print("Preparing vision model pre total parameters {} pre trained parameters {}".format(pre_total,pre_train))
        
    if 'xcit' in model_name:
      for name,param in model.named_parameters():
          if 'gamma' in name or 'attn.temperature' in name:
            param.requires_grad=False
            
    if 'cait' in model_name:
      for name,param in model.named_parameters():
          if 'gamma_' in name:
            param.requires_grad=False

    if 'convnext' in model_name:
        for name,param in model.named_parameters():
            if '.gamma' in name or 'head.norm.' in name or 'downsample.0' in name or 'stem.1' in name:
                param.requires_grad=False

    if 'convit' in model_name:
        for name,param in model.named_parameters():
            if 'attn.gating_param' in name:
                param.requires_grad=False
                
    if 'beit' in model_name:
        for name,param in model.named_parameters():
            if 'gamma_' in name or 'relative_position_bias_table' in name or 'attn.qkv.weight' in name or 'attn.q_bias' in name or 'attn.v_bias' in name:
                param.requires_grad=False


    for name,param in model.named_parameters():
        if 'cls_token' in name or 'pos_embed' in name:
            param.requires_grad=False

    pos_total, pos_train = count_params(model)
    print("Preparing vision model post total parameters {} post trained parameters {}".format(pos_total,pos_train))
    return model

def count_params(model):
    n_params = sum([p.numel() for p in model.parameters()])
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n_params,trainable_params

def type_params(model):
    for name,params in model.named_parameters():
        print(name,type(params))

def prepare_vit_model(model):
    for name,param in model.named_parameters():
          if 'attn.' in name:
            param.requires_grad = False

#Load model from timm
def load_model(model_name,n_classes,lib):
    print('Path',os.getcwd())
    print('==> Building model..', model_name, 'with n_classes',n_classes)
    model = None
    # Model
    if 'vit_base_patch16_224' in model_name:
        #model = timm.create_model('vit_base_patch16_224.augreg_in21k_ft_in1k',pretrained=True,num_classes=int(n_classes))
        model = timm.create_model(model_name,pretrained=True,num_classes=int(n_classes))
        pre_total, pre_train = count_params(model)
        print("pre total parameters {} pre trained parameters {}".format(pre_total,pre_train))
        #model = ModuleValidator.fix(model)
        pos_total, pos_train = count_params(model)
        print("post total parameters {} post trained parameters {}".format(pos_total,pos_train))
    elif 'BiT-M-R' in model_name:
        std = False
        if lib == 'non' or lib == 'opacus':
            std = True
        model = models.KNOWN_MODELS[model_name](head_size=100, zero_head=True,std = std)
        model.load_from(np.load(f"/models_files/{model_name}.npz"))
        pos_total, pos_train = count_params(model)
        print("post total parameters {} post trained parameters {}".format(pos_total,pos_train))
    else:
        model = timm.create_model(model_name,pretrained=True,num_classes=int(n_classes))
        pre_total, pre_train = count_params(model)
        print("pre total parameters {} pre trained parameters {}".format(pre_total,pre_train))
        print(ModuleValidator.validate(model))
        if not ModuleValidator.is_valid(model) and not lib == 'non':
            model = ModuleValidator.fix(model)
        model = ModuleValidator.fix(model)
        print('After validation: \n',ModuleValidator.validate(model))
        
        pos_total, pos_train = count_params(model)
        print("post total parameters {} post trained parameters {}".format(pos_total,pos_train))
    
    model = models.DpFslLinear(model_name,model,n_classes)

    return model

#Train step. 
#   device. For cuda training
#   model. The current instance of the model
#   lib. Library that is being used. It can be fastDP, private_vision, opacus or non
#   loader. Train loader
#   optimizer. Optimizer ex. Adam
#   criterion. Loss function, in this case is CrossEntropyLoss
#   epoch. Index of the current epoch
#   n_acc_steps 
def train(device,model,lib,loader,optimizer,criterion,epoch,physical_batch):

    flag = EndingLogicalBatchSignal()
    print('training {} model with load size {}'.format(lib,len(loader)))
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    batch_idx = 0
    total_time_epoch = 0
    total_time = 0
    correct_batch = 0
    total_batch = 0
    samples_used = 0
    loss = None
    small_flag = True
    print('Epoch',epoch,'physical batch size',physical_batch,flush=True)
    with MyBatchMemoryManager(
        data_loader=loader, 
        max_physical_batch_size=physical_batch, 
        signaler=flag
    ) as memory_safe_data_loader:
        for batch_idx, (inputs, targets) in enumerate(memory_safe_data_loader):
            if small_flag:
                print('Inputs type',inputs.dtype,flush=True)
                small_flag = False
            starter_t, ender_t = torch.cuda.Event(enable_timing=True),   torch.cuda.Event(enable_timing=True)
            starter_t.record()
            size_b = len(inputs)
            #print('Batch size of ',size_b)
            samples_used += size_b
            inputs, targets = inputs.to(device), targets.type(torch.LongTensor).to(device)

            #Measure time, after loading data to the GPU
            starter, ender = torch.cuda.Event(enable_timing=True),   torch.cuda.Event(enable_timing=True)
            starter.record()  # type: ignore
            torch.set_grad_enabled(True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if lib == 'private_vision':
                if flag._check_skip_next_step():
                    optimizer.virtual_step(loss=loss)
                else:
                    print('take step batch idx: ',batch_idx+1,flush=True)
                    optimizer.step(loss=loss)
                    optimizer.zero_grad()
            else:
                loss.backward()
                if not flag._check_skip_next_step():
                    print('take step batch idx: ',batch_idx+1,flush=True)
                    optimizer.step()
                    optimizer.zero_grad()

            ender.record() #type: ignore
            torch.cuda.synchronize()

            curr_time = starter.elapsed_time(ender)/1000
            total_time_epoch += curr_time
            if lib  == 'private_vision':
                train_loss += loss.mean().item()
            else:
                train_loss += loss.item()
            _, predicted = outputs.max(1)
            del outputs,inputs
            total_batch += targets.size(0)
            correct_batch += predicted.eq(targets).sum().item()
            if  (batch_idx + 1) % 100 == 0 or ((batch_idx + 1) == len(memory_safe_data_loader)):
                print('samples_used',samples_used,'batch_idx',batch_idx,flush=True)
                print('Epoch: ', epoch, 'Batch: ',batch_idx,'total_batch',total_batch,flush=True)
                print('Epoch: ', epoch, 'Batch: ',batch_idx, 'Train Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (train_loss/(batch_idx+1), 100.*correct_batch/total_batch, correct_batch, total_batch),flush=True)
                total += total_batch
                correct += correct_batch
                total_batch = 0
                correct_batch = 0
                
        ender_t.record() #type: ignore
        torch.cuda.synchronize()
        curr_t = starter_t.elapsed_time(ender_t)/1000
        total_time += curr_t  
    del loss
    print('Epoch: ', epoch, len(loader), 'Train Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total),flush=True)
    print('batch_idx',batch_idx,'samples used',samples_used,'samples used / batch_idx',samples_used/batch_idx,'physical batch size',physical_batch,flush=True)
    throughput = (samples_used)/total_time_epoch
    throughput_complete = (samples_used)/total_time
    print('Epoch {} Total time computing {} Throughput computing {}'.format(epoch,total_time_epoch,throughput),flush=True)
    print('Epoch {} Total time {} Throughput {}'.format(epoch,total_time,throughput_complete),flush=True)
    return throughput,throughput_complete


def train_non_private_2(device,model,lib,loader,optimizer,criterion,epoch,physical_batch):

    flag = EndingLogicalBatchSignal()
    print('training {} model with load size {}'.format(lib,len(loader)))
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    batch_idx = 0
    total_time_epoch = 0
    total_time = 0
    correct_batch = 0
    total_batch = 0
    samples_used = 0
    loss = None
    times_up = 0
    print('Epoch',epoch,'physical batch size',physical_batch,'batch size',len(loader.dataset),flush=True)
    with MyBatchMemoryManager(
        data_loader=loader, 
        max_physical_batch_size=physical_batch, 
        signaler=flag
    ) as memory_safe_data_loader:
        for batch_idx, (inputs, targets) in enumerate(memory_safe_data_loader):
            starter_t, ender_t = torch.cuda.Event(enable_timing=True),   torch.cuda.Event(enable_timing=True)
            starter_t.record()
            size_b = len(inputs)
            #print('Batch size of ',size_b)
            samples_used += size_b
            inputs, targets = inputs.to(device), targets.type(torch.LongTensor).to(device)
            with torch.set_grad_enabled(True):


                #Measure time, after loading data to the GPU
                starter, ender = torch.cuda.Event(enable_timing=True),   torch.cuda.Event(enable_timing=True)
                starter.record()  # type: ignore
                torch.set_grad_enabled(True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                loss.backward()
                if not flag._check_skip_next_step():
                    print('take step batch idx: ',batch_idx+1,flush=True)
                    optimizer.step()
                    optimizer.zero_grad()
                    times_up += 1

                ender.record() #type: ignore
                torch.cuda.synchronize()

                curr_time = starter.elapsed_time(ender)/1000
                total_time_epoch += curr_time
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                del outputs,inputs
                total_batch += targets.size(0)
                correct_batch += predicted.eq(targets).sum().item()
                if  (batch_idx + 1) % 100 == 0 or ((batch_idx + 1) == len(memory_safe_data_loader)):
                    print('samples_used',samples_used,'batch_idx',batch_idx,flush=True)
                    print('Epoch: ', epoch, 'Batch: ',batch_idx,'total_batch',total_batch,flush=True)
                    print('Epoch: ', epoch, 'Batch: ',batch_idx, 'Train Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                % (train_loss/(batch_idx+1), 100.*correct_batch/total_batch, correct_batch, total_batch),flush=True)
                    total += total_batch
                    correct += correct_batch
                    total_batch = 0
                    correct_batch = 0
                
        ender_t.record() #type: ignore
        torch.cuda.synchronize()
        curr_t = starter_t.elapsed_time(ender_t)/1000
        total_time += curr_t  
    #del loss
    print('Epoch: ', epoch, len(loader), 'Train Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total),flush=True)
    print('times updated',times_up,flush=True)
    print('batch_idx',batch_idx,'samples used',samples_used,'samples used / batch_idx',samples_used/batch_idx,'physical batch size',physical_batch,flush=True)
    throughput = (samples_used)/total_time_epoch
    throughput_complete = (samples_used)/total_time
    print('Epoch {} Total time computing {} Throughput computing {}'.format(epoch,total_time_epoch,throughput),flush=True)
    print('Epoch {} Total time {} Throughput {}'.format(epoch,total_time,throughput_complete),flush=True)
    return throughput,throughput_complete

#Method for Non private learning.
#It still uses the gradient accumulation, just to compare it to the other methods.
def train_non_private(device,model,loader,optimizer,criterion,epoch,physical_batch,n_acc_steps):
    print('training {} model with load size {}'.format('non-private',len(loader)))
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    batch_idx = 0
    total_time_epoch = 0
    total_time = 0
    correct_batch = 0
    total_batch = 0
    samples_used = 0
    optimizer.zero_grad()
    loss = None

    for batch_idx, (inputs, targets) in enumerate(loader):
        starter_t, ender_t = torch.cuda.Event(enable_timing=True),   torch.cuda.Event(enable_timing=True)
        starter_t.record()
        size_b = len(inputs)
        #batch_sizes.append(size_b)
        samples_used += size_b
        inputs, targets = inputs.to(device), targets.type(torch.LongTensor).to(device)
        #with collector(tag='batch'):
        
        #Measure time, after loading data to the GPU
        starter, ender = torch.cuda.Event(enable_timing=True),   torch.cuda.Event(enable_timing=True)
        starter.record()  # type: ignore

        torch.set_grad_enabled(True)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        if ((batch_idx + 1) % n_acc_steps == 0) or ((batch_idx + 1) == len(loader)):
            print('take step batch idx: ',batch_idx+1,flush=True)
            optimizer.step()
            #train_loss += loss.item() * n_acc_steps * inputs.size(0)
            optimizer.zero_grad()
        _, predicted = outputs.max(1)
        
        
        train_loss += loss.item() 
        total_batch += targets.size(0)
        correct_batch += predicted.eq(targets).sum().item()
        del outputs,inputs

        ender.record() #type: ignore
        torch.cuda.synchronize()

        curr_time = starter.elapsed_time(ender)/1000
        total_time_epoch += curr_time

        if  (batch_idx + 1) % 100 == 0 or ((batch_idx + 1) == len(loader)):
            print('Epoch: ', epoch, 'Batch: ',batch_idx, 'Train Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/len(loader.dataset), 100.*correct_batch/total_batch, correct_batch, total_batch),flush=True)
            total += total_batch
            correct += correct_batch
            total_batch = 0
            correct_batch = 0
        
        ender_t.record() #type: ignore
        torch.cuda.synchronize()
        curr_t = starter_t.elapsed_time(ender_t)/1000
        total_time += curr_t  
    del loss
    print('Epoch: ', epoch, len(loader), 'Train Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total),flush=True)
    print('batch_idx',batch_idx,'samples used',samples_used,'samples used / batch_idx',samples_used/batch_idx,'physical batch size',physical_batch,flush=True)
    throughput = (samples_used)/total_time_epoch
    throughput_complete = (samples_used)/total_time
    print('Epoch {} Total time computing {} Throughput computing {}'.format(epoch,total_time_epoch,throughput),flush=True)
    print('Epoch {} Total time {} Throughput {}'.format(epoch,total_time,throughput_complete),flush=True)
    return throughput,throughput_complete

#Opacus needs its own training method, since it needs the BatchMemoryManager.
def train_opacus(device,model,loader,optimizer,criterion,epoch,physical_batch):
    print('training opacus model')
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    batch_idx = 0
    total_time_epoch = 0
    total_time = 0 
    correct_batch = 0
    total_batch = 0
    samples_used = 0
    loss = None
    print('Epoch',epoch,'physical batch size',physical_batch,flush=True)
    with BatchMemoryManager(
        data_loader=loader, 
        max_physical_batch_size=physical_batch, 
        optimizer=optimizer
    ) as memory_safe_data_loader:
        #len(memory)
        for batch_idx, (inputs, targets) in enumerate(memory_safe_data_loader):
            starter_t, ender_t = torch.cuda.Event(enable_timing=True),   torch.cuda.Event(enable_timing=True)
            starter_t.record()
            #batch_sizes.append(len(inputs))
            samples_used += len(inputs)
            inputs, targets = inputs.to(device), targets.type(torch.LongTensor).to(device)
            #with collector(tag='batch'):
            #Measure time, after loading data to the GPU
            starter, ender = torch.cuda.Event(enable_timing=True),   torch.cuda.Event(enable_timing=True)
            starter.record()  # type: ignore
            optimizer.zero_grad()
            torch.set_grad_enabled(True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()

            optimizer.step()
            #We want to measure just the actual computation, we do not care about the computation of the metrics
            ender.record() #type: ignore
            torch.cuda.synchronize()

            curr_time = starter.elapsed_time(ender)/1000
            total_time_epoch += curr_time
                
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            del outputs,inputs
            total_batch += targets.size(0)
            correct_batch += predicted.eq(targets).sum().item()

            if not optimizer._is_last_step_skipped:
                print('optimizer step skip queue',optimizer._is_last_step_skipped, len(optimizer._step_skip_queue),optimizer._step_skip_queue,'batch idx',batch_idx,flush=True)

            if (batch_idx + 1) % 100 == 0 or ((batch_idx + 1) == len(memory_safe_data_loader)):
                print('Epoch: ', epoch, 'Batch: ',batch_idx, 'Train Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (train_loss/(batch_idx+1), 100.*correct_batch/total_batch, correct_batch, total_batch),flush=True)
                total += total_batch
                correct += correct_batch
                total_batch = 0
                correct_batch = 0
                print('samples_used',samples_used,'batch_idx',batch_idx,flush=True)
                
                
            ender_t.record() #type: ignore
            torch.cuda.synchronize()
            curr_t = starter_t.elapsed_time(ender_t)/1000
            total_time += curr_t  
    del loss
    print('Epoch: ', epoch, len(loader), 'Train Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total),flush=True)
    print('batch_idx',batch_idx,'samples used',samples_used,'samples used / batch_idx',samples_used/batch_idx,'physical batch size',physical_batch,flush=True)
    throughput = (samples_used)/total_time_epoch
    throughput_complete = (samples_used)/total_time
    print('Epoch {} Total time computing {} Throughput computing {}'.format(epoch,total_time_epoch,throughput),flush=True)
    print('Epoch {} Total time {} Throughput {}'.format(epoch,total_time,throughput_complete),flush=True)
    
    return throughput,throughput_complete

#Test
#All algorithms and implementations use this test method. It is very general.
def test(device,model,lib,loader,criterion,epoch):
    model.eval()
    test_loss = 0
    batch_idx = 0
    accs = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if lib  == 'private_vision':
                test_loss += loss.mean().item()
            else:
                test_loss += loss.item()
            _, preds = outputs.max(1)
            acc = preds.eq(targets).sum().item()/targets.size(0)
            accs.append(acc)
            del inputs,targets,outputs

    acc = np.mean(accs)

    dict_test = {'Test Loss':test_loss/(batch_idx+1),'Accuracy': acc}
    print('Epoch: ', epoch, len(loader), 'Test Loss: %.3f | Acc: %.3f%% '
                        % (dict_test['Test Loss'], dict_test['Accuracy']))
    
    return acc

def ddp_setup(rank, world_size,port):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    #os.environ["MASTER_PORT"] = "12355"
    os.environ["MASTER_PORT"] = port
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def main(local_rank,rank, world_size, args):
    
    print(args)
    models_dict = {'fastDP':['BK-ghost', 'BK-MixGhostClip', 'BK-MixOpt'],'private_vision':['PV-ghost','PV-ghost_mixed'],'opacus':['O-flat','O-adaptive','O-per_layer'],'non':['non-private']} # Map from model to library
    
    lib = None

    if args.tf32 == 'True':
        torch.backends.cuda.matmul.allow_tf32=True
        torch.backends.cudnn.allow_tf32=True

    for key,val in models_dict.items():
        if args.clipping_mode in val:
            lib = key

    print('run for the lib {} and model {}'.format(lib,args.clipping_mode))
    timestamp = datetime.now().strftime('%Y%m%d')
    #writer = SummaryWriter('./runs/{}_cifar_{}_{}_model_{}_e_{}_{}'.format(args.test,args.model,args.ten,args.clipping_mode,args.epsilon,timestamp),flush_secs=30)
    #collector = None
    print('Model from',timestamp)
    
    device = local_rank

    generator_gpu,g_cpu = set_seeds(args.seed,device)

    train_loader,test_loader = load_data_cifar(args.ten,args.dimension,args.bs,args.phy_bs,num_workers=args.n_workers,normalization=args.normalization,lib=lib,generator=g_cpu,world_size=world_size)

    expected_batch_size = int(len(train_loader.dataset) * sample_rate)

    n_acc_steps = expected_batch_size // args.phy_bs # gradient accumulation steps

    print('For lib {} with train_loader dataset size {} and train loader size {}'.format(lib,len(train_loader.dataset),len(train_loader)))

    print('Gradient Accumulation Steps',n_acc_steps)

    model_s = load_model(args.model,n_classes=args.ten,lib=lib).to(device)
    if lib == 'non':
        model = DDP(model_s,device_ids=[device])
    else:
        model = DPDDP(model_s)
    
    #If there are layers not supported by the private vision library. In the case of the ViT, it shouldn't freeze anything
    if lib=='private_vision':
        model = prepare_vision_model(model,args.model)

    total_params,trainable_params = count_params(model)

    print("The model has in total {} params, and {} are trainable".format(total_params,trainable_params),flush=True)

    criterion = get_loss_function(lib)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    privacy_engine = None

    #Get the privacy engine depending on the library. In the case of the non private version, the privacy engine will be None
    if lib == 'opacus':
        model, optimizer, train_loader,privacy_engine = get_privacy_engine_opacus(model,train_loader,optimizer,generator_gpu,args)
        print('Opacus model type',type(model))
        print('Opacus optimizer type',type(optimizer))
        print('Opacus loader type',type(train_loader))

    elif lib != 'non':
        train_loader = privatize_dataloader(train_loader) #The BatchMemoryManager of Opacus does this step. Since here we are implementing our own, we have to do this step explicitly before.
        sample_rate = 1 / len(train_loader)
        expected_batch_size = int(len(train_loader.dataset) * sample_rate)
        world_size = torch.distributed.get_world_size()
        expected_batch_size /= world_size
        privacy_engine = get_privacy_engine(model,train_loader,optimizer,lib,sample_rate,expected_batch_size,args)
    
    if args.torch2 == 'True':
        model = torch.compile(model)

    print('memory summary before training: \n',torch.cuda.memory_summary(),flush=True)
    
    test_accs = np.zeros(args.epochs)
    throughs = np.zeros(args.epochs)
    total_thr = np.zeros(args.epochs)
    acc_wt = test(device,model,lib,test_loader,criterion,0)
    print('Without training accuracy',acc_wt)
    for epoch in range(args.epochs):
        print('memory allocated ',torch.cuda.memory_allocated()/1024**2,flush=True)
        if lib == 'opacus':
            th,t_th = train_opacus(device,model,train_loader,optimizer,criterion,epoch,args.phy_bs)
            privacy_results = privacy_engine.get_epsilon(args.target_delta) # type: ignore
            privacy_results = {'eps_rdp':privacy_results}
            print('Privacy results after training {}'.format(privacy_results),flush=True)
        elif lib == 'non':
            #train_loader.sampler.set_epoch(epoch)
            #th,t_th = train_non_private(device,model,train_loader,optimizer,criterion,epoch,args.phy_bs,n_acc_steps)
            th,t_th = train_non_private_2(device,model,lib,train_loader,optimizer,criterion,epoch,args.phy_bs)
        else:
            th,t_th = train(device,model,lib,train_loader,optimizer,criterion,epoch,args.phy_bs)
            privacy_results = privacy_engine.get_privacy_spent() # type: ignore
            print('Privacy results after training {}'.format(privacy_results),flush=True)
        throughs[epoch] = th
        total_thr[epoch] = t_th
        test_accs[epoch] = test(device,model,lib,test_loader,criterion,epoch)
         
        torch.cuda.empty_cache()

    print('--- Finished training ---',flush=True)
    
    thr = np.mean(throughs)
    acc = test_accs[-1]
    t_th = np.mean(total_thr)

    err = None

    row = [args.model,args.clipping_mode,args.epochs,args.phy_bs,err,thr,t_th,acc,args.epsilon]

    path_log = args.file+str(int(rank))+ ".csv"

    exists = os.path.exists(path_log)

    with open(path_log, mode="a") as f:    
        writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)

        if not exists:
            writer.writerow(["model", "clipping_mode","epochs", "physical_batch", "fail",'throughput','total_throughput','acc_test',"epsilon"])

        writer.writerow(row)

    if world_size > 1:  
        torch.distributed.destroy_process_group()