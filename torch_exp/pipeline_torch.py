import os

import torch
import numpy as np



import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn

from opacus.utils.batch_memory_manager import BatchMemoryManager
from MyOwnBatchManager import MyBatchMemoryManager, EndingLogicalBatchSignal
from opacus.data_loader import DPDataLoader
from datetime import datetime

from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from torch.nn.parallel import DistributedDataParallel as DDP

import warnings

warnings.filterwarnings("ignore")

import csv
import time
import gc

from privacy_engines import get_privacy_engine, get_privacy_engine_opacus
from model_functions import count_params, load_model, prepare_vision_model, print_param_shapes
from seeding_utils import set_seeds
from data import load_data_cifar

gc.collect()
torch.cuda.empty_cache()




def get_loss_function(lib):
    if lib == "private_vision":
        criterion = nn.CrossEntropyLoss(reduction="none")
    else:
        criterion = nn.CrossEntropyLoss()
    return criterion


def privatize_dataloader(data_loader, dist):
    return DPDataLoader.from_data_loader(data_loader, distributed=dist)


# Train step.
#   device. For cuda training
#   model. The current instance of the model
#   lib. Library that is being used. It can be fastDP, private_vision, opacus or non
#   loader. Train loader
#   optimizer. Optimizer ex. Adam
#   criterion. Loss function, in this case is CrossEntropyLoss
#   epoch. Index of the current epoch
#   n_acc_steps
def train(device, model, lib, loader, optimizer, criterion, epoch, physical_batch):

    flag = EndingLogicalBatchSignal()
    print("training {} model with load size {}".format(lib, len(loader)))
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
    print("Epoch", epoch, "physical batch size", physical_batch, flush=True)
    with MyBatchMemoryManager(
        data_loader=loader, max_physical_batch_size=physical_batch, signaler=flag
    ) as memory_safe_data_loader:
        for batch_idx, (inputs, targets) in enumerate(memory_safe_data_loader):
            if small_flag:
                print("Inputs type", inputs.dtype, flush=True)
                small_flag = False
            size_b = len(inputs)
            # print('Batch size of ',size_b)
            samples_used += size_b
            starter_t, ender_t = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter_t.record()

            inputs, targets = inputs.to(device), targets.type(torch.LongTensor).to(device)
            torch.set_grad_enabled(True)
            # Measure time, after loading data to the GPU
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()  # type: ignore
            start_time = time.perf_counter()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if lib == "private_vision":
                if flag._check_skip_next_step():
                    optimizer.virtual_step(loss=loss)
                else:
                    print("take step batch idx: ", batch_idx + 1, flush=True)
                    optimizer.step(loss=loss)
                    optimizer.zero_grad()
            else:
                loss.backward()
                if not flag._check_skip_next_step():
                    print("take step batch idx: ", batch_idx + 1, flush=True)
                    optimizer.step()
                    optimizer.zero_grad()

            torch.cuda.synchronize()
            end_time = time.perf_counter()

            total_time_prf = end_time - start_time
            ender.record()  # type: ignore

            curr_time = starter.elapsed_time(ender) / 1000
            # total_time_epoch += curr_time
            total_time_epoch = total_time_prf
            if lib == "private_vision":
                train_loss += loss.mean().item()
            else:
                train_loss += loss.item()
            _, predicted = outputs.max(1)
            del outputs, inputs
            total_batch += targets.size(0)
            correct_batch += predicted.eq(targets).sum().item()
            if (batch_idx + 1) % 100 == 0 or ((batch_idx + 1) == len(memory_safe_data_loader)):
                print("samples_used", samples_used, "batch_idx", batch_idx, flush=True)
                print("Epoch: ", epoch, "Batch: ", batch_idx, "total_batch", total_batch, flush=True)
                print(
                    "Epoch: ",
                    epoch,
                    "Batch: ",
                    batch_idx,
                    "Train Loss: %.3f | Acc: %.3f%% (%d/%d)"
                    % (train_loss / (batch_idx + 1), 100.0 * correct_batch / total_batch, correct_batch, total_batch),
                    flush=True,
                )
                total += total_batch
                correct += correct_batch
                total_batch = 0
                correct_batch = 0
        torch.cuda.synchronize()
        ender_t.record()  # type: ignore

        curr_t = starter_t.elapsed_time(ender_t) / 1000
        total_time += curr_t
    del loss
    print(
        "Epoch: ",
        epoch,
        len(loader),
        "Train Loss: %.3f | Acc: %.3f%% (%d/%d)"
        % (train_loss / (batch_idx + 1), 100.0 * correct / total, correct, total),
        flush=True,
    )
    print(
        "batch_idx",
        batch_idx,
        "samples used",
        samples_used,
        "samples used / batch_idx",
        samples_used / batch_idx,
        "physical batch size",
        physical_batch,
        flush=True,
    )
    throughput = (samples_used) / total_time_epoch
    throughput_complete = (samples_used) / total_time
    print(
        "Epoch {} Total time computing {} Throughput computing {}".format(epoch, total_time_epoch, throughput),
        flush=True,
    )
    print("Epoch {} Total time {} Throughput {}".format(epoch, total_time, throughput_complete), flush=True)
    return throughput, throughput_complete


def train_non_private(device, model, lib, loader, optimizer, criterion, epoch, physical_batch, expected_acc_steps):

    flag = EndingLogicalBatchSignal()
    print("training {} model with load size {}".format(lib, len(loader)))
    model.train()
    train_loss = 0
    batch_loss = 0
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
    acc = 0
    actual_batch_size = 0
    print("Epoch", epoch, "physical batch size", physical_batch, "expected_acc", expected_acc_steps, flush=True)
    with MyBatchMemoryManager(
        data_loader=loader, max_physical_batch_size=physical_batch, signaler=flag
    ) as memory_safe_data_loader:
        for batch_idx, (inputs, targets) in enumerate(memory_safe_data_loader):
            starter_t, ender_t = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter_t.record()
            size_b = len(inputs)
            actual_batch_size += len(inputs)
            # print('Batch size of ',size_b)
            samples_used += size_b
            inputs, targets = inputs.to(device), targets.type(torch.LongTensor).to(device)
            torch.set_grad_enabled(True)

            # Measure time, after loading data to the GPU
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()  # type: ignore
            start_time = time.perf_counter()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if not flag._check_skip_next_step():
                loss = loss / actual_batch_size
                loss.backward()
                acc += 1
                print("take step batch idx: ", batch_idx + 1, flush=True)
                optimizer.step()
                optimizer.zero_grad()
                times_up += 1
                actual_batch_size = 0
            else:
                loss.backward()
                acc += 1

            ender.record()  # type: ignore
            torch.cuda.synchronize()
            end_time = time.perf_counter()

            total_time_perf = end_time - start_time

            curr_time = starter.elapsed_time(ender) / 1000
            # total_time_epoch += curr_time
            total_time_epoch += total_time_perf
            batch_loss += loss.item()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            del outputs, inputs
            total_batch += targets.size(0)
            correct_batch += predicted.eq(targets).sum().item()
            if (batch_idx + 1) % 100 == 0 or ((batch_idx + 1) == len(memory_safe_data_loader)):
                print("samples_used", samples_used, "batch_idx", batch_idx, flush=True)
                print("Epoch: ", epoch, "Batch: ", batch_idx, "total_batch", total_batch, flush=True)
                print(
                    "Epoch: ",
                    epoch,
                    "Batch: ",
                    batch_idx,
                    "Train Loss: %.3f | Acc: %.3f%% (%d/%d)"
                    % (batch_loss / (acc), 100.0 * correct_batch / total_batch, correct_batch, total_batch),
                    flush=True,
                )
                total += total_batch
                correct += correct_batch
                total_batch = 0
                correct_batch = 0
                batch_loss = 0
                acc = 0

        ender_t.record()  # type: ignore
        torch.cuda.synchronize()
        curr_t = starter_t.elapsed_time(ender_t) / 1000
        total_time += curr_t
    del loss
    print(
        "Epoch: ",
        epoch,
        len(loader),
        "Train Loss: %.3f | Acc: %.3f%% (%d/%d)" % (train_loss / len(loader), 100.0 * correct / total, correct, total),
        flush=True,
    )
    print("times updated", times_up, flush=True)
    print(
        "batch_idx",
        batch_idx,
        "samples used",
        samples_used,
        "samples used / batch_idx",
        samples_used / batch_idx,
        "physical batch size",
        physical_batch,
        flush=True,
    )
    throughput = (samples_used) / total_time_epoch
    throughput_complete = (samples_used) / total_time
    print(
        "Epoch {} Total time computing {} Throughput computing {}".format(epoch, total_time_epoch, throughput),
        flush=True,
    )
    print("Epoch {} Total time {} Throughput {}".format(epoch, total_time, throughput_complete), flush=True)
    return throughput, throughput_complete


# Opacus needs its own training method, since it needs the BatchMemoryManager.
def train_opacus(device, model, loader, optimizer, criterion, epoch, physical_batch):
    print("training opacus model")
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
    print("Epoch", epoch, "physical batch size", physical_batch, flush=True)
    with BatchMemoryManager(
        data_loader=loader, max_physical_batch_size=physical_batch, optimizer=optimizer
    ) as memory_safe_data_loader:
        # len(memory)
        for batch_idx, (inputs, targets) in enumerate(memory_safe_data_loader):
            starter_t, ender_t = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter_t.record()
            # batch_sizes.append(len(inputs))
            samples_used += len(inputs)
            inputs, targets = inputs.to(device), targets.type(torch.LongTensor).to(device)
            # with collector(tag='batch'):
            # Measure time, after loading data to the GPU
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()  # type: ignore
            start_time = time.perf_counter()
            optimizer.zero_grad()
            torch.set_grad_enabled(True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()

            optimizer.step()
            # We want to measure just the actual computation, we do not care about the computation of the metrics
            ender.record()  # type: ignore
            torch.cuda.synchronize()
            end_time = time.perf_counter()

            total_time_perf = end_time - start_time

            curr_time = starter.elapsed_time(ender) / 1000
            # total_time_epoch += curr_time
            total_time_epoch += total_time_perf

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            del outputs, inputs
            total_batch += targets.size(0)
            correct_batch += predicted.eq(targets).sum().item()

            if not optimizer._is_last_step_skipped:
                print(
                    "optimizer step skip queue",
                    optimizer._is_last_step_skipped,
                    len(optimizer._step_skip_queue),
                    optimizer._step_skip_queue,
                    "batch idx",
                    batch_idx,
                    flush=True,
                )

            if (batch_idx + 1) % 100 == 0 or ((batch_idx + 1) == len(memory_safe_data_loader)):
                print(
                    "Epoch: ",
                    epoch,
                    "Batch: ",
                    batch_idx,
                    "Train Loss: %.3f | Acc: %.3f%% (%d/%d)"
                    % (train_loss / (batch_idx + 1), 100.0 * correct_batch / total_batch, correct_batch, total_batch),
                    flush=True,
                )
                total += total_batch
                correct += correct_batch
                total_batch = 0
                correct_batch = 0
                print("samples_used", samples_used, "batch_idx", batch_idx, flush=True)

            ender_t.record()  # type: ignore
            torch.cuda.synchronize()
            curr_t = starter_t.elapsed_time(ender_t) / 1000
            total_time += curr_t
    del loss
    print(
        "Epoch: ",
        epoch,
        len(loader),
        "Train Loss: %.3f | Acc: %.3f%% (%d/%d)"
        % (train_loss / (batch_idx + 1), 100.0 * correct / total, correct, total),
        flush=True,
    )
    print(
        "batch_idx",
        batch_idx,
        "samples used",
        samples_used,
        "samples used / batch_idx",
        samples_used / batch_idx,
        "physical batch size",
        physical_batch,
        flush=True,
    )
    throughput = (samples_used) / total_time_epoch
    throughput_complete = (samples_used) / total_time
    print(
        "Epoch {} Total time computing {} Throughput computing {}".format(epoch, total_time_epoch, throughput),
        flush=True,
    )
    print("Epoch {} Total time {} Throughput {}".format(epoch, total_time, throughput_complete), flush=True)

    return throughput, throughput_complete


# Test
# All algorithms and implementations use this test method. It is very general.
def test(device, model, lib, loader, criterion, epoch):
    model.eval()
    test_loss = 0
    batch_idx = 0
    correct_test = 0
    total_test = 0
    accs = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if lib == "private_vision":
                test_loss += loss.mean().item()
            else:
                test_loss += loss.item()
            _, preds = outputs.max(1)
            correct_test += preds.eq(targets).sum().item()
            total_test += targets.size(0)
            acc = preds.eq(targets).sum().item() / targets.size(0)
            accs.append(acc)
            del inputs, targets, outputs

    acc = np.mean(accs)

    dict_test = {"Test Loss": test_loss / len(loader), "Accuracy": acc}
    print(
        "Epoch: ",
        epoch,
        len(loader),
        "Test Loss: %.3f | Acc: %.3f " % (dict_test["Test Loss"], dict_test["Accuracy"]),
        flush=True,
    )

    print("correctly classified", correct_test, "/", total_test, 100.0 * correct_test / total_test, flush=True)

    return acc


def main(local_rank, rank, world_size, args):

    print(args)
    models_dict = {
        "fastDP": ["BK-ghost", "BK-MixGhostClip", "BK-MixOpt"],
        "private_vision": ["PV-ghost", "PV-ghost_mixed"],
        "opacus": ["O-flat", "O-adaptive", "O-per_layer", "O-ghost"],
        "non": ["non-private"],
    }  # Map from model to library

    lib = None

    if args.tf32 == "True":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    for key, val in models_dict.items():
        if args.clipping_mode in val:
            lib = key

    dist = True

    print("run for the lib {} and model {}".format(lib, args.clipping_mode))
    timestamp = datetime.now().strftime("%Y%m%d")
    # writer = SummaryWriter('./runs/{}_cifar_{}_{}_model_{}_e_{}_{}'.format(args.test,args.model,args.ten,args.clipping_mode,args.epsilon,timestamp),flush_secs=30)
    # collector = None
    print("Model from", timestamp)

    device = local_rank

    generator_gpu, g_cpu = set_seeds(args.seed, device)

    train_loader, test_loader = load_data_cifar(
        args.dimension,
        args.bs,
        args.phy_bs,
        num_workers=args.n_workers,
        normalization=args.normalization,
        lib=lib,
        generator=g_cpu,
        world_size=world_size,
    )

    print(
        "For lib {} with train_loader dataset size {} and train loader size {} and world size {}".format(
            lib, len(train_loader.dataset), len(train_loader), world_size
        )
    )

    model_s = load_model(args.model, n_classes=100, lib=lib).to(device)
    print("device", device, "world size", world_size, "rank", rank)
    if lib == "non":
        model = DDP(model_s, device_ids=[device])
    else:
        model = DPDDP(model_s)

    # If there are layers not supported by the private vision library. In the case of the ViT, it shouldn't freeze anything
    if lib == "private_vision":
        model = prepare_vision_model(model, args.model)

    total_params, trainable_params = count_params(model)

    print("The model has in total {} params, and {} are trainable".format(total_params, trainable_params), flush=True)
    print_param_shapes(model)

    criterion = get_loss_function(lib)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    privacy_engine = None

    # Get the privacy engine depending on the library. In the case of the non private version, the privacy engine will be None
    if lib == "opacus":
        criterion_opacus = get_loss_function(lib)
        model, optimizer, train_loader, privacy_engine, criterion_opacus = get_privacy_engine_opacus(
            model, train_loader, optimizer, criterion_opacus, generator_gpu, args
        )
        print("Opacus model type", type(model))
        print("Opacus optimizer type", type(optimizer))
        print("Opacus loader type", type(train_loader))
    elif lib != "non":
        train_loader = privatize_dataloader(
            train_loader, dist
        )  # The BatchMemoryManager of Opacus does this step. Since here we are implementing our own, we have to do this step explicitly before.
        sample_rate = 1 / len(train_loader)
        expected_batch_size = int(len(train_loader.dataset) * sample_rate)
        world_size = torch.distributed.get_world_size()
        expected_batch_size /= world_size
        privacy_engine = get_privacy_engine(
            model, train_loader, optimizer, lib, sample_rate, expected_batch_size, args
        )
    elif lib == "non":
        train_loader = privatize_dataloader(
            train_loader, dist
        )  # In this case is only to be consistent with the sampling
        sample_rate = 1 / len(train_loader)

        expected_batch_size = int(len(train_loader.dataset) * sample_rate)

        n_acc_steps = expected_batch_size // args.phy_bs  # gradient accumulation steps

        print("Gradient Accumulation Steps", n_acc_steps)

    if args.torch2 == "True":
        model = torch.compile(model)

    print("memory summary before training: \n", torch.cuda.memory_summary(), flush=True)

    test_accs = np.zeros(args.epochs)
    throughs = np.zeros(args.epochs)
    total_thr = np.zeros(args.epochs)
    acc_wt = test(device, model, lib, test_loader, criterion, 0)
    print("Without training accuracy", acc_wt)
    for epoch in range(args.epochs):
        print("memory allocated ", torch.cuda.memory_allocated() / 1024**2, flush=True)
        if lib == "opacus":
            th, t_th = train_opacus(device, model, train_loader, optimizer, criterion_opacus, epoch, args.phy_bs)
            privacy_results = privacy_engine.get_epsilon(args.target_delta)  # type: ignore
            privacy_results = {"eps_rdp": privacy_results}
            print("Privacy results after training {}".format(privacy_results), flush=True)
        elif lib == "non":
            # train_loader.sampler.set_epoch(epoch)
            # th,t_th = train_non_private(device,model,train_loader,optimizer,criterion,epoch,args.phy_bs,n_acc_steps)
            th, t_th = train_non_private(
                device, model, lib, train_loader, optimizer, criterion, epoch, args.phy_bs, n_acc_steps
            )
        else:
            th, t_th = train(device, model, lib, train_loader, optimizer, criterion, epoch, args.phy_bs)
            privacy_results = privacy_engine.get_privacy_spent()  # type: ignore
            print("Privacy results after training {}".format(privacy_results), flush=True)
        throughs[epoch] = th
        total_thr[epoch] = t_th
        test_accs[epoch] = test(device, model, lib, test_loader, criterion, epoch)

        torch.cuda.empty_cache()

    print("--- Finished training ---", flush=True)
    acc = test(device, model, lib, test_loader, criterion, epoch)
    thr = np.mean(throughs)
    # acc = test_accs[-1]
    t_th = np.mean(total_thr)

    err = None

    row = [
        args.model,
        args.clipping_mode,
        args.normalization,
        args.epochs,
        args.phy_bs,
        err,
        thr,
        t_th,
        acc,
        args.epsilon,
    ]

    path_log = args.file + str(int(rank)) + ".csv"

    exists = os.path.exists(path_log)

    with open(path_log, mode="a") as f:
        writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)

        if not exists:
            writer.writerow(
                [
                    "model",
                    "clipping_mode",
                    "normalization",
                    "epochs",
                    "physical_batch",
                    "fail",
                    "throughput",
                    "total_throughput",
                    "acc_test",
                    "epsilon",
                ]
            )

        writer.writerow(row)

    if world_size > 1:
        torch.distributed.destroy_process_group()


def main_non_distributed(args):

    print(args)
    models_dict = {
        "fastDP": ["BK-ghost", "BK-MixGhostClip", "BK-MixOpt"],
        "private_vision": ["PV-ghost", "PV-ghost_mixed"],
        "opacus": ["O-flat", "O-adaptive", "O-per_layer", "O-ghost"],
        "non": ["non-private"],
    }  # Map from model to library

    lib = None

    if args.tf32 == "True":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    for key, val in models_dict.items():
        if args.clipping_mode in val:
            lib = key

    dist = False

    print("run for the lib {} and model {}".format(lib, args.clipping_mode))
    timestamp = datetime.now().strftime("%Y%m%d")
    # writer = SummaryWriter('./runs/{}_cifar_{}_{}_model_{}_e_{}_{}'.format(args.test,args.model,args.ten,args.clipping_mode,args.epsilon,timestamp),flush_secs=30)
    # collector = None
    print("Model from", timestamp)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator_gpu, g_cpu = set_seeds(args.seed, device)

    train_loader, test_loader = load_data_cifar(
        args.dimension,
        args.bs,
        args.phy_bs,
        num_workers=args.n_workers,
        normalization=args.normalization,
        lib=lib,
        generator=g_cpu,
        world_size=1,
    )

    print(
        "For lib {} with train_loader dataset size {} and train loader size {} and world size {}".format(
            lib, len(train_loader.dataset), len(train_loader), 1
        )
    )

    model = load_model(args.model, n_classes=100, lib=lib).to(device)
    print("device", device)

    # If there are layers not supported by the private vision library. In the case of the ViT, it shouldn't freeze anything
    if lib == "private_vision":
        model = prepare_vision_model(model, args.model)

    total_params, trainable_params = count_params(model)

    print("The model has in total {} params, and {} are trainable".format(total_params, trainable_params), flush=True)
    print_param_shapes(model)

    criterion = get_loss_function(lib)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    privacy_engine = None

    # Get the privacy engine depending on the library. In the case of the non private version, the privacy engine will be None
    if lib == "opacus":
        model, optimizer, train_loader, privacy_engine, criterion = get_privacy_engine_opacus(
            model, train_loader, optimizer, criterion, generator_gpu, args
        )
        print("Opacus model type", type(model))
        print("Opacus optimizer type", type(optimizer))
        print("Opacus loader type", type(train_loader))
    elif lib != "non":
        train_loader = privatize_dataloader(
            train_loader, dist
        )  # The BatchMemoryManager of Opacus does this step. Since here we are implementing our own, we have to do this step explicitly before.
        sample_rate = 1 / len(train_loader)
        expected_batch_size = int(len(train_loader.dataset) * sample_rate)
        world_size = 1
        expected_batch_size /= world_size
        privacy_engine = get_privacy_engine(
            model, train_loader, optimizer, lib, sample_rate, expected_batch_size, args
        )
    elif lib == "non":
        train_loader = privatize_dataloader(
            train_loader, dist
        )  # In this case is only to be consistent with the sampling
        sample_rate = 1 / len(train_loader)

        expected_batch_size = int(len(train_loader.dataset) * sample_rate)

        n_acc_steps = expected_batch_size // args.phy_bs  # gradient accumulation steps

        print("Gradient Accumulation Steps", n_acc_steps)

    if args.torch2 == "True":
        model = torch.compile(model)

    print("memory summary before training: \n", torch.cuda.memory_summary(), flush=True)

    test_accs = np.zeros(args.epochs)
    throughs = np.zeros(args.epochs)
    total_thr = np.zeros(args.epochs)
    acc_wt = test(device, model, lib, test_loader, criterion, 0)
    print("Without training accuracy", acc_wt)
    for epoch in range(args.epochs):
        print("memory allocated ", torch.cuda.memory_allocated() / 1024**2, flush=True)
        if lib == "opacus":
            th, t_th = train_opacus(device, model, train_loader, optimizer, criterion, epoch, args.phy_bs)
            privacy_results = privacy_engine.get_epsilon(args.target_delta)  # type: ignore
            privacy_results = {"eps_rdp": privacy_results}
            print("Privacy results after training {}".format(privacy_results), flush=True)
        elif lib == "non":
            # train_loader.sampler.set_epoch(epoch)
            # th,t_th = train_non_private(device,model,train_loader,optimizer,criterion,epoch,args.phy_bs,n_acc_steps)
            th, t_th = train_non_private(
                device, model, lib, train_loader, optimizer, criterion, epoch, args.phy_bs, n_acc_steps
            )
        else:
            th, t_th = train(device, model, lib, train_loader, optimizer, criterion, epoch, args.phy_bs)
            privacy_results = privacy_engine.get_privacy_spent()  # type: ignore
            print("Privacy results after training {}".format(privacy_results), flush=True)
        throughs[epoch] = th
        total_thr[epoch] = t_th
        test_accs[epoch] = test(device, model, lib, test_loader, criterion, epoch)

        torch.cuda.empty_cache()

    print("--- Finished training ---", flush=True)
    acc = test(device, model, lib, test_loader, criterion, epoch)
    thr = np.mean(throughs)
    # acc = test_accs[-1]
    t_th = np.mean(total_thr)

    err = None

    row = [
        args.model,
        args.clipping_mode,
        args.normalization,
        args.epochs,
        args.phy_bs,
        args.lr,
        err,
        thr,
        t_th,
        acc,
        args.epsilon,
    ]

    path_log = args.file + ".csv"

    exists = os.path.exists(path_log)

    with open(path_log, mode="a") as f:
        writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)

        if not exists:
            writer.writerow(
                [
                    "model",
                    "clipping_mode",
                    "normalization",
                    "epochs",
                    "physical_batch",
                    "lr",
                    "fail",
                    "throughput",
                    "total_throughput",
                    "acc_test",
                    "epsilon",
                ]
            )

        writer.writerow(row)
