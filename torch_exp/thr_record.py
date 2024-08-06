import argparse
from pipeline_torch import main
import os
import csv
import numpy as np
import torch
import torch.multiprocessing as mp
import socket
import torch.distributed as dist


def get_free_port():
    s = socket.socket()
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return str(port)


if __name__ == '__main__':

    port = get_free_port()

    path_log = 'thr_record'

    parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
    parser.add_argument('--lr', default=0.0005, type=float, help='learning rate')
    parser.add_argument('--epochs', default=3, type=int,
                        help='numter of epochs')
    parser.add_argument('--bs', default=1000, type=int, help='batch size')
    parser.add_argument('--epsilon', default=2, type=float, help='target epsilon')
    #parser.add_argument('--clipping_mode', default='BK-MixOpt', type=str)
    parser.add_argument('--clipping_mode', default='O-flat', type=str)
    parser.add_argument('--model', default='vit_base_patch16_224.augreg_in21k_ft_in1k', type=str)
    #parser.add_argument('--cifar_data', type=str, default='CIFAR10')
    parser.add_argument('--dimension', type=int,default=224)
    parser.add_argument('--origin_params', nargs='+', default=None)
    parser.add_argument('--ten',default=10,type=int)
    parser.add_argument('--n_workers',default=10,type=int)
    parser.add_argument('--phy_bs',default=50,type=int,help='Physical Batch')
    parser.add_argument('--accountant',default='rdp',type=str)
    parser.add_argument('--grad_norm', '-gn', default=0.1,
                        type=float, help='max grad norm')
    parser.add_argument('--target_delta',default=1e-5,type=float,help='target delta')
    parser.add_argument('--seed',default=1234,type=int)
    parser.add_argument('--normalization',default=True,type=bool)
    parser.add_argument('--test',type=str,default='train')
    parser.add_argument('--file',type=str,default='thr_record')
    parser.add_argument('--tf32',type=str,default='False')
    parser.add_argument('--torch2',type=str,default='False')
    args = parser.parse_args()
    path_log = args.file
    thr = None
    acc = None
    t_th = None
    try:
        world_size = torch.cuda.device_count()
        dist.init_process_group(backend='nccl')
        world_size = dist.get_world_size()
        local_rank = int(os.environ['LOCAL_RANK'])
        rank = int(os.environ['RANK'])
        torch.cuda.set_device(local_rank)
        main(local_rank,rank,world_size,args)
        err = 'None'
    except RuntimeError as e:
        print(e)
        err = 'OOM'