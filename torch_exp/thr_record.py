import argparse
from pipeline_torch import distributed_main, main_non_distributed
import os
import torch
import socket
import torch.distributed as dist


def get_free_port():
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return str(port)


if __name__ == "__main__":

    path_log = "thr_record"

    parser = argparse.ArgumentParser(description="PyTorch CIFAR Training")
    parser.add_argument("--lr", default=0.0005, type=float, help="learning rate")
    parser.add_argument("--epochs", default=3, type=int, help="number of epochs")
    parser.add_argument("--bs", default=1000, type=int, help="batch size")
    parser.add_argument("--epsilon", default=2, type=float, help="target epsilon")
    parser.add_argument("--clipping_mode", default="O-flat", type=str, help="Which clipping algorithm to use.")
    parser.add_argument("--model", default="vit_base_patch16_224.augreg_in21k_ft_in1k", type=str, help="The name of the model (for loading from timm library).")
    parser.add_argument("--dimension", type=int, default=224, help="The size of the cifar100 images.")
    parser.add_argument("--origin_params", nargs="+", default=None)
    parser.add_argument("--n_workers", default=10, type=int, help="The number of workers in the data loader.")
    parser.add_argument("--phy_bs", default=50, type=int, help="Physical Batch Size")
    parser.add_argument("--accountant", default="rdp", type=str, help="The privacy accountant for DP training.")
    parser.add_argument("--grad_norm", "-gn", default=0.1, type=float, help="max grad norm")
    parser.add_argument("--target_delta", default=1e-5, type=float, help="target delta")
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--normalization", default="True", type=str, help="Normalize the data.")
    parser.add_argument("--file", type=str, default="thr_record", help="The file name for saving the results.")
    parser.add_argument("--tf32", type=str, default="False", help="Use TF32 precision.")
    parser.add_argument("--torch2", type=str, default="False", help="Try to use torch2 and compile the model.")
    parser.add_argument("--distributed", type=str, default="True", help="Run the training in a distributed manner over multiple GPUs.")
    args = parser.parse_args()
    path_log = args.file
    thr = None
    acc = None
    t_th = None

    if args.distributed == "True":
        port = get_free_port()
        try:
            world_size = torch.cuda.device_count()
            dist.init_process_group(backend="nccl")
            world_size = dist.get_world_size()
            local_rank = int(os.environ["LOCAL_RANK"])
            rank = int(os.environ["RANK"])
            torch.cuda.set_device(local_rank)
            distributed_main(local_rank, rank, world_size, args)
            err = "None"
        except RuntimeError as e:
            print(e)
            err = "OOM"
    else:
        try:
            main_non_distributed(args)
            err = "None"
        except RuntimeError as e:
            print(e)
            err = "OOM"
