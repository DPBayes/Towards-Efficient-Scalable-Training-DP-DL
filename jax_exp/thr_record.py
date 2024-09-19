import argparse
from pipeline_jax import main
import os
import csv

if __name__ == '__main__':

    path_log = 'thr_record.csv'

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
    #main(args)
    path_log = args.file
    thr = None
    acc = None
    t_th = None
    comp_time = 0
    try:
        thr,t_th,comp_time,acc = main(args)
        err = 'None'
    except RuntimeError as e:
        print(e)
        err = 'OOM'

    row = [args.model,args.clipping_mode,args.normalization,args.epochs,args.phy_bs,args.lr,err,thr,t_th,acc,comp_time,args.epsilon]

    exists = os.path.exists(path_log)

    with open(path_log, mode="a") as f:    
        writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)

        if not exists:
            writer.writerow(["model", "clipping_mode","normalization","epochs", "physical_batch","lr", "fail",'throughput','total_throughput','acc_test','compilation_time','epsilon'])

        writer.writerow(row)