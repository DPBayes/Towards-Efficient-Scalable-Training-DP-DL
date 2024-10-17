# Towards Efficient and Scalable Training of Differentially Private Deep Learning

This repository contains the code for reproducing the experiments in Towards Efficient and Scalable Training of Differentially Private Deep Learning.

## Dependencies

The main dependencies are:

* Python 3.12.5 (any version >=3.11)
* JAX 0.4.30
* transformers 4.45.1
* Opacus 1.5.2
* dp-accounting 0.4.4
* objax 1.8.0

The source code libraries are also necessary. The rest are included in the requirements file.

## Source Code LIbraries

On our work, we use the following open source code libraries, and evaluate their performance:

- timm (For downloading ViT torch models). https://github.com/fastai/timmdocs 
- private_vision (For Ghost Clipping implementations). https://github.com/woodyx218/private_vision
- FastDP (For Ghost Clipping and Book Keeping implementations). Copyright 2024. https://github.com/awslabs/fast-differential-privacy
- Flax (For downloading and instantiate ViT JAX models). Google 2024. https://github.com/google/flax

private_vision and FastDP require building from source, using:

```bash
python -m setup develop
```
After cloning the repositories locally.

## GPU Requirements

The experiments in the paper are executed on NVIDIA V100 and A100 GPUs with 32 and 40 VRAM respectively. 

## Experiments

### PyTorch

```
    python3 Towards-Efficient-Scalable-Training-DP-DL/torch_exp/thr_record.py --n_workers 16 --bs 32492 --phy_bs 32 --seed 11 --epochs 2 --grad_norm 4.63 --epsilon 8 --lr 0.00031 --ten 100 --normalization 'True' --clipping_mode 'O-flat' --file 'record_throughput' --torch2 'False' --tf32 'False' --model 'vit_base_patch16_224.augreg_in21k' --distributed 'False'
```

- Clipping methods:

    ```
    --clipping_mode <'non-private','O-flat','PV-ghost','BK-ghost','BK-MixOpt'>
    ```
    
    According to the clipping mode, a different library and privacy engine will be used. If chosen 'non-private', it will be the PyTorch training loop, with virtual batching. It also accepts 'O-ghost' for Opacus ghost clipping, but at the time of publication, it was not working properly. 

- file:
    The file name can be anything, and it should be without the file type. It will automatically create a csv.
    

There are different flags:

- Distributed experiments
    ```
    --distributed <'False','True'>
    ```

- Compilation
    ```
    --torch2 <'False','True'>
    ```
    Although with torch 2 it is possible to compile the model, we found out that it falls back to NVIDIA kernels, and does not compile. It takes more time trying to compile and then falling back.

- Lower Precision
    ```
    --tf32 <'False','True'>
    ```
    Use this flag for the Lower Precision experiments. It will use TF32, only available for A100 NVIDIA GPUs, not V100.

### JAX

#### Mask Efficient Implementation
```
    python3 Towards-Efficient-Scalable-Training-DP-DL/jax_exp/thr_record_for_mask.py --bs 32492 --phy_bs=64 --file 'record_throughput.csv' --model 'google/vit-base-patch16-224-in21k' --n_workers 0 --epsilon 8 --seed 20 --grad_norm 4.637 --clipping_mode 'DP' --epochs 2 --lr 0.00031
```

- Clipping methods:

    ```
    --clipping_mode <'non-private','DP'>
    ```

    Choose clipping mode between the non-private and DP.

#### Naive Implementation
```
    python3 Towards-Efficient-Scalable-Training-DP-DL/jax_exp/thr_record.py --bs 32492 --phy_bs=64 --file 'record_throughput.csv' --model 'google/vit-base-patch16-224-in21k' --n_workers 16 --epsilon 8 --seed 20 --grad_norm 4.637 --clipping_mode 'DP' --epochs 2 --lr 0.00031
```

- Clipping methods:

    ```
    --clipping_mode <'non-private-virtual','private-mini'>
    ```
    
    Choose clipping mode between the non-private and DP.

## Contact

For comments and issues, create a new issue in the repository.

## Citation

If you use this code, please cite our [paper](https://arxiv.org/abs/2406.17298)

```
@misc{beltran2024efficientscalabletrainingdifferentially,
      title={Towards Efficient and Scalable Training of Differentially Private Deep Learning}, 
      author={Sebastian Rodriguez Beltran and Marlon Tobaben and Niki Loppi and Antti Honkela},
      year={2024},
      eprint={2406.17298},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2406.17298}, 
}
```