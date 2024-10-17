from torchvision import datasets, transforms
import torch
from seeding_utils import seed_worker

def load_data_cifar(
    ten, dimension, batch_size_train, physical_batch_size, num_workers, normalization, lib, generator, world_size
):

    print("load_data_cifar", lib, batch_size_train, physical_batch_size, num_workers)

    if normalization == "True":
        means = (0.5, 0.5, 0.5)
        stds = (0.5, 0.5, 0.5)
    else:
        means = (0.485, 0.456, 0.406)
        stds = (0.229, 0.224, 0.225)

    transformation = transforms.Compose(
        [
            transforms.Resize(dimension),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ]
    )

    if ten == 10:
        trainset = datasets.CIFAR10(root="../data_cifar10/", train=True, download=True, transform=transformation)
        testset = datasets.CIFAR10(root="../data_cifar10/", train=False, download=True, transform=transformation)
    else:
        trainset = datasets.CIFAR100(root="../data_cifar100/", train=True, download=True, transform=transformation)
        testset = datasets.CIFAR100(root="../data_cifar100/", train=False, download=True, transform=transformation)

    if lib == "non" and world_size > 1:
        trainloader = torch.utils.data.DataLoader(
            trainset,
            # batch_size=batch_size_train if lib == 'opacus' else physical_batch_size,  #If it is opacus, it uses the normal batch size, because is the BatchMemoryManager the one that handles the phy and bs sizes
            batch_size=batch_size_train // world_size,
            shuffle=False,
            num_workers=num_workers,
            generator=generator,
            worker_init_fn=seed_worker,
            sampler=torch.utils.data.DistributedSampler(trainset, drop_last=True),
            drop_last=True,
        )
    else:
        trainloader = torch.utils.data.DataLoader(
            trainset,
            # batch_size=batch_size_train if lib == 'opacus' else physical_batch_size,  #If it is opacus, it uses the normal batch size, because is the BatchMemoryManager the one that handles the phy and bs sizes
            batch_size=batch_size_train,
            shuffle=True,
            num_workers=num_workers,
            generator=generator,
            worker_init_fn=seed_worker,
        )

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=80, shuffle=False, num_workers=num_workers, generator=generator, worker_init_fn=seed_worker
    )

    return trainloader, testloader