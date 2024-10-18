import numpy as np
import jax
import torchvision
import torch.utils.data as data

def import_data_efficient_mask():
    train_images = np.load("numpy_cifar100/train_images.npy")# .to_device(device=jax.devices("cpu")[0]) Use this with JAX > 0.4.40, with 0.4.30 still doesn't exist
    train_labels = np.load("numpy_cifar100/train_labels.npy")# .to_device(device=jax.devices("cpu")[0])

    train_images = jax.device_put(train_images, device=jax.devices("cpu")[0])
    train_labels = jax.device_put(train_labels, device=jax.devices("cpu")[0])

    #Load test data
    
    test_images = np.load("numpy_cifar100/test_images.npy")# .to_device(device=jax.devices("cpu")[0]) 
    test_labels = np.load("numpy_cifar100/test_labels.npy")# .to_device(device=jax.devices("cpu")[0])

    test_images = jax.device_put(test_images, device=jax.devices("cpu")[0])
    test_labels = jax.device_put(test_labels, device=jax.devices("cpu")[0])

    return train_images,train_labels,test_images,test_labels

DATA_MEANS = np.array([0.5, 0.5, 0.5])
DATA_STD = np.array([0.5,0.5, 0.5])

DATA_MEANS2 = (0.485, 0.456, 0.406)
DATA_STD2 =  (0.229, 0.224, 0.225)

def image_to_numpy(img):
    img = np.array(img, dtype=np.float32)
    img = ((img / 255.) - DATA_MEANS) / DATA_STD
    img = np.transpose(img,[2,0,1])
    return img

#Some implementations of training with CIFAR use different values than 0.5
def image_to_numpy_diff(img):
    img = np.array(img, dtype=np.float32)
    img = ((img / 255.) - DATA_MEANS2) / DATA_STD2
    img = np.transpose(img,[2,0,1])
    return img

#Turn data into numpy, instead of torch tensors
def numpy_collate(batch):
    if isinstance(batch[0],np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0],(tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

#Load CIFAR data
def load_data_cifar(dimension,batch_size_train,physical_batch_size,num_workers,generator,norm,seed_worker):

    print('load_data_cifar',batch_size_train,physical_batch_size,num_workers)

    if norm == 'True':
        fn = image_to_numpy
    else:
        fn = image_to_numpy_diff


    transformation = torchvision.transforms.Compose([
        torchvision.transforms.Resize(dimension),
        fn,
    ])
    

    trainset = torchvision.datasets.CIFAR100(root='../data_cifar100/', train=True, download=True, transform=transformation)
    testset = torchvision.datasets.CIFAR100(root='../data_cifar100/', train=False, download=True, transform=transformation)

    trainloader = data.DataLoader(
        trainset, batch_size=batch_size_train, shuffle=True,collate_fn=numpy_collate, num_workers=num_workers,generator=generator,worker_init_fn=seed_worker)

    testloader = data.DataLoader(
        testset, batch_size=80, shuffle=False,collate_fn=numpy_collate, num_workers=num_workers,generator=generator,worker_init_fn=seed_worker)

    return trainloader,testloader