import torch
import random
import numpy as np

# Defines each worker seed. Since each worker needs a different seed.
# The worker_id is a parameter given by the loader, but it is not used inside the method
def seed_worker(worker_id):

    # print(torch.initial_seed(),flush=True)

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# Set seeds.
# Returns the generator, that will be used for the data loader
def set_seeds(seed, device):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)

    g_cuda = torch.Generator(device)

    g_cuda.manual_seed(seed)

    g_cpu = torch.Generator("cpu")

    g_cpu.manual_seed(seed)

    np.random.seed(seed)

    print("set seeds seed", seed, flush=True)

    print(torch.initial_seed(), flush=True)

    return g_cuda, g_cpu