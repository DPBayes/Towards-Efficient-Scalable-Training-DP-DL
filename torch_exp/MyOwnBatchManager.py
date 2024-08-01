# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import List

import numpy as np
#from opacus.optimizers import DPOptimizer
from opacus.utils.uniform_sampler import (
    DistributedUniformWithReplacementSampler,
    UniformWithReplacementSampler,
)
from torch.utils.data import BatchSampler, DataLoader, Sampler

class EndingLogicalBatchSignal():

    def __init__(self) -> None:
        self.skip_queue = []
    
    def signal_skip_step(self, do_skip=True):
        """
        Signals the optimizer to skip an optimization step and only perform clipping and
        per sample gradient accumulation.

        On every call of ``.step()`` optimizer will check the queue of skipped step
        signals. If non-empty and the latest flag is ``True``, optimizer will call
        ``self.clip_and_accumulate``, but won't proceed to adding noise and performing
        the actual optimization step.
        It also affects the behaviour of ``zero_grad()``. If the last step was skipped,
        optimizer will clear per sample gradients accumulated by
        ``self.clip_and_accumulate`` (``p.grad_sample``), but won't touch aggregated
        clipped gradients (``p.summed_grad``)

        Used by :class:`~opacus.utils.batch_memory_manager.BatchMemoryManager` to
        simulate large virtual batches with limited memory footprint.

        Args:
            do_skip: flag if next step should be skipped
        """
        self.skip_queue.append(do_skip)

    def _check_skip_next_step(self, pop_next=True):
        """
        Checks if next step should be skipped by the optimizer.
        This is for large Poisson batches that get split into smaller physical batches
        to fit on the device. Batches that do not correspond to the end of a Poisson
        batch or thus `skipped` as their gradient gets accumulated for one big step.
        """
        if self.skip_queue:
            if pop_next:
                return self.skip_queue.pop(0)
            else:
                return self.skip_queue[0]
        else:
            return False

class BatchSplittingSampler(Sampler[List[int]]):
    """
    Samples according to the underlying instance of ``Sampler``, but splits
    the index sequences into smaller chunks.

    Used to split large logical batches into physical batches of a smaller size,
    while coordinating with DPOptimizer when the logical batch has ended.
    """

    def __init__(
        self,
        *,
        sampler: Sampler[List[int]],
        max_batch_size: int,
        signaler: EndingLogicalBatchSignal,
    ):
        """

        Args:
            sampler: Wrapped Sampler instance
            max_batch_size: Max size of emitted chunk of indices
            signaler: EndingLogicalBatchSignal instance to notify when the logical batch is over
        """
        self.sampler = sampler
        self.max_batch_size = max_batch_size
        self.signaler = signaler

    def __iter__(self):
        for batch_idxs in self.sampler:
            #print('logical batch size',len(batch_idxs))
            if len(batch_idxs) == 0:
                self.signaler.signal_skip_step(do_skip=False)
                yield []
                continue

            split_idxs = np.array_split(
                batch_idxs, math.ceil(len(batch_idxs) / self.max_batch_size)
            )
            split_idxs = [s.tolist() for s in split_idxs]
            for x in split_idxs[:-1]:
                self.signaler.signal_skip_step(do_skip=True)
                yield x
            self.signaler.signal_skip_step(do_skip=False)
            yield split_idxs[-1]

    def __len__(self):
        if isinstance(self.sampler, BatchSampler):
            return int(
                len(self.sampler) * (self.sampler.batch_size / self.max_batch_size)
            )
        elif isinstance(self.sampler, UniformWithReplacementSampler) or isinstance(
            self.sampler, DistributedUniformWithReplacementSampler
        ):
            expected_batch_size = self.sampler.sample_rate * self.sampler.num_samples
            return int(len(self.sampler) * (expected_batch_size / self.max_batch_size))

        return len(self.sampler)


def wrap_data_loader(
    *, data_loader: DataLoader, 
    max_batch_size: int, 
    signaler: EndingLogicalBatchSignal
):
    """
    Replaces batch_sampler in the input data loader with ``BatchSplittingSampler``

    Args:
        data_loader: Wrapper DataLoader
        max_batch_size: max physical batch size we want to emit
        signaler: EndingLogicalBatchSignal instance used for stopping

    Returns:
        New DataLoader instance with batch_sampler wrapped in ``BatchSplittingSampler``
    """

    return DataLoader(
        dataset=data_loader.dataset,
        batch_sampler=BatchSplittingSampler(
            sampler=data_loader.batch_sampler,
            max_batch_size=max_batch_size,
            signaler=signaler,
        ),
        num_workers=data_loader.num_workers,
        collate_fn=data_loader.collate_fn,
        pin_memory=data_loader.pin_memory,
        timeout=data_loader.timeout,
        worker_init_fn=data_loader.worker_init_fn,
        multiprocessing_context=data_loader.multiprocessing_context,
        generator=data_loader.generator,
        prefetch_factor=data_loader.prefetch_factor,
        persistent_workers=data_loader.persistent_workers,
    )


class MyBatchMemoryManager(object):
    """
    Context manager to manage memory consumption during training.

    Allows setting hard limit on the physical batch size as a just one line code change.
    Can be used both for simulating large logical batches with limited memory and for
    safeguarding against occasional large batches produced by
    :class:`~opacus.utils.uniform_sampler.UniformWithReplacementSampler`.

    Note that it doesn't modify the input DataLoader, you'd need to use new DataLoader
    returned by the context manager.

    BatchSplittingSampler will split large logical batches into smaller sub-batches with
    certain maximum size.

    This implementation is different than the Opacus one. It does not take the optimizer,
    but instead uses a signaler, to tell the optimizer when to take a step, instead of 
    keep accumulating.

    It adapts to the library being used, either Private Vision or FastDP. Because their optimizers behave
    differently.

    Example:
        >>> # Assuming you've initialized your objects and passed them to PrivacyEngine.
        >>> # For this example we assume data_loader is initialized with batch_size=4
        >>> model, optimizer, data_loader = _init_private_training()
        >>> criterion = nn.CrossEntropyLoss()
        >>> flag = EndingLogicalBatchSignal()
        >>> with BatchMemoryManager(
        ...     data_loader=data_loader, max_physical_batch_size=2, signaler=flag
        ... ) as new_data_loader:
        ...     for data, label in new_data_loader:
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
    """

    def __init__(
        self,
        *,
        data_loader: DataLoader,
        max_physical_batch_size: int,
        signaler: EndingLogicalBatchSignal,
    ):
        self.data_loader = data_loader
        self.signaler = signaler
        self.max_physical_batch_size = max_physical_batch_size

    def __enter__(self):
        return wrap_data_loader(
            data_loader=self.data_loader,
            max_batch_size=self.max_physical_batch_size,
            signaler=self.signaler,
        )

    def __exit__(self, type, value, traceback):
        pass
