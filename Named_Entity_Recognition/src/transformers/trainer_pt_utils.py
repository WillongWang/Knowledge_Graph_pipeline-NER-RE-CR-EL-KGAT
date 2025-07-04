# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
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
"""
Torch utilities for the Trainer class.
"""

import math
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, Sampler

from .file_utils import is_torch_tpu_available
from .utils import logging


if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm

# this is used to supress an undesired warning emitted by pytorch versions 1.4.2-1.7.0
try:
    from torch.optim.lr_scheduler import SAVE_STATE_WARNING
except ImportError:
    SAVE_STATE_WARNING = ""

logger = logging.get_logger(__name__)


def torch_pad_and_concatenate(tensor1, tensor2, padding_index=-100):
    """Concatenates `tensor1` and `tensor2` on first axis, applying padding on the second if necessary."""
    if len(tensor1.shape) == 1 or tensor1.shape[1] == tensor2.shape[1]:
        return torch.cat((tensor1, tensor2), dim=0)

    # Let's figure out the new shape
    new_shape = (tensor1.shape[0] + tensor2.shape[0], max(tensor1.shape[1], tensor2.shape[1])) + tensor1.shape[2:]

    # Now let's fill the results tensor
    result = tensor1.new_full(new_shape, padding_index)
    result[: tensor1.shape[0], : tensor1.shape[1]] = tensor1
    result[tensor1.shape[0] :, : tensor2.shape[1]] = tensor2
    return result


def numpy_pad_and_concatenate(array1, array2, padding_index=-100):
    """Concatenates `array1` and `array2` on first axis, applying padding on the second if necessary."""
    if len(array1.shape) == 1 or array1.shape[1] == array2.shape[1]:
        return np.concatenate((array1, array2), dim=0)

    # Let's figure out the new shape
    new_shape = (array1.shape[0] + array2.shape[0], max(array1.shape[1], array2.shape[1])) + array1.shape[2:]

    # Now let's fill the results tensor
    result = np.full_like(array1, padding_index, shape=new_shape)
    result[: array1.shape[0], : array1.shape[1]] = array1
    result[array1.shape[0] :, : array2.shape[1]] = array2
    return result


def nested_concat(tensors, new_tensors, padding_index=-100):
    """
    Concat the `new_tensors` to `tensors` on the first dim and pad them on the second if needed. Works for tensors or
    nested list/tuples of tensors.
    """
    assert type(tensors) == type(
        new_tensors
    ), f"Expected `tensors` and `new_tensors` to have the same type but found {type(tensors)} and {type(new_tensors)}."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_concat(t, n, padding_index=padding_index) for t, n in zip(tensors, new_tensors))
    elif isinstance(tensors, torch.Tensor):
        return torch_pad_and_concatenate(tensors, new_tensors, padding_index=padding_index)
    elif isinstance(tensors, np.ndarray):
        return numpy_pad_and_concatenate(tensors, new_tensors, padding_index=padding_index)
    else:
        raise TypeError(f"Unsupported type for concatenation: got {type(tensors)}")


def nested_numpify(tensors):
    "Numpify `tensors` (even if it's a nested list/tuple of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_numpify(t) for t in tensors)
    return tensors.cpu().numpy()


def nested_detach(tensors):
    "Detach `tensors` (even if it's a nested list/tuple of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_detach(t) for t in tensors)
    return tensors.detach()


def nested_xla_mesh_reduce(tensors, name):
    if is_torch_tpu_available():
        import torch_xla.core.xla_model as xm

        if isinstance(tensors, (list, tuple)):
            return type(tensors)(nested_xla_mesh_reduce(t, f"{name}_{i}") for i, t in enumerate(tensors))
        return xm.mesh_reduce(name, tensors, torch.cat)
    else:
        raise ImportError("Torch xla must be installed to use `nested_xla_mesh_reduce`")


def distributed_concat(tensor: "torch.Tensor", num_total_examples: Optional[int] = None) -> torch.Tensor:
    try:
        if isinstance(tensor, (tuple, list)):
            return type(tensor)(distributed_concat(t, num_total_examples) for t in tensor)
        output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output_tensors, tensor)
        concat = torch.cat(output_tensors, dim=0)

        # truncate the dummy elements added by SequentialDistributedSampler
        if num_total_examples is not None:
            concat = concat[:num_total_examples]
        return concat
    except AssertionError:
        raise AssertionError("Not currently using distributed training")


def distributed_broadcast_scalars(
    scalars: List[Union[int, float]], num_total_examples: Optional[int] = None
) -> torch.Tensor:
    try:
        tensorized_scalar = torch.tensor(scalars).cuda()
        output_tensors = [tensorized_scalar.clone() for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output_tensors, tensorized_scalar)
        concat = torch.cat(output_tensors, dim=0)

        # truncate the dummy elements added by SequentialDistributedSampler
        if num_total_examples is not None:
            concat = concat[:num_total_examples]
        return concat
    except AssertionError:
        raise AssertionError("Not currently using distributed training")


def reissue_pt_warnings(caught_warnings):
    # Reissue warnings that are not the SAVE_STATE_WARNING
    if len(caught_warnings) > 1:
        for w in caught_warnings:
            if w.category != UserWarning or w.message != SAVE_STATE_WARNING:
                warnings.warn(w.message, w.category)


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.

    Args:
        local_rank (:obj:`int`): The rank of the local process.
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()


class SequentialDistributedSampler(Sampler):
    """
    Distributed Sampler that subsamples indices sequentially, making it easier to collate all results at the end.

    Even though we only use this sampler for eval and predict (no training), which means that the model params won't
    have to be synced (i.e. will not hang for synchronization even if varied number of forward passes), we still add
    extra samples to the sampler to make it evenly divisible (like in `DistributedSampler`) to make it easy to `gather`
    or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert (
            len(indices) == self.total_size
        ), f"Indices length {len(indices)} and total size {self.total_size} mismatched"

        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        assert (
            len(indices) == self.num_samples
        ), f"Indices length {len(indices)} and sample number {self.num_samples} mismatched"

        return iter(indices)

    def __len__(self):
        return self.num_samples


def get_tpu_sampler(dataset: torch.utils.data.dataset.Dataset):
    if xm.xrt_world_size() <= 1:
        return RandomSampler(dataset)
    return DistributedSampler(dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal())


def nested_new_like(arrays, num_samples, padding_index=-100):
    """ Create the same nested structure as `arrays` with a first dimension always at `num_samples`."""
    if isinstance(arrays, (list, tuple)):
        return type(arrays)(nested_new_like(x, num_samples) for x in arrays)
    return np.full_like(arrays, padding_index, shape=(num_samples, *arrays.shape[1:]))


def nested_expand_like(arrays, new_seq_length, padding_index=-100):
    """ Expand the `arrays` so that the second dimension grows to `new_seq_length`. Uses `padding_index` for padding."""
    if isinstance(arrays, (list, tuple)):
        return type(arrays)(nested_expand_like(x, new_seq_length, padding_index=padding_index) for x in arrays)

    result = np.full_like(arrays, padding_index, shape=(arrays.shape[0], new_seq_length) + arrays.shape[2:])
    result[:, : arrays.shape[1]] = arrays
    return result


def nested_truncate(tensors, limit):
    "Truncate `tensors` at `limit` (even if it's a nested list/tuple of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_truncate(t, limit) for t in tensors)
    return tensors[:limit]


def _get_first_shape(arrays):
    """Return the shape of the first array found in the nested struct `arrays`."""
    if isinstance(arrays, (list, tuple)):
        return _get_first_shape(arrays[0])
    return arrays.shape


class DistributedTensorGatherer:
    """
    A class responsible for properly gathering tensors (or nested list/tuple of tensors) on the CPU by chunks.

    If our dataset has 16 samples with a batch size of 2 on 3 processes and we gather then transfer on CPU at every
    step, our sampler will generate the following indices:

        :obj:`[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1]`

    to get something of size a multiple of 3 (so that each process gets the same dataset length). Then process 0, 1 and
    2 will be responsible of making predictions for the following samples:

        - P0: :obj:`[0, 1, 2, 3, 4, 5]`
        - P1: :obj:`[6, 7, 8, 9, 10, 11]`
        - P2: :obj:`[12, 13, 14, 15, 0, 1]`

    The first batch treated on each process will be

        - P0: :obj:`[0, 1]`
        - P1: :obj:`[6, 7]`
        - P2: :obj:`[12, 13]`

    So if we gather at the end of the first batch, we will get a tensor (nested list/tuple of tensor) corresponding to
    the following indices:

        :obj:`[0, 1, 6, 7, 12, 13]`

    If we directly concatenate our results without taking any precautions, the user will then get the predictions for
    the indices in this order at the end of the prediction loop:

        :obj:`[0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5, 10, 11, 0, 1]`

    For some reason, that's not going to roll their boat. This class is there to solve that problem.

    Args:

        world_size (:obj:`int`):
            The number of processes used in the distributed training.
        num_samples (:obj:`int`):
            The number of samples in our dataset.
        make_multiple_of (:obj:`int`, `optional`):
            If passed, the class assumes the datasets passed to each process are made to be a multiple of this argument
            (by adding samples).
        padding_index (:obj:`int`, `optional`, defaults to -100):
            The padding index to use if the arrays don't all have the same sequence length.
    """

    def __init__(self, world_size, num_samples, make_multiple_of=None, padding_index=-100):
        self.world_size = world_size
        self.num_samples = num_samples
        total_size = world_size if make_multiple_of is None else world_size * make_multiple_of
        self.total_samples = int(np.ceil(num_samples / total_size)) * total_size
        self.process_length = self.total_samples // world_size
        self._storage = None
        self._offsets = None
        self.padding_index = padding_index

    def add_arrays(self, arrays):
        """
        Add :obj:`arrays` to the internal storage, Will initialize the storage to the full size at the first arrays
        passed so that if we're bound to get an OOM, it happens at the beginning.
        """
        if arrays is None:
            return
        if self._storage is None:
            self._storage = nested_new_like(arrays, self.total_samples, padding_index=self.padding_index)
            self._offsets = list(range(0, self.total_samples, self.process_length))
        else:
            storage_shape = _get_first_shape(self._storage)
            arrays_shape = _get_first_shape(arrays)
            if len(storage_shape) > 1 and storage_shape[1] < arrays_shape[1]:
                # If we get new arrays that are too big too fit, we expand the shape fo the storage
                self._storage = nested_expand_like(self._storage, arrays_shape[1], padding_index=self.padding_index)
        slice_len = self._nested_set_tensors(self._storage, arrays)
        for i in range(self.world_size):
            self._offsets[i] += slice_len

    def _nested_set_tensors(self, storage, arrays):
        if isinstance(arrays, (list, tuple)):
            for x, y in zip(storage, arrays):
                slice_len = self._nested_set_tensors(x, y)
            return slice_len
        assert (
            arrays.shape[0] % self.world_size == 0
        ), f"Arrays passed should all have a first dimension multiple of {self.world_size}, found {arrays.shape[0]}."

        slice_len = arrays.shape[0] // self.world_size
        for i in range(self.world_size):
            if len(arrays.shape) == 1:
                storage[self._offsets[i] : self._offsets[i] + slice_len] = arrays[i * slice_len : (i + 1) * slice_len]
            else:
                storage[self._offsets[i] : self._offsets[i] + slice_len, : arrays.shape[1]] = arrays[
                    i * slice_len : (i + 1) * slice_len
                ]
        return slice_len

    def finalize(self):
        """
        Return the properly gathered arrays and truncate to the number of samples (since the sampler added some extras
        to get each process a dataset of the same length).
        """
        if self._storage is None:
            return
        if self._offsets[0] != self.process_length:
            logger.warn("Not all data has been set. Are you sure you passed all values?")
        return nested_truncate(self._storage, self.num_samples)


@dataclass
class LabelSmoother:
    """
    Adds label-smoothing on a pre-computed output from a Transformers model.

    Args:
        epsilon (:obj:`float`, `optional`, defaults to 0.1):
            The label smoothing factor.
        ignore_index (:obj:`int`, `optional`, defaults to -100):
            The index in the labels to ignore when computing the loss.
    """

    epsilon: float = 0.1
    ignore_index: int = -100

    def __call__(self, model_output, labels):
        model_loss = model_output["loss"] if isinstance(model_output, dict) else model_output[0]
        logits = model_output["logits"] if isinstance(model_output, dict) else model_output[1]
        log_probs = -torch.nn.functional.log_softmax(logits, dim=-1)

        # Look at the ignored index and mask the corresponding log_probs.
        padding_mask = labels.unsqueeze(-1).eq(self.ignore_index)
        log_probs.masked_fill_(padding_mask, 0.0)

        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        smoothed_loss = log_probs.mean(dim=-1).sum() / (padding_mask.numel() - padding_mask.long().sum())
        return (1 - self.epsilon) * model_loss + self.epsilon * smoothed_loss
