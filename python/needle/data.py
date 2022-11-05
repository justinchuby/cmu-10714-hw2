import abc
import gzip
import struct
from typing import Any, Iterable, Iterator, List, Optional, Sized, Union
from collections import OrderedDict

import numpy as np

from .autograd import Tensor


class Transform(abc.ABC):
    @abc.abstractmethod
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img: np.ndarray):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if not flip_img:
            return img
        return np.flip(img, axis=1)
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(
            low=-self.padding, high=self.padding + 1, size=2
        )
        ### BEGIN YOUR SOLUTION
        # The tests specified an opposite direction
        shift_x = -shift_x
        shift_y = -shift_y
        shifted = np.roll(img, shift_x, axis=0)
        shifted = np.roll(shifted, shift_y, axis=1)
        if shift_x < 0:
            shifted[shift_x:, :] = 0
        elif shift_x > 0:
            shifted[:shift_x, :] = 0
        if shift_y < 0:
            shifted[:, shift_y:] = 0
        elif shift_y > 0:
            shifted[:, :shift_y] = 0
        return shifted
        ### END YOUR SOLUTION


class Dataset(abc.ABC):
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader(abc.ABC):
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(
                np.arange(len(dataset)), range(batch_size, len(dataset), batch_size)
            )
        self._i = 0

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        # Reshuffle the ordering if shuffle
        if self.shuffle:
            self.ordering = np.array_split(
                np.random.permutation(len(self.dataset)),
                range(self.batch_size, len(self.dataset), self.batch_size),
            )
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        if self._i >= len(self.ordering):
            self._i = 0
            raise StopIteration
        batch = self.ordering[self._i]
        output = OrderedDict()
        # TODO: Can this be simplified?
        for i in batch:
            example = self.dataset[i]
            if isinstance(example, tuple):
                for j, value in enumerate(example):
                    output.setdefault(j, []).append(value)
            else:
                output.setdefault(0, []).append(example)
        for key in output:
            output[key] = Tensor(np.stack(output[key]))
        self._i += 1
        return tuple(output.values())
        ### END YOUR SOLUTION


def parse_mnist(
    image_filesname: str, label_filename: str
) -> tuple[np.ndarray, np.ndarray, int]:
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        tuple (X,y):
            X (numpy.ndarray[np.float32]): 3D numpy array containing the loaded
                data.

            y (numpy.ndarray[dypte=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    # Load images
    with gzip.open(image_filesname) as f:
        magic, n_images, rows, cols = struct.unpack(">IIII", f.read(16))
        print("magic number is", magic)
        # f already points to the first image
        image_data = np.frombuffer(f.read(), dtype=np.uint8).reshape(
            n_images, rows, cols
        )
        images = image_data.astype(np.float32) / 255.0

    with gzip.open(label_filename) as f:
        magic, n_labels = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return images, labels, n_images


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        self.images, self.labels, self.n_images = parse_mnist(
            image_filename, label_filename
        )
        self.transforms = transforms
        ### END YOUR SOLUTION

    def __getitem__(self, index_or_slice) -> object:
        # NOTE: Input can be a slice which is tricky to handle
        ### BEGIN YOUR SOLUTION
        if isinstance(index_or_slice, slice):
            image = self.images[index_or_slice]
            label = self.labels[index_or_slice]
            image = image.transpose(1, 2, 0)
            image = self.apply_transforms(image)
            image = image.transpose(2, 0, 1)
            return image, label

        image = self.images[index_or_slice]
        label = self.labels[index_or_slice]
        image = self.apply_transforms(image)
        return image, label
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.n_images
        ### END YOUR SOLUTION


class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])
