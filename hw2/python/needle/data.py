import numpy as np
from .autograd import Tensor
import gzip
import struct
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
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
        if flip_img:
            img = np.flip(img, axis=1)
        return img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        ### BEGIN YOUR SOLUTION
        padded_img = np.pad(img, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant')
        clipped_img = padded_img[shift_x+self.padding:shift_x+self.padding+img.shape[0],
                                 shift_y+self.padding:shift_y+self.padding+img.shape[1]]
        return clipped_img
        ### END YOUR SOLUTION


class Dataset:
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


class DataLoader:
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
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)), 
                                           range(batch_size, len(dataset), batch_size))

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        if self.shuffle:
            temp_index = np.arange(len(self.dataset))
            np.random.shuffle(temp_index)
            self.ordering = np.array_split(temp_index,
                                            range(self.batch_size, len(self.dataset), self.batch_size))
        self.cur_batch_id = 0
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        if self.cur_batch_id >= len(self.ordering):
            raise StopIteration
        batch_data_X = [self.dataset[i][0] for i in self.ordering[self.cur_batch_id]]
        batch_data_X = Tensor(batch_data_X)
        if len(self.dataset[0]) == 1:
            batch_data = (batch_data_X, )
        else:
            
            batch_data_y = [self.dataset[i][1] for i in self.ordering[self.cur_batch_id]]
            batch_data_y = Tensor(batch_data_y)
            batch_data = (batch_data_X, batch_data_y)
        self.cur_batch_id += 1
        return batch_data
        ### END YOUR SOLUTION


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        self.image_filename = image_filename
        self.label_filename = label_filename
        self.transforms = transforms
        
        self.images, self.labels = self._parse_mnist(self.image_filename, self.label_filename)
        ### END YOUR SOLUTION

    def _parse_mnist(self, image_filename: str, label_filename: str):
        with gzip.open(image_filename, "rb") as img_file:
            magic_num, imgs_num, rows, cols = struct.unpack(">4I", img_file.read(16)) 
            img_list = []
            for _ in range(imgs_num):
                cur_img = np.array(struct.unpack(f">{rows*cols}B", img_file.read(rows * cols)), dtype="float32")
                img_list.append(cur_img)
        
        X = np.stack(img_list)

        with gzip.open(label_filename, "rb") as lab_file:
            magic_num, labs_num = struct.unpack(">2I", lab_file.read(8))
            y = np.array(struct.unpack(f">{labs_num}B", lab_file.read(labs_num)), dtype="int8")
        
        X = (X - X.min()) / (X.max() - X.min())

        X = X.reshape(imgs_num, rows, cols, 1)
        return X, y

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        image, label = self.images[index], self.labels[index]
        if self.transforms:
            for transform in self.transforms:
                image = transform(image)
        return (image, label)
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.images.shape[0]
        ### END YOUR SOLUTION

class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])
