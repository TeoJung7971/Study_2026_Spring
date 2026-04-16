import os
import pickle
import numpy as np
from torch.utils.data import Dataset


def unpickle(file):
    '''
    CIFAR-10 배치 파일을 역직렬화하여 딕셔너리로 반환한다.

    Parameters
    ----------
    file : str
        CIFAR-10 pickle 파일 경로 (예: 'data_batch_1', 'test_batch').

    Returns
    -------
    dict
        바이트 키를 가진 딕셔너리.
        주요 키: b'data' (numpy array, shape=(N, 3072)),
                 b'labels' (list of int),
                 b'filenames' (list of bytes).
    '''
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


class MNIST_Dataset(Dataset):
    '''
    MNIST 바이너리 파일(.idx3-ubyte, .idx1-ubyte)을 읽어
    PyTorch Dataset으로 제공한다.

    Parameters
    ----------
    data_dir : str
        MNIST 파일들이 위치한 디렉토리 경로.
    train : bool, optional
        True이면 훈련 데이터(train-*), False이면 테스트 데이터(t10k-*)를 로드. 기본값 True.
    transform : callable, optional
        이미지에 적용할 변환 함수 (예: torchvision.transforms.Compose).
    target_transform : callable, optional
        레이블에 적용할 변환 함수.

    Notes
    -----
    이미지는 (N, 28, 28) uint8 numpy 배열로 로드되며,
    레이블은 int64로 캐스팅된다.
    '''

    def __init__(self, data_dir, train=True, transform=None, target_transform=None):
        split = 'train' if train else 't10k'

        with open(os.path.join(data_dir, f'{split}-images.idx3-ubyte'), 'rb') as f:
            self.img = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(-1, 28, 28).copy()

        with open(os.path.join(data_dir, f'{split}-labels.idx1-ubyte'), 'rb') as f:
            self.labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8).astype(np.int64).copy()

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.img[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label


class CIFAR_Dataset(Dataset):
    '''
    CIFAR-10 pickle 배치 파일들을 읽어 PyTorch Dataset으로 제공한다.

    Parameters
    ----------
    data_dir : str
        CIFAR-10 파일들이 위치한 디렉토리 경로.
        훈련 시 data_batch_1 ~ data_batch_5, 테스트 시 test_batch를 읽는다.
    train : bool, optional
        True이면 5개 훈련 배치(50,000장), False이면 테스트 배치(10,000장)를 로드. 기본값 True.
    transform : callable, optional
        이미지에 적용할 변환 함수.
    target_transform : callable, optional
        레이블에 적용할 변환 함수.

    Notes
    -----
    원본 데이터는 (N, 3072) 형태이며, __getitem__에서 (32, 32, 3) HWC 배열로 변환된다.
    '''

    def __init__(self, data_dir, train=True, transform=None, target_transform=None):
        img_list    = []
        labels_list = []

        if train:
            for i in range(1, 6):
                batch = unpickle(os.path.join(data_dir, f'data_batch_{i}'))
                img_list.append(batch[b'data'])
                labels_list.extend(batch[b'labels'])
        else:
            batch = unpickle(os.path.join(data_dir, 'test_batch'))
            img_list.append(batch[b'data'])
            labels_list.extend(batch[b'labels'])

        self.img    = np.concatenate(img_list, axis=0)
        self.labels = np.array(labels_list, dtype=np.int64)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.img[idx].reshape(3, 32, 32).transpose(1, 2, 0)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
