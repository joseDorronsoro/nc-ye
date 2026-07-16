# ------------------------------------------------------------------ Unchanged:
import joblib
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class MyData(Dataset):
    """
    Custom dataset for loading and transforming data.

    Args:
        x (torch.Tensor): Input data tensor.
        y (np.ndarray or torch.Tensor): Target labels.
        transform (callable, optional): Optional transform to be applied on a sample.

    Attributes:
        x (torch.Tensor): Input data tensor.
        y (np.ndarray or torch.Tensor): Target labels.
        transform (callable, optional): Transform to apply to each sample.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(index): Retrieves the sample and label at the specified index, applying the transform if provided.
    """
    def __init__(self, x, y, transform=None):
        """
        Initializes the MyData dataset.

        Args:
            x (torch.Tensor): Input data tensor.
            y (np.ndarray or torch.Tensor): Target labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.x = x
        self.y = y 
        self.transform = transform

    def __len__(self):
        """
        Returns:
            int: Number of samples in the dataset.
        """
        return self.y.shape[0]

    def __getitem__(self, index):
        """
        Retrieves the sample and label at the specified index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple: (sample, label) where sample is the transformed input and label is the corresponding target.
        """
        sample = self.x[index]
        if self.transform:
            sample = self.transform(sample)
        return sample, self.y[index]


def loader_from_numpy(x, y, x_ts, y_ts, batch_size=128, train_shuffle=False):
    """
    Loads data into PyTorch DataLoaders with optional shuffling.

    Args:
        x (np.ndarray or torch.Tensor): Training data.
        y (np.ndarray or torch.Tensor): Training labels.
        x_ts (np.ndarray or torch.Tensor): Test data.
        y_ts (np.ndarray or torch.Tensor): Test labels.
        shuffle (bool): Whether to shuffle the training data. Test data is not shuffled

    Returns:
        tuple: (train_loader, test_loader)

    Raises:
        TypeError: If x, y, x_ts, or y_ts are not numpy arrays or torch tensors.
        ValueError: If x and y or x_ts and y_ts do not have the same number of samples.
    """
    # Argument control
    if not (isinstance(x, (np.ndarray, torch.Tensor)) and isinstance(x_ts, (np.ndarray, torch.Tensor))):
        raise TypeError("x and x_ts must be numpy arrays or torch tensors.")
    if not (isinstance(y, (np.ndarray, torch.Tensor)) and isinstance(y_ts, (np.ndarray, torch.Tensor))):
        raise TypeError("y and y_ts must be numpy arrays or torch tensors.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of samples.")
    if x_ts.shape[0] != y_ts.shape[0]:
        raise ValueError("x_ts and y_ts must have the same number of samples.")

    mm, ss = x.mean(), x.std()

    # mnist image params
    im_size = 28
    padded_im_size = 32
    transform = transforms.Compose(
        [
            transforms.Pad((padded_im_size - im_size) // 2),
            transforms.Normalize(mm, ss),
        ]
    )

    xx = torch.Tensor(x.reshape(-1, 1, 28, 28))
    xx_ts = torch.Tensor(x_ts.reshape(-1, 1, 28, 28))
    
    train_data = MyData(xx, y, transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=train_shuffle)

    test_data = MyData(xx_ts, y_ts, transform)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def ye_targs(pr):
    """
    Construct Ye target vectors from class probabilities.
    """
    return (1. / np.sqrt(pr)) * np.eye(len(pr)) -  np.sqrt(pr).reshape(-1, 1) * np.ones(len(pr)).reshape(1, -1)


def encode_targets(
        y,
        encoding,
        n_classes,
        probabilities=None):
    """
    Encode class labels using OHE or Ye target representations.
    """
    if encoding == 'ohe':
        basis = np.eye(n_classes)
        return np.array([basis[:, int(lbl)] for lbl in y])

    ye_basis = (
        np.diag(1.0 / np.sqrt(probabilities))
        - np.sqrt(probabilities).reshape(-1, 1)
        @ np.ones((1, n_classes))
    )

    return np.array([
        ye_basis[:, int(lbl)]
        for lbl in y
    ])


def imbalanced_subsampling(
        bunch,
        full_class_size,
        full_class_labels,
        frac,
        target_encoding='ohe'):
    """
    Generate imbalanced train data by subsampling minority classes
    and encode targets using OHE or Ye representations.
    """
    target = bunch['target']

    classes = sorted(np.unique(target))

    indices_dict = {
        cls: np.where(target == cls)[0]
        for cls in classes
    }

    minority_classes = [
        cls for cls in classes
        if cls not in full_class_labels
    ]

    full_idx = [
        indices_dict[c][:full_class_size]
        for c in full_class_labels
    ]

    minority_idx = [
        indices_dict[c][:int(full_class_size * frac)]
        for c in minority_classes
    ]

    selected = np.concatenate(
        full_idx + minority_idx
    )

    x_train = bunch['data'][selected]
    y_train = bunch['target'][selected]

    x_test = bunch['data_test']
    y_test = bunch['target_test']

    n_classes = len(classes)

    if target_encoding == 'ohe':

        yy_train = encode_targets(
            y_train,
            'ohe',
            n_classes
        )

        yy_test = encode_targets(
            y_test,
            'ohe',
            n_classes
        )

    else:

        probs = np.array([
            1.0 if c in full_class_labels else frac
            for c in classes
        ])

        probs /= probs.sum()

        yy_train = encode_targets(
            y_train,
            'ye',
            n_classes,
            probs
        )

        yy_test = encode_targets(
            y_test,
            'ye',
            n_classes,
            probs
        )

    return x_train, yy_train, x_test, yy_test


def load_dataset(cfg, data_dir):
    """
    Load the experimental dataset and prepare it for training.

    The function performs imbalanced subsampling, target encoding
    (OHE or Ye) and DataLoader construction according to the
    experiment configuration.
    """

    full_class_size = (
        6000
        if "fashion_mnist" in cfg.bunch_file
        else 5000
    )

    bunch = joblib.load(
        data_dir + cfg.bunch_file
    )

    x_tr, y_tr, x_ts, y_ts = (
        imbalanced_subsampling(
            bunch=bunch,
            full_class_size=full_class_size,
            full_class_labels=[0, 1, 2, 3, 4],
            frac=cfg.frac,
            target_encoding=cfg.encoding,
        )
    )

    train_loader, test_loader = (
        loader_from_numpy(
            x_tr,
            y_tr,
            x_ts,
            y_ts,
            batch_size=cfg.batch_size,
            train_shuffle=True,
        )
    )

    return train_loader, test_loader
