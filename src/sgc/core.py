"""Entities, interfaces and core
functionalities.
"""
from __future__ import annotations

import time
from abc import ABC
from pathlib import Path
from typing import Callable

import gdown
import pandas as pd
import torch
from torch import nn, optim
from torchmetrics.classification import Accuracy, F1Score, Precision, Recall
from torchvision.io import read_image


class Trainer(ABC):
    def __init__(self,
                 network: nn.Module,
                 optimizer: optim.optimizer.Optimizer,
                 loss_fn: Callable,
                 batch_size: int,
                 epochs: int,
                 train_metrics: bool = True,
                 verbose: bool = True,
                 device=None):
        self._store_metrics = train_metrics
        self._batch_size = batch_size
        self._device = device
        self._optim = optimizer
        self._nn = network
        self._verbose = verbose
        self._epochs = epochs
        self._loss = loss_fn
        self._train_error = []
        self._val_error = []

        # Send network to the correct device
        self._nn.to(self._device)

    def fit(self,
            train: torch.utils.data.Dataset,
            val: torch.utils.data.Dataset | None = None) -> None:
        # Create DataLoader from dataset
        train_loader = torch.utils.data.DataLoader(dataset=train,
                                                   batch_size=self._batch_size,
                                                   pin_memory=True,
                                                   shuffle=True)
        val_loader = torch.utils.data.DataLoader(dataset=val,
                                                 batch_size=self._batch_size,
                                                 pin_memory=True,
                                                 shuffle=False)

        # Store initial validation and train error
        self._evaluate_on(val_loader,
                          is_val_loader=True)
        self._evaluate_on(train_loader,
                          is_val_loader=False,
                          switch_to_train=True)

        # Start training
        for i in range(self._epochs):
            start = time.perf_counter()
            total_loss = torch.tensor(0.0).to(self._device)
            n = 0

            # Set network to train mode
            self._nn.train()

            # For each batch
            for X, y in train_loader:
                X = X.to(self._device)
                y = y.to(self._device)

                # Zero gradient for optimizer
                self._optim.zero_grad()

                # Predictions for current batch
                predictions = self._nn(X)

                # Loss for current batch
                loss = self._loss(predictions, y)

                # Backward pass
                loss.backward()

                # Optimization step
                self._optim.step()

                # Update epoch loss
                with torch.no_grad():
                    total_loss += loss
                    n += 1

            # Obtain average loss over batches
            total_loss = (total_loss / n).cpu()
            duration = time.perf_counter() - start

            # Evaluate on validation set
            self._evaluate_on(val_loader,
                              is_val_loader=True,
                              switch_to_train=True)

            # Maybe print progress
            if self._verbose:
                print(f'Epoch {i}: '
                      f'Train loss {total_loss:.5f} '
                      f'({duration:.1f}s)')

            # Maybe store metrics
            if self._store_metrics:
                self._train_error.append(total_loss)

    def predict(self, X: torch.Tensor):
        self._nn.eval()
        return self._nn(X)

    def train_history(self) -> tuple[torch.Tensor, torch.Tensor]:
        train_error = torch.tensor(self._train_error)
        val_error = torch.tensor(self._val_error)
        return train_error, val_error

    def classification_metrics(self,
                               ds: torch.utils.data.Dataset,
                               cls_mapper: dict = None) -> pd.DataFrame:
        target = []
        preds = []
        loader = torch.utils.data.DataLoader(dataset=ds,
                                             batch_size=1,
                                             pin_memory=True,
                                             shuffle=False)
        # Collect predictions
        for X, y in loader:
            X = X.to(self._device)
            target.append(y)

            # Obtain predictions
            pred = self.predict(X).cpu()
            _, pred = torch.max(pred, 1)
            preds.append(pred)

        # Concat
        target = torch.concat(target)
        preds = torch.concat(preds)
        n_classes = target.unique().size(dim=0)
        data = []

        # If no mapper was provided,
        #   create a default one
        if cls_mapper is None:
            cls_mapper = {i: str(i) for i in range(n_classes)}

        # Metrics
        for avg in [None, 'weighted', 'micro', 'macro']:
            for m in [F1Score, Precision, Recall, Accuracy]:
                value = m(task="multiclass",
                          num_classes=n_classes,
                          average=avg)(preds, target).numpy()
                name = m.__name__

                if not avg:
                    data.extend([dict(Class=cls_mapper[i],
                                      Value=v,
                                      Metric=name)
                                 for i, v, in enumerate(value)])
                else:
                    data.append(dict(Class=avg.capitalize(),
                                     Value=value,
                                     Metric=name))

        # Create DataFrane
        return pd.DataFrame(data).sort_values(by=["Class", "Metric"])

    def _evaluate_on(self,
                     loader,
                     is_val_loader: bool,
                     switch_to_train: bool = False):
        # Set network to evaluation mode
        self._nn.eval()

        # Set list to append
        target = self._train_error
        if is_val_loader:
            target = self._val_error

        # If shouldn't store metrics, return
        if not self._store_metrics:
            return

        # Obtain error in test set
        with torch.no_grad():
            total_loss = torch.tensor(0.0).to(self._device)
            n = 0
            for X, y in loader:
                X, y = X.to(self._device), y.to(self._device)
                with torch.no_grad():
                    total_loss += self._loss(self._nn(X), y)
                    n += 1

            # Store test loss
            total_loss = (total_loss / n).cpu()
            target.append(total_loss)

        # Maybe set the network to train mode
        if switch_to_train:
            self._nn.train()


class Dataset(torch.utils.data.Dataset):
    """Star-Galaxy classification data.

    The splits were created manually following
        a 70-30 strategy.

    Freely available at Kaggle:
        https://doi.org/10.34740/kaggle/ds/1396185
    """
    _URL = ('https://drive.google.com/uc?id='
            '10Abx-1eFraeOV1FrYxBnjq2lVDlF8NeR')
    _DIR = Path(__file__).parent.joinpath('star-galaxy-dataset')

    def __init__(self,
                 train_split: bool,
                 data_dir=None,
                 transform=None,
                 target_transform=None):
        if data_dir is None:
            data_dir = self._DIR

        self._data_dir = data_dir
        self._transform = transform
        self._target_transform = target_transform

        # Maybe download the dataset
        if not self._data_dir.exists():
            self._download()

        # Selecting the correct split
        self._labels = pd.read_csv(self._data_dir.joinpath('labels.csv'))
        cond = self._labels.is_train == train_split
        self._labels = self._labels.loc[cond]

    @property
    def data_dir(self) -> Path:
        return self._data_dir

    @property
    def label_mapper(self) -> dict:
        return {0: 'Star', 1: 'Galaxy'}

    @property
    def metadata(self) -> pd.DataFrame:
        return self._labels.copy()

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        img_path = self._data_dir.joinpath(self._labels.iloc[idx, 0])
        image = read_image(str(img_path))
        label = self._labels.iloc[idx, 1]

        if self._transform:
            image = self._transform(image)

        if self._target_transform:
            label = self._target_transform(label)

        return image, label

    def _download(self):
        self._data_dir.mkdir(parents=False, exist_ok=False)
        zip_path = self._data_dir.joinpath('data.zip')

        # Fazendo download do zip
        gdown.cached_download(self._URL,
                              str(zip_path),
                              postprocess=gdown.extractall)

        # Apagando o zip após download e extraçaõ
        zip_path.unlink()
