import os
from typing import Optional, Tuple

import torch
from torch.nn import Module
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

from experiment.experiment import Experiment, ResultDict
from model.resnet import resnet20, resnet32, resnet44, resnet56, resnet110
from optimizer.optimizer import Optimizer

MODEL_DICT = dict(
    ResNet20=resnet20,
    ResNet32=resnet32,
    ResNet44=resnet44,
    ResNet56=resnet56,
    ResNet110=resnet110,
)


class ExperimentCIFAR10(Experiment):
    def prepare_data_loader(self, batch_size: int, data_dir: str) -> Tuple[DataLoader, DataLoader, dict]:
        root = os.path.join(data_dir, 'cifar10')
        os.makedirs(root, exist_ok=True)

        train_data = CIFAR10(root, train=True, download=True, transform=ToTensor())
        test_data = CIFAR10(root, train=False, download=True, transform=ToTensor())
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader, dict()

    def prepare_model(self, model_name: Optional[str], **kwargs) -> Module:
        if model_name in MODEL_DICT:
            return MODEL_DICT[model_name](**kwargs)
        else:
            raise ValueError(f'Invalid model name: {model_name}')

    def epoch_train(self, net: Module, optimizer: Optimizer, train_loader: DataLoader) -> Tuple[Module, ResultDict]:
        running_loss = 0.0
        i = 0
        total = 0
        correct = 0
        criterion = CrossEntropyLoss()
        for inputs, labels in train_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step(closure=None)
            running_loss += loss.item()
            total += labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            i += 1
        return net, dict(train_loss=running_loss / i, train_accuracy=correct / total)

    def epoch_validate(self, net: Module, test_loader: DataLoader, **kwargs) -> ResultDict:
        running_loss = 0.0
        i = 0
        total = 0
        correct = 0
        criterion = CrossEntropyLoss()
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                i += 1
        return dict(test_loss=running_loss / i, test_accuracy=correct / total)

