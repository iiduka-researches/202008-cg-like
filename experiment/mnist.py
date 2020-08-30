import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from experiment.experiment import Experiment, ResultDict
from optimizer.optimizer import Optimizer


class ExperimentMNIST(Experiment):
    def prepare_data_loader(self, batch_size: int, data_dir: str) -> Tuple[DataLoader, DataLoader]:
        root = os.path.join(data_dir, 'mnist')
        os.makedirs(root, exist_ok=True)

        train_data = MNIST(root, train=True, download=True, transform=ToTensor())
        test_data = MNIST(root, train=False, download=True, transform=ToTensor())
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader

    def prepare_model(self, model_name: Optional[str]) -> Module:
        return Net()

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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.25),
            nn.Conv2d(32, 64, 5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.25),
        )
        self.mlp = nn.Sequential(
            nn.Linear(1024, 200),
            nn.Dropout(0.25),
            nn.Linear(200, 10),
        )

    def forward(self, x):
        out = self.conv_net(x)
        out = out.view(out.size(0), -1)
        out = self.mlp(out)
        return out
