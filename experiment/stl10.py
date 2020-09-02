import os
from typing import Tuple, Optional

import torch
from torch.nn import Module, CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.datasets import STL10
from torchvision.models import inception_v3
from torchvision.transforms import CenterCrop, Compose, Normalize, RandomHorizontalFlip, Resize, ToTensor
from optimizer.optimizer import Optimizer
from .experiment import Experiment, ResultDict


class ExperimentSTL10(Experiment):
    def prepare_data_loader(self, batch_size: int, data_dir: str) -> Tuple[DataLoader, DataLoader]:
        root = os.path.join(data_dir, self.dataset_name)
        os.makedirs(root, exist_ok=True)

        transform_train = Compose([
            # RandomHorizontalFlip(),
            Resize(299),
            CenterCrop(299),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        transform_test = Compose([
            Resize(299),
            CenterCrop(299),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        train_data = STL10(root, split='train', download=True, transform=transform_train)
        test_data = STL10(root, split='test', download=True, transform=transform_test)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader

    def prepare_model(self, model_name: Optional[str]) -> Module:
        r"""Inception v3 model architecture from
            `"Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>`_.
        """
        return inception_v3(pretrained=False, num_classes=10, aux_logits=False)

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
