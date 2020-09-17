import os
from typing import Tuple, Optional

import torch
from torch.nn import Embedding, Linear, LSTM, Module, BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torchtext.data import BucketIterator, Field, LabelField
from torchtext.datasets import IMDB

from optimizer.optimizer import Optimizer
from .experiment import Experiment, ResultDict


class ExperimentIMDb(Experiment):
    def prepare_data_loader(self, batch_size: int, data_dir: str) -> Tuple[DataLoader, DataLoader, dict]:
        root = os.path.join(data_dir, 'imdb')
        os.makedirs(root, exist_ok=True)

        text = Field(sequential=True, fix_length=80, batch_first=True, lower=True)
        label = LabelField(sequential=False)
        train_data, test_data = IMDB.splits(root=root, text_field=text, label_field=label)

        # build the vocabulary
        text.build_vocab(train_data, max_size=25000)
        label.build_vocab(train_data)
        vocab_size = len(text.vocab)
        print(vocab_size)

        # train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        # test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        train_loader, test_loader = BucketIterator.splits((train_data, test_data), batch_size=batch_size)
        return train_loader, test_loader, dict(in_dim=vocab_size)

    def prepare_model(self, model_name: Optional[str], **kwargs) -> Module:
        return Net(**kwargs)

    def epoch_train(self, net: Module, optimizer: Optimizer, train_loader: DataLoader) -> Tuple[Module, ResultDict]:
        running_loss = 0.0
        i = 0
        total = 0
        correct = 0
        criterion = BCEWithLogitsLoss()
        for inputs, labels in train_loader:
            inputs = inputs.to(self.device, dtype=torch.long)
            labels = labels.to(self.device, dtype=torch.float)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step(closure=None)
            running_loss += loss.item()
            total += labels.size(0)
            predicted = torch.where(outputs <= .5, torch.zeros_like(outputs), torch.ones_like(outputs))
            correct += (predicted == labels).sum().item()
            i += 1
        return net, dict(train_loss=running_loss / i, train_accuracy=correct / total)

    def epoch_validate(self, net: Module, test_loader: DataLoader, **kwargs) -> ResultDict:
        running_loss = 0.0
        i = 0
        total = 0
        correct = 0
        criterion = BCEWithLogitsLoss()
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device, dtype=torch.long)
                labels = labels.to(self.device, dtype=torch.float)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                predicted = torch.where(outputs <= .5, torch.zeros_like(outputs), torch.ones_like(outputs))
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                i += 1
        return dict(test_loss=running_loss / i, test_accuracy=correct / total)


class Net(Module):
    def __init__(self, in_dim: int, embedding_dim=50, hidden_size=50, num_layers=2, dropout=0.5) -> None:
        super().__init__()
        self.emb = Embedding(in_dim, embedding_dim, padding_idx=0)
        self.lstm = LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.linear = Linear(hidden_size * 2, 1)

    def forward(self, x):
        x = self.emb(x)
        _, (h, _) = self.lstm(x)
        x = torch.cat([h[0], h[-1]], dim=1)
        x = self.linear(x)
        return x.squeeze()
