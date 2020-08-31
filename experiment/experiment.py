from abc import ABCMeta, abstractmethod
from datetime import datetime
import os
import random
from tempfile import TemporaryDirectory
from time import time
from typing import Any, Dict, Optional, Sequence, Tuple
import numpy as np
from pandas import concat, DataFrame, read_csv
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm
from optimizer.optimizer import Optimizer
from utils.gmail.transmitter import GMailTransmitter, ACCOUNT_JSON
from utils.line.notify import notify, notify_error

ParamDict = Dict[str, Any]
OptimDict = Dict[str, Tuple[Any, ParamDict]]
ResultDict = Dict[str, float]
Result = Dict[str, Sequence[float]]


class Experiment(metaclass=ABCMeta):
    def __init__(self, batch_size: int, max_epoch: int, model_name='model', data_dir='./dataset/data/') -> None:
        self.model_name = model_name
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.device = select_device()

    def __call__(self, *args, batch_size: int, **kwargs) -> None:
        self.execute(*args, **kwargs)

    @abstractmethod
    def prepare_data_loader(self, batch_size: int, data_dir: str) -> Tuple[DataLoader, DataLoader]:
        raise NotImplementedError

    @abstractmethod
    def prepare_model(self, model_name: Optional[str]) -> Module:
        raise NotImplementedError

    @abstractmethod
    def epoch_train(self, net: Module, optimizer: Optimizer, train_loader: DataLoader) -> Tuple[Module, ResultDict]:
        raise NotImplementedError

    @abstractmethod
    def epoch_validate(self, net: Module, test_loader: DataLoader, **kwargs) -> ResultDict:
        raise NotImplementedError

    def train(self, net: Module, optimizer: Optimizer, train_loader: DataLoader,
              test_loader: DataLoader) -> Tuple[Module, Result]:

        results = []
        for epoch in tqdm(range(self.max_epoch)):
            start = time()
            net, train_result = self.epoch_train(net, optimizer=optimizer, train_loader=train_loader)
            validate_result = self.epoch_validate(net, test_loader=test_loader)
            result = arrange_result_as_dict(t=time() - start, train=train_result, validate=validate_result)
            results.append(result)
            if epoch % 10 == 0:
                notify(str(result))
        return net, concat_dicts(results)

    @notify_error
    def execute(self, optimizers: OptimDict, result_dir: str) -> None:
        model_dir = os.path.join(result_dir, self.model_name)
        train_loader, test_loader = self.prepare_data_loader(batch_size=self.batch_size, data_dir=self.data_dir)
        for name, (optimizer, optimizer_kw) in optimizers.items():
            fix_seed()
            net = self.prepare_model(self.model_name)
            net.to(self.device)
            _, result = self.train(net=net, optimizer=optimizer(net.parameters(), **optimizer_kw),
                                   train_loader=train_loader, test_loader=test_loader)
            result_to_csv(result, name=name, optimizer_kw=optimizer_kw,
                          result_dir=model_dir)



def select_device() -> str:
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    print(f'Using {device} ...')
    return device


def arrange_result_as_dict(t: float, train: Dict[str, float], validate: Dict[str, float]) -> Dict[str, float]:
    train = {k if 'train' in k else f'train_{k}': v for k, v in train.items()}
    validate = {k if 'test' in k else f'test_{k}': v for k, v in validate.items()}
    return dict(time=t, **train, **validate)


def concat_dicts(results: Sequence[ResultDict]) -> Result:
    keys = results[0].keys()
    return {k: [r[k] for r in results] for k in keys}


def fix_seed(seed=0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def result_to_csv(r: Result, name: str, optimizer_kw: ParamDict, result_dir: str) -> None:
    df = DataFrame(r)
    df['optimizer'] = name
    df['optimizer_parameters'] = str(optimizer_kw)
    df['epoch'] = np.arange(1, df.shape[0] + 1)
    df.set_index(['optimizer', 'optimizer_parameters', 'epoch'], drop=True, inplace=True)

    os.makedirs(result_dir, exist_ok=True)
    path = os.path.join(result_dir, result_format(name))
    df.to_csv(path, encoding='utf-8')


def result_format(name: str, extension='csv') -> str:
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f'{name}_{ts}.{extension}'


def send_csv(path: str, body: str, to=None) -> None:
    if os.path.isfile(ACCOUNT_JSON):
        transmitter = GMailTransmitter()
        subject = f'[実験結果] {os.path.basename(path)}'
        if to is None:
            to = transmitter.sender_account
        transmitter.send(subject=subject, to=to, body=body, file_path=path, extension=os.path.splitext(path)[-1])


def send_collected_csv(result_dir: str) -> None:
    paths = (os.path.join(result_dir, f) for f in os.listdir(result_dir) if f[-4:] == '.csv')
    df = concat([read_csv(path, encoding='utf-8') for path in paths])
    with TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, 'result.csv')
        df.to_csv(path, index=False, encoding='utf-8')
        send_csv(path, body='')

