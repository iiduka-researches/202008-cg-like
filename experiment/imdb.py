from typing import Tuple, Optional

from torch.nn import Module
from torch.utils.data import DataLoader

from optimizer.optimizer import Optimizer
from .experiment import Experiment, ResultDict


class ExperimentIMDb(Experiment):
    def prepare_data_loader(self, batch_size: int, data_dir: str) -> Tuple[DataLoader, DataLoader]:
        pass

    def prepare_model(self, model_name: Optional[str]) -> Module:
        pass

    def epoch_train(self, net: Module, optimizer: Optimizer, train_loader: DataLoader) -> Tuple[Module, ResultDict]:
        pass

    def epoch_validate(self, net: Module, test_loader: DataLoader, **kwargs) -> ResultDict:
        pass
