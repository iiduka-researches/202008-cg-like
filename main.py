from experiment.mnist import ExperimentMNIST
from experiment.cifar10 import ExperimentCIFAR10
from torch.optim import SGD, Adagrad, RMSprop, Adam
from optimizer.cg_like_momentum import CGLikeMomentum


def mnist() -> None:
    lr = 1e-3
    optimizers = dict(
        Momentum=(SGD, dict(lr=lr, momentum=.9)),
    )
    e = ExperimentMNIST(max_epoch=1, batch_size=32)
    e.execute(optimizers, './result/mnist')


def cifar10() -> None:
    lr = 1e-3
    optimizers = dict(
        Momentum_Exiting=(SGD, dict(lr=lr, momentum=.9)),
        AdaGrad_Existing=(Adagrad, dict(lr=lr)),
        RMSProp_Existing=(RMSprop, dict(lr=lr)),
        Adam_Existing=(Adam, dict(lr=lr, betas=(.9, .999), amsgrad=False)),
        AMSGrad_Existing=(Adam, dict(lr=lr, betas=(.9, .999), amsgrad=True)),

        Momentum_C1=(SGD, dict(lr=lr, momentum=.1e-1)),
        Momentum_C2=(SGD, dict(lr=lr, momentum=.1e-2)),
        Momentum_C3=(SGD, dict(lr=lr, momentum=.1e-3)),

        # Adam_C1=(Adam, dict(lr=lr, betas=(.9, .999), amsgrad=False)),
        # AMSGrad_C2=(Adam, dict(lr=lr, betas=(.9, .999), amsgrad=True)),
    )
    e = ExperimentCIFAR10(max_epoch=2, batch_size=128, model_name='ResNet44')
    e.execute(optimizers, './result/cifar10')


if __name__ == '__main__':
    cifar10()
