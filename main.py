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
        # Momentum_Exiting=(SGD, dict(lr=lr, momentum=.9)),
        # AdaGrad_Existing=(Adagrad, dict(lr=lr)),
        # RMSProp_Existing=(RMSprop, dict(lr=lr)),
        # Adam_Existing=(Adam, dict(lr=lr, betas=(.9, .999), amsgrad=False)),
        # AMSGrad_Existing=(Adam, dict(lr=lr, betas=(.9, .999), amsgrad=True)),

        # Momentum_C1=(SGD, dict(lr=1e-1, momentum=.1e-1)),
        # Momentum_C2=(SGD, dict(lr=1e-2, momentum=.1e-2)),
        # Momentum_C3=(SGD, dict(lr=1e-3, momentum=.1e-3)),
        CGLikeMomentum_C1=(CGLikeMomentum, dict(alpha_type='C1', beta_type='C1', gamma_type='C1')),
        CGLikeMomentum_C2=(CGLikeMomentum, dict(alpha_type='C2', beta_type='C2', gamma_type='C2')),
        CGLikeMomentum_C3=(CGLikeMomentum, dict(alpha_type='C3', beta_type='C3', gamma_type='C3')),
        CGLikeMomentum_D1=(CGLikeMomentum, dict(alpha_type='D1', beta_type='D1', gamma_type='D1')),
        CGLikeMomentum_D2=(CGLikeMomentum, dict(alpha_type='D1', beta_type='D1', gamma_type='D2')),

        # Adam_C1=(Adam, dict(lr=lr, betas=(.9, .999), amsgrad=False)),
        # AMSGrad_C2=(Adam, dict(lr=lr, betas=(.9, .999), amsgrad=True)),
    )
    e = ExperimentCIFAR10(max_epoch=200, batch_size=128, model_name='ResNet44')
    e(optimizers, './result/cifar10')


if __name__ == '__main__':
    cifar10()
