from experiment.cifar10 import ExperimentCIFAR10
from experiment.imdb import ExperimentIMDb
from torch.optim import SGD, Adagrad, RMSprop
from optimizer.adam import Adam
from optimizer.cg_like_momentum import CGLikeMomentum
from optimizer.cg_like_adam import CGLikeAdam


def _prepare_optimizers(lr: float):
    return dict(
        Momentum_Existing=(SGD, dict(lr=lr, momentum=.9)),
        AdaGrad_Existing=(Adagrad, dict(lr=lr)),
        RMSProp_Existing=(RMSprop, dict(lr=lr)),
        Adam_Existing=(Adam, dict(lr=lr, amsgrad=False)),
        AMSGrad_Existing=(Adam, dict(lr=lr, amsgrad=True)),

        Momentum_C1=(CGLikeMomentum, dict(alpha_type='C1', beta_type='C1', gamma_type='No')),
        Momentum_C2=(CGLikeMomentum, dict(alpha_type='C2', beta_type='C2', gamma_type='No')),
        Momentum_C3=(CGLikeMomentum, dict(alpha_type='C3', beta_type='C3', gamma_type='No')),
        Momentum_D1=(CGLikeMomentum, dict(alpha_type='D0', beta_type='D1', gamma_type='No')),
        CGLikeMomentum_C1=(CGLikeMomentum, dict(alpha_type='C1', beta_type='C1', gamma_type='C1')),
        CGLikeMomentum_C2=(CGLikeMomentum, dict(alpha_type='C2', beta_type='C2', gamma_type='C2')),
        CGLikeMomentum_C3=(CGLikeMomentum, dict(alpha_type='C3', beta_type='C3', gamma_type='C3')),
        CGLikeMomentum_D1=(CGLikeMomentum, dict(alpha_type='D0', beta_type='D1', gamma_type='D1')),
        CGLikeMomentum_D2=(CGLikeMomentum, dict(alpha_type='D0', beta_type='D1', gamma_type='D2')),

        Adam_C1=(CGLikeAdam, dict(alpha_type='C1', beta_type='C1', gamma_type='No', amsgrad=False)),
        Adam_C2=(CGLikeAdam, dict(alpha_type='C2', beta_type='C2', gamma_type='No', amsgrad=False)),
        Adam_C3=(CGLikeAdam, dict(alpha_type='C3', beta_type='C3', gamma_type='No', amsgrad=False)),
        Adam_D1=(CGLikeAdam, dict(alpha_type='D0', beta_type='D1', gamma_type='No', amsgrad=False)),
        CGLikeAdam_C1=(CGLikeAdam, dict(alpha_type='C1', beta_type='C1', gamma_type='C1', amsgrad=False)),
        CGLikeAdam_C2=(CGLikeAdam, dict(alpha_type='C2', beta_type='C2', gamma_type='C2', amsgrad=False)),
        CGLikeAdam_C3=(CGLikeAdam, dict(alpha_type='C3', beta_type='C3', gamma_type='C3', amsgrad=False)),
        CGLikeAdam_D1=(CGLikeAdam, dict(alpha_type='D0', beta_type='D1', gamma_type='D1', amsgrad=False)),
        CGLikeAdam_D2=(CGLikeAdam, dict(alpha_type='D0', beta_type='D1', gamma_type='D2', amsgrad=False)),

        AMSGrad_C1=(CGLikeAdam, dict(alpha_type='C1', beta_type='C1', gamma_type='No', amsgrad=True)),
        AMSGrad_C2=(CGLikeAdam, dict(alpha_type='C2', beta_type='C2', gamma_type='No', amsgrad=True)),
        AMSGrad_C3=(CGLikeAdam, dict(alpha_type='C3', beta_type='C3', gamma_type='No', amsgrad=True)),
        AMSGrad_D1=(CGLikeAdam, dict(alpha_type='D0', beta_type='D1', gamma_type='No', amsgrad=True)),
        CGLikeAMSGrad_C1=(CGLikeAdam, dict(alpha_type='C1', beta_type='C1', gamma_type='C1', amsgrad=True)),
        CGLikeAMSGrad_C2=(CGLikeAdam, dict(alpha_type='C2', beta_type='C2', gamma_type='C2', amsgrad=True)),
        CGLikeAMSGrad_C3=(CGLikeAdam, dict(alpha_type='C3', beta_type='C3', gamma_type='C3', amsgrad=True)),
        CGLikeAMSGrad_D1=(CGLikeAdam, dict(alpha_type='D0', beta_type='D1', gamma_type='D1', amsgrad=True)),
        CGLikeAMSGrad_D2=(CGLikeAdam, dict(alpha_type='D0', beta_type='D1', gamma_type='D2', amsgrad=True)),
    )


def prepare_optimizers(lr: float):
    return dict(
        Momentum_D1=(CGLikeMomentum, dict(alpha_type='D0', beta_type='D1', gamma_type='No')),
        CGLikeMomentum_D1=(CGLikeMomentum, dict(alpha_type='D0', beta_type='D1', gamma_type='D1')),
        CGLikeMomentum_D2=(CGLikeMomentum, dict(alpha_type='D0', beta_type='D1', gamma_type='D2')),

        Adam_D1=(CGLikeAdam, dict(alpha_type='D0', beta_type='D1', gamma_type='No', amsgrad=False)),
        CGLikeAdam_D1=(CGLikeAdam, dict(alpha_type='D0', beta_type='D1', gamma_type='D1', amsgrad=False)),
        CGLikeAdam_D2=(CGLikeAdam, dict(alpha_type='D0', beta_type='D1', gamma_type='D2', amsgrad=False)),

        AMSGrad_D1=(CGLikeAdam, dict(alpha_type='D0', beta_type='D1', gamma_type='No', amsgrad=True)),
        CGLikeAMSGrad_D1=(CGLikeAdam, dict(alpha_type='D0', beta_type='D1', gamma_type='D1', amsgrad=True)),
        CGLikeAMSGrad_D2=(CGLikeAdam, dict(alpha_type='D0', beta_type='D1', gamma_type='D2', amsgrad=True)),
    )


def imdb() -> None:
    lr = 1e-3
    optimizers = prepare_optimizers(lr)
    e = ExperimentIMDb(dataset_name='imdb', max_epoch=100, batch_size=32)
    e.execute(optimizers)


def cifar10() -> None:
    lr = 1e-3
    optimizers = prepare_optimizers(lr)
    e = ExperimentCIFAR10(dataset_name='cifar10', max_epoch=200, batch_size=128, model_name='ResNet44')
    e(optimizers)


if __name__ == '__main__':
    imdb()
    cifar10()
