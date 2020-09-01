import os
from typing import Set
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame, read_csv

color_dict = dict(
        Momentum='C2',
        Adam='C4',
        AMSGrad='C0',
        AdaGrad='C1',
        RMSProp='C3',
    )
marker_dict = dict(
    Existing='',
    C1='o',
    C2='x',
    C3='s',
    D1='^',
    D2='h',
)

x_label_dict = dict(
    epoch='Epoch',
    time='Elapsed time [s]',
)


def plot(dataset: str, model: str, title='', result_path='./result/result.csv', save_extension='png') -> None:
    name_col = 'optimizer'
    param_col = 'optimizer_parameters'
    epoch_col = 'epoch'
    time_col = 'time'

    # load result
    data = read_csv(result_path, encoding='utf-8')
    data.replace('Momentum_Exiting', 'Momentum_Existing', inplace=True)
    names = set(data[name_col])
    index_col = [name_col, epoch_col]
    data.set_index(index_col, inplace=True)

    for metric, y_label in (('train_loss', 'training loss'),
                            ('test_loss', 'test loss'),
                            ('train_accuracy', 'training error rate'),
                            ('test_accuracy', 'test error rate')):
        for x_axis in ('epoch', 'time'):
            _plot(data, optimizer_names=names, metric=metric, title=title, x_axis=x_axis, y_label=y_label,
                  time_col=time_col, save_name=f'{dataset}_{model}_{metric}_{x_axis}.{save_extension}')


def _plot(df: DataFrame, optimizer_names: Set[str], metric: str, time_col: str, title: str, y_label: str,
          save_name: str, width=12., height=8., x_axis='epoch', fig_dir='./figure') -> None:
    plt.figure(figsize=(width, height))
    for i, name in enumerate(optimizer_names):
        if x_axis == 'epoch':
            series = df.loc[name, metric]
            x = series.index
            y = series.values
        elif x_axis == 'time':
            d = df.loc[name, [metric, time_col]]
            x = np.cumsum(d[time_col].values)
            y = d[metric].values
        else:
            raise ValueError(f"x_axis should be 'epoch' or 'time' : x_axis = {x_axis}")

        if 'accuracy' in metric:
            y = 1. - y + 1e-8

        base_name, lr_type = name.split('_')
        color = color_dict[base_name.replace('CGLike', '')]
        linestyle = get_linestyle(base_name, lr_type)

        plt.plot(x, y, label=name, linestyle=linestyle, color=color,
                 marker=marker_dict[lr_type], markevery=5)

    if title:
        plt.title(title)

    ax = plt.gca()
    arrange_legend(ax, names=optimizer_names)

    plt.xlabel(x_label_dict[x_axis])
    plt.ylabel(ylabel=y_label)
    plt.grid(True, which='both')
    plt.yscale('log')

    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, save_name), dpi=300, bbox_inches='tight')
    plt.close()


def get_linestyle(name: str, lr_type: str) -> str:
    if lr_type == 'Existing':
        return 'dotted'
    elif 'CGLike' in name:
        return 'solid'
    else:
        return 'dashed'


def arrange_legend(ax, names: Set[str], ex_suffix='Existing', pp_base_names=None) -> None:
    if pp_base_names is None:
        pp_base_names = ('Momentum', 'Adam', 'AMSGrad', 'CGLikeMomentum', 'CGLikeAdam', 'CGLikeAMSGrad')

    handles, labels = ax.get_legend_handles_labels()
    handles_dict = dict(zip(labels, handles))

    # legends order
    existings = [n for n in names if ex_suffix in n]
    proposeds = [n for n in names if ex_suffix not in n]
    proposeds = [sorted([n for n in proposeds if n.split('_')[0] == bn]) for bn in pp_base_names]
    proposeds = [n for p in proposeds for n in p]
    labels = [*proposeds, *existings]
    handles = [handles_dict[l] for l in labels]
    ax.legend(handles=handles, labels=labels)


if __name__ == '__main__':
    plot(dataset='CIFAR-10', model='ResNet44')
