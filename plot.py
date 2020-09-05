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
    epoch='epoch',
    time='elapsed time [s]',
)


def plot(dataset: str, model: str, title='', result_path=None, save_extension='pdf') -> None:
    name_col = 'optimizer'
    param_col = 'optimizer_parameters'
    epoch_col = 'epoch'
    time_col = 'time'

    # load result
    if result_path is None:
        result_path = os.path.join('result', dataset, model, 'result.csv')
    data = read_csv(result_path, encoding='utf-8')
    # data.replace('Momentum_Exiting', 'Momentum_Existing', inplace=True)
    names = set(data[name_col])
    index_col = [name_col, epoch_col]
    data.set_index(index_col, inplace=True)

    constants = {n for n in names if n.split('_')[-1][0] == 'C' or n.split('_')[-1] == 'Existing'}
    diminishings = {n for n in names if n.split('_')[-1][0] == 'D'}
    for type_label, optimizer_names in (('constant', constants), ('diminishing', diminishings)):
        for metric, y_label in (('train_loss', 'training loss'),
                                ('test_loss', 'test loss'),
                                ('train_accuracy', 'training error rate'),
                                ('test_accuracy', 'test error rate')):
            for x_axis in ('epoch', ):  # ('epoch', 'time')
                _plot(data, optimizer_names=optimizer_names, metric=metric, title=title, x_axis=x_axis, y_label=y_label,
                      time_col=time_col, save_name=f'{dataset}_{model}_{type_label}_{metric}_{x_axis}.{save_extension}',
                      fig_dir=os.path.join('./figure', dataset, model))


def _plot(df: DataFrame, optimizer_names: Set[str], metric: str, time_col: str, title: str, y_label: str,
          save_name: str, width=12., height=9., x_axis='epoch', fig_dir='./figure') -> None:
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
    plt.savefig(os.path.join(fig_dir, save_name), dpi=300, bbox_inches='tight', pad_inches=.05)
    plt.close()


def get_linestyle(name: str, lr_type: str) -> str:
    if lr_type == 'Existing':
        return 'dotted'
    elif 'CGLike' in name:
        return 'solid'
    else:
        return 'dashed'


def arrange_legend(ax, names: Set[str], ex_suffix='Existing') -> None:
    ex_base_names = (f'Momentum_{ex_suffix}', f'AdaGrad_{ex_suffix}', f'RMSProp_{ex_suffix}', f'Adam_{ex_suffix}',
                     f'AMSGrad_{ex_suffix}')
    pp_base_names = ('Momentum', 'CGLikeMomentum', 'Adam', 'CGLikeAdam', 'AMSGrad', 'CGLikeAMSGrad')

    handles, labels = ax.get_legend_handles_labels()
    handles_dict = dict(zip(labels, handles))

    # legends order
    existings = [n for n in names if ex_suffix in n]
    existings = [n for n in ex_base_names if n in existings]

    proposeds = [n for n in names if ex_suffix not in n]
    proposeds = [sorted([n for n in proposeds if n.split('_')[0] == bn]) for bn in pp_base_names]
    proposeds = [n for p in proposeds for n in p]  # flatten
    labels = [*proposeds, *existings]
    handles = [handles_dict[l] for l in labels]
    labels = [label_format(l) for l in labels]
    ax.legend(handles=handles, labels=labels, bbox_to_anchor=(1.05, 1.0), loc='upper left')


def label_format(label: str) -> str:
    name, lr_type = label.split('_')
    if lr_type == 'Existing':
        return name
    else:
        if 'CGLike' in name:
            name = name.replace('CGLike', '')
            name = f'{name}CG'
        return f'{name}-{lr_type}'


if __name__ == '__main__':
    from sys import argv
    dataset, model = argv[1:3]
    plot(dataset=dataset, model=model)
