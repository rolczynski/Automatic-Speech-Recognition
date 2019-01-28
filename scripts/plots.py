import os
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def lighten(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def save_confusion_matrix(confusions: np.ndarray, labels: list, directory: str, fig_size=(8, 8)):
    fig, ax = plt.subplots(figsize=fig_size)
    labels = list(labels)               # Copy the list (not change object inplace)
    labels[0], labels[-1] = '$', '_'    # To be visible on the plot
    plot_confusion_matrix(ax, confusions, labels)
    file_name = os.path.join(directory, 'confusion_matrix.svg')
    fig.savefig(file_name, format='svg', dpi=1200)


def plot_confusion_matrix(ax, confusions, labels):
    ax = sns.heatmap(confusions, square=True, xticklabels=labels, yticklabels=labels,
                     annot=False, linewidths=.1, ax=ax, fmt='d', cmap="YlGnBu")
    ax.set_xlabel('True label')
    ax.set_ylabel('Predicted label')
    ax.set_title('Confusion Matrix')
    return ax


def save_donut(inserts, deletes, confusion_matrix, directory):
    group_names = ['Insert', 'Delete', 'Substitute']
    tot_inserts = sum(inserts.values())
    tot_deletes = sum(deletes.values())
    tot_substitute = confusion_matrix.sum()
    group_size = [tot_inserts, tot_deletes, tot_substitute]

    subgroup_size = [inserts[' '], tot_inserts-inserts[' '], 
                     tot_deletes-deletes[' '], deletes[' '],
                     tot_substitute]
    
    # Create colors
    cmap = plt.get_cmap('YlGnBu')

    # First Ring (outside)
    fig, ax = plt.subplots()
    ax.axis('equal')
    mypie, _ = ax.pie(group_size, radius=1.3, labels=group_names, colors=[cmap(0.7), cmap(0.5), cmap(0.2)])
    plt.setp(mypie, width=0.3, edgecolor='white')

    # Second Ring (Inside)
    mypie2, _ = ax.pie(subgroup_size, radius=1.3 - 0.3, colors=[cmap(0), lighten(cmap(0.7)), lighten(cmap(0.5)), cmap(0), cmap(0)])
    plt.setp(mypie2, edgecolor='white')
    plt.margins(0, 0)

    file_name = os.path.join(directory, 'donut.svg')
    fig.savefig(file_name, format='svg', dpi=1200)
