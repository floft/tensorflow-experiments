import numpy as np
import matplotlib.pyplot as plt

def plot_embedding(x, y, d, title=None, filename=None):
    """
    Plot an embedding X with the class label y colored by the domain d.
    
    From: https://github.com/pumpikano/tf-dann/blob/master/utils.py
    """
    x_min, x_max = np.min(x, 0), np.max(x, 0)
    x = (x - x_min) / (x_max - x_min)

    #colors = ["xkcd:orange", "xkcd:teal", "xkcd:darkgreen", "xkcd:orchid", "xkcd:blue", "xkcd:indigo"]

    colors = {
        (0, 0): 'xkcd:orange', # source 0
        (0, 1): 'xkcd:teal', # source 1
        (1, 0): 'xkcd:darkgreen', # target 0
        (1, 1): 'xkcd:indigo', # target 1
    }

    domain = {
        0: 'S',
        1: 'T',
    }

    # Plot colors numbers
    plt.figure(figsize=(10,10))
    ax = plt.subplot(111)
    for i in range(x.shape[0]):
        # plot colored number
        plt.text(x[i, 0], x[i, 1], domain[d[i]]+str(y[i]),
                 color=colors[(d[i], y[i])],
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])

    if title is not None:
        plt.title(title)

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0, transparent=True)