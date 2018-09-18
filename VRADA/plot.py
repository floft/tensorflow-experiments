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


def plot_time_series(mu, sigma, num_samples=5, title=None, filename=None):
    """
    Using the mu and sigma given at each time step, generate sample time-series
    using these Gaussian parameters learned by the VRNN

    Input:
        mu, sigma -- each time step, learned in VRNN
        num_samples -- how many lines/curves you want to plot
        title, filename -- optional
    Output:
        plot of sample time-series
    """
    assert sigma.shape == mu.shape, "mu and sigma must have same shape"
    assert len(sigma.shape) == 2, "mu and sigma should be 2D, shape: [time_steps, 1]"

    mu = np.squeeze(mu)
    sigma = np.squeeze(sigma)
    length = mu.shape[0]

    # x axis is just 0, 1, 2, 3, ...
    x = np.arange(length)

    # y is values sampled from mu and simga
    y = sigma*np.random.normal(0, 1, (num_samples, length)) + mu

    plt.figure()
    for i in range(y.shape[0]):
        plt.plot(x, y[i,:])

    if title is not None:
        plt.title(title)

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0, transparent=True)

if __name__ == '__main__':
    # Test the plot time series function
    plot_time_series(
        np.array([[1,2,3,4,5,6,7,8,9,10]]),
        np.array([[1]*5+[3]*5]),
        title='Test')
    plt.show()