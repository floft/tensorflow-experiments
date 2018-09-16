"""
Generate some extremely simple time-series datasets that the RNNs should be
able to get 100% classification accuracy on

Positive-slope -- Identify if the slope of a line is positive (2) or negative (1)
Positive-sine -- Identify if a sine wave (2) or negative sine wave (1)

"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def linear(m, b, length=100, minvalue=0, maxvalue=100):
    x = np.arange(minvalue, maxvalue, (maxvalue-minvalue)/length).reshape(-1,1)
    y = m*x + b
    return x, y

def is_positive_slope(m):
    return m > 0

def generate_positive_slope_data(n, display=False):
    """ Positive or negative slope lines """
    m = np.random.normal(0, 1, (1,n))
    b = np.random.randint(-100, 100, (1,n))
    x, y = linear(m, b)
    labels = is_positive_slope(m)

    if display:
        plt.figure()
        for i in range(y.shape[1]):
            plt.plot(x, y[:,i])
        plt.show()

    df = pd.DataFrame(y.T)
    df.insert(0, 'class', pd.Series(np.squeeze(labels).astype(np.int32)+1, index=df.index))
    return df

def sine(m, b, length=100, minvalue=0, maxvalue=100):
    x = np.arange(minvalue, maxvalue, (maxvalue-minvalue)/length).reshape(-1,1)
    y = m*np.sin(x/10) + b # x/10 to make adjust horizontal scale
    return x, y

def generate_positive_sine_data(n, display=False):
    """ Sine wave multiplied by positive or negative number and offset some """
    m = np.random.normal(0, 10, (1,n))
    b = np.random.randint(-100, 100, (1,n))
    x, y = sine(m, b)
    labels = is_positive_slope(m)

    if display:
        plt.figure()
        for i in range(y.shape[1]):
            plt.plot(x, y[:,i])
        plt.show()

    df = pd.DataFrame(y.T)
    df.insert(0, 'class', pd.Series(np.squeeze(labels).astype(np.int32)+1, index=df.index))
    return df

if __name__ == '__main__':
    if not os.path.exists('trivial'):
        os.makedirs('trivial')

    generate_positive_slope_data(1000).to_csv('trivial/positive_slope_TRAIN', header=False, index=False)
    generate_positive_slope_data(100).to_csv('trivial/positive_slope_TEST', header=False, index=False)
    generate_positive_sine_data(1000).to_csv('trivial/positive_sine_TRAIN', header=False, index=False)
    generate_positive_sine_data(100).to_csv('trivial/positive_sine_TEST', header=False, index=False)
