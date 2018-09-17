"""
Generate some extremely simple time-series datasets that the RNNs should be
able to get 100% classification accuracy on

Positive-slope -- Identify if the slope of a line is positive (2) or negative (1)
Positive-slope-noise -- same but with noise
Positive-sine -- Identify if a sine wave (2) or negative sine wave (1)
Positive-sine-noise -- same but with noise
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

def generate_positive_slope_data(n, display=False, add_noise=False,
        bmin=0, bmax=100, m_mu=0, m_std=1):
    """ Positive or negative slope lines """
    m = np.random.normal(m_mu, m_std, (1,n))
    b = np.random.randint(bmin, bmax, (1,n))
    x, y = linear(m, b)
    labels = is_positive_slope(m)

    if add_noise:
        noise = np.random.normal(0, 10, (y.shape[0],n))
        y += noise

    if display:
        plt.figure()
        for i in range(y.shape[1]):
            plt.plot(x, y[:,i])
        plt.show()

    df = pd.DataFrame(y.T)
    df.insert(0, 'class', pd.Series(np.squeeze(labels).astype(np.int32)+1, index=df.index))
    return df

def sine(m, b, length=100, minvalue=0, maxvalue=100, horiz_scale=10):
    x = np.arange(minvalue, maxvalue, (maxvalue-minvalue)/length).reshape(-1,1)
    y = m*np.sin(1.0*x/horiz_scale) + b
    return x, y

def generate_positive_sine_data(n, display=False, add_noise=False,
        bmin=0, bmax=100, m_mu=0, m_std=10, horiz_scale=10):
    """ Sine wave multiplied by positive or negative number and offset some """
    m = np.random.normal(m_mu, m_std, (1,n))
    b = np.random.randint(bmin, bmax, (1,n))
    x, y = sine(m, b, horiz_scale=horiz_scale)
    labels = is_positive_slope(m)

    if add_noise:
        noise = np.random.normal(0, 1, (y.shape[0],n))
        y += noise

    if display:
        plt.figure()
        for i in range(y.shape[1]):
            plt.plot(x, y[:,i])
        plt.show()

    df = pd.DataFrame(y.T)
    df.insert(0, 'class', pd.Series(np.squeeze(labels).astype(np.int32)+1, index=df.index))
    return df

def save_data(func, fn):
    """ Use func to create examples that are saved to fn_TRAIN and fn_TEST """
    func(1000).to_csv('trivial/'+fn+'_TRAIN', header=False, index=False)
    func(100).to_csv('trivial/'+fn+'_TEST', header=False, index=False)

if __name__ == '__main__':
    # For reproducability
    np.random.seed(0)

    if not os.path.exists('trivial'):
        os.makedirs('trivial')

    # No noise
    save_data(generate_positive_slope_data, 'positive_slope')
    save_data(generate_positive_sine_data, 'positive_sine')

    # Noisy
    save_data(lambda x: generate_positive_slope_data(x, add_noise=True), 'positive_slope_noise')
    save_data(lambda x: generate_positive_sine_data(x, add_noise=True), 'positive_sine_noise')

    # No noise - but different y-intercept
    save_data(lambda x: generate_positive_slope_data(x, bmin=-200, bmax=-100), 'positive_slope_low')
    save_data(lambda x: generate_positive_sine_data(x, bmin=-200, bmax=-100), 'positive_sine_low')
