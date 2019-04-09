#!/usr/bin/env python3
"""
Implementation of self-ensembling for visual domain adaptation
"""
import tensorflow as tf

from mnist import load_mnist
from usps import load_usps
from svhn import load_svhn
from load_data import load_dataset

if __name__ == "__main__":
    # Note: "It is worth noting that only the training sets of the small image
    # datasets were used during training; the test sets usedfor reporting scores
    # only." -- so, only use *_test for evaluation.
    usps_train, usps_test = load_dataset(*load_usps())
    mnist_train, mnist_test = load_dataset(*load_mnist())
    svhn_train, svhn_test = load_dataset(*load_svhn())
