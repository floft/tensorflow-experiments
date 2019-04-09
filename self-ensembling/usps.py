"""
Load the USPS dataset

Usage:
    from usps import load_usps
    train_images, train_labels, test_images, test_labels = load_usps()
"""
import gzip
import numpy as np
import tensorflow as tf

from load_data import dataset_download

def download_usps():
    """ Download the USPS files from online """
    train_fp, test_fp = dataset_download(
        ["zip.train.gz", "zip.test.gz", "zip.info.txt"],
        "https://web.stanford.edu/~hastie/ElemStatLearn/datasets/")
    return train_fp, test_fp

def floats_to_list(line):
    """ Return list of floating-point numbers from space-separated string """
    items = line.split(" ")
    floats = []

    for item in items:
        try:
            f = float(item)
        except ValueError:
            raise ValueError("floats_to_list() requires space-separated floats")
        floats.append(f)

    return floats

def load_usps_file(filename):
    """ See zip.info.txt for the file format, which is gzipped """
    images = []
    labels = []

    with gzip.open(filename, "rb") as f:
        for line in f:
            label, *pixels = floats_to_list(line.strip().decode("utf-8"))
            assert len(pixels) == 256, "Should be 256 pixels"
            assert label >= 0 and label <= 9, "Label should be in [0,9]"
            images.append(pixels)
            labels.append(label)

    images = np.vstack(images)
    labels = np.hstack(labels)

    return images, labels

def process_usps(data, labels):
    """ Reshape (already normalized to [-1,1], should already be float) """
    data = data.reshape(data.shape[0], 16, 16, 1).astype("float32")
    labels = labels.astype("float32")
    return data, labels

def load_usps():
    """
    Loads the USPS dataset

    Returns: train_images, train_labels, test_images, test_labels
    """
    train_fp, test_fp = download_usps()
    train_images, train_labels = load_usps_file(train_fp)
    test_images, test_labels = load_usps_file(test_fp)
    train_images, train_labels = process_usps(train_images, train_labels)
    test_images, test_labels = process_usps(test_images, test_labels)
    return train_images, train_labels, test_images, test_labels

if __name__ == "__main__":
    train_images, train_labels, test_images, test_labels = load_usps()
