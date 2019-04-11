"""
Load the SVHN (cropped) dataset

Usage:
    from svhn import load_svhn
    train_images, train_labels, test_images, test_labels = load_svhn()
"""
import scipy.io

from load_data import dataset_download


def download_svhn():
    """ Download the SVHN files from online """
    train_fp, test_fp = dataset_download(
        ["train_32x32.mat", "test_32x32.mat", "extra_32x32.mat"],
        "http://ufldl.stanford.edu/housenumbers/")
    return train_fp, test_fp


def process_svhn(data, labels):
    """ Reshape, convert to float, normalize to [-1,1] """
    data = data.reshape(data.shape[0], 32, 32, 3).astype("float32")
    data = (data - 127.5) / 127.5
    labels = labels.astype("float32")
    return data, labels


def load_svhn_file(filename):
    """ Load from .mat file """
    data = scipy.io.loadmat(filename)
    images = data["X"].transpose([3, 0, 1, 2])
    labels = data["y"].reshape([-1])
    return images, labels


def load_svhn():
    """
    Loads the SVHN dataset

    Returns: train_images, train_labels, test_images, test_labels
    """
    train_fp, test_fp = download_svhn()
    train_images, train_labels = load_svhn_file(train_fp)
    test_images, test_labels = load_svhn_file(test_fp)
    train_images, train_labels = process_svhn(train_images, train_labels)
    test_images, test_labels = process_svhn(test_images, test_labels)
    return train_images, train_labels, test_images, test_labels


if __name__ == "__main__":
    train_images, train_labels, test_images, test_labels = load_svhn()
