"""
Load the MNIST dataset

Usage:
    from mnist import load_mnist
    train_images, train_labels, test_images, test_labels = load_mnist()
"""
import tensorflow as tf

def process_mnist(data):
    """ Reshape, convert to float, normalize to [-1,1] """
    data = data.reshape(data.shape[0], 28, 28, 1).astype("float32")
    data = (data - 127.5) / 127.5
    return data

def load_mnist():
    """
    Load the MNIST dataset

    Returns: train_images, train_labels, test_images, test_labels
    """
    (train_images, train_labels), (test_images, test_labels) = \
        tf.keras.datasets.mnist.load_data()
    train_images = process_mnist(train_images)
    test_images = process_mnist(test_images)
    return train_images, train_labels, test_images, test_labels

if __name__ == "__main__":
    train_images, train_labels, test_images, test_labels = load_mnist()
