#!/usr/bin/env python3
"""
Plot some MNIST real data
"""
import tensorflow as tf
import matplotlib.pyplot as plt

def load_raw_mnist():
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    # Reshape to (# examples, 28, 28, 1) and set to float32
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    # Normalize the images to [-1, 1]
    train_images = (train_images - 127.5) / 127.5
    return train_images

def gen_save_real(real, num):
    fig = plt.figure(1, figsize=(4,4))

    for i in range(real.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(real[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
        if i == num-1:
            break

    plt.savefig('real_data.png')
    plt.show()

if __name__ == "__main__":
    num_examples_to_generate = 16
    gen_save_real(load_raw_mnist(), num_examples_to_generate)
