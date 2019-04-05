#!/usr/bin/env python3
"""
Hello World demo for TensorFlow 2.0
https://www.tensorflow.org/alpha/tutorials/quickstart/beginner
"""
import tensorflow as tf

def load_mnist():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return x_train, y_train, x_test, y_test

def train(x_train, y_train):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5)
    return model

def test(model, x_test, y_test):
    model.evaluate(x_test, y_test)

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_mnist()
    model = train(x_train, y_train)
    test(model, x_test, y_test)
