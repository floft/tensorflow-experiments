#!/usr/bin/env python3
"""
GAN in TensorFlow 2.0

Based on:
https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/r2/tutorials/generative/dcgan.ipynb
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/eager/python/examples/generative_examples/dcgan.ipynb
https://towardsdatascience.com/gan-by-example-using-keras-on-tensorflow-backend-1a6d515a60d0
https://pgaleone.eu/tensorflow/gan/2018/11/04/tensorflow-2-models-migration-and-new-design/
"""
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from IPython import display
from tensorflow.keras import layers

def load_mnist(buffer_size=60000, batch_size=256, prefetch_buffer_size=1):
    # Load
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    # Reshape to (# examples, 28, 28, 1) and set to float32
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    # Normalize the images to [-1, 1]
    train_images = (train_images - 127.5) / 127.5
    # Shuffle and batch
    return tf.data.Dataset.from_tensor_slices(train_images).\
        shuffle(buffer_size).batch(batch_size).prefetch(prefetch_buffer_size)

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model

def make_discriminator_model():
    return tf.keras.Sequential([
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(1),
    ])

def make_losses():
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def discriminator_loss(real_output, fake_output):
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(fake_output):
        return cross_entropy(tf.ones_like(fake_output), fake_output)

    return discriminator_loss, generator_loss

@tf.function
def train_step(images, generator, discriminator, generator_loss,
        discriminator_loss, generator_optimizer, discriminator_optimizer,
        noise_dim):
    batch_size = images.shape[0]
    noise = tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, generator, discriminator, generator_loss, discriminator_loss,
        generator_optimizer, discriminator_optimizer, noise_dim, seed, epochs,
        checkpoint_manager):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch, generator, discriminator, generator_loss,
                discriminator_loss, generator_optimizer,
                discriminator_optimizer, noise_dim)

        display.clear_output(wait=True)
        generate_and_save_images(generator, epoch+1, seed)

        # Save the model every 5 epochs
        if (epoch+1)%5 == 0:
            checkpoint_manager.save(checkpoint_number=epoch+1)

        print('Time for epoch {} is {} sec'.format(epoch+1, time.time()-start))

def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)

    plt.ion()
    fig = plt.figure(1, figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()
    fig.canvas.flush_events()

if __name__ == "__main__":
    # Load training data
    train_dataset = load_mnist()

    # Create models, losses, optimizers
    generator = make_generator_model()
    discriminator = make_discriminator_model()
    discriminator_loss, generator_loss = make_losses()
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    # Allow loading the model again after training
    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator, discriminator=discriminator)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint,
        directory="./training_checkpoints", max_to_keep=5)
    #checkpoint.restore(checkpoint_manager.latest_checkpoint)

    # Train
    epochs = 150
    noise_dim = 100
    num_examples_to_generate = 16
    seed = tf.random.normal([num_examples_to_generate, noise_dim])

    train(train_dataset, generator, discriminator, generator_loss,
        discriminator_loss, generator_optimizer, discriminator_optimizer,
        noise_dim, seed, epochs, checkpoint_manager)
