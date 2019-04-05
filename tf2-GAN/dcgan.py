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

def load_mnist(buffer_size=6000, batch_size=32, prefetch_buffer_size=1):
    # Load
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    # Reshape to (# examples, 28, 28, 1) and set to float32
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    # Normalize the images to [-1, 1]
    train_images = (train_images - 127.5) / 127.5
    # Shuffle and batch
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).\
        repeat().shuffle(buffer_size).batch(batch_size).\
        prefetch(prefetch_buffer_size)

    return train_dataset

def make_generator_model(noise_dim, weight_decay=0.001):
    model = tf.keras.Sequential([
        layers.Dense(1024, use_bias=False, input_shape=(noise_dim,),
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay)),
        layers.BatchNormalization(),
        layers.Activation("relu"),

        layers.Dense(7*7*256, use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay)),
        layers.BatchNormalization(),
        layers.Activation("relu"),

        layers.Reshape((7, 7, 256)),
        layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay)),
        layers.BatchNormalization(),
        layers.Activation("relu"),

        layers.Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same', use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay)),
        layers.BatchNormalization(),
        layers.Activation("relu"),

        #layers.Conv2D(1, (4, 4), strides=(2, 2), padding='same', use_bias=False, activation='tanh'),
        layers.Conv2D(1, (4, 4), strides=(1, 1), padding='same', use_bias=False, activation='tanh'),
    ])
    assert model.output_shape == (None, 28, 28, 1)
    return model

def make_discriminator_model(weight_decay=0.001):
    return tf.keras.Sequential([
        layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same', input_shape=[28, 28, 1],
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
            bias_regularizer=tf.keras.regularizers.l2(weight_decay)),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
            bias_regularizer=tf.keras.regularizers.l2(weight_decay)),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(1024,
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
            bias_regularizer=tf.keras.regularizers.l2(weight_decay)),
        layers.BatchNormalization(),

        layers.Dense(1,
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
            bias_regularizer=tf.keras.regularizers.l2(weight_decay)),
    ])

def make_losses(wgan=False):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def discriminator_loss(real_output, fake_output):
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(fake_output):
        return cross_entropy(tf.ones_like(fake_output), fake_output)

    def wgan_discriminator_loss(real_output, fake_output):
        return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

    def wgan_generator_loss(fake_output):
        return -tf.reduce_mean(fake_output)

    if wgan:
        return wgan_discriminator_loss, wgan_generator_loss
    else:
        return discriminator_loss, generator_loss

# TODO use two separate networks sharing weights and use train_on_batch?
# and test_on_batch?
@tf.function
def train_step(images, generator, discriminator, generator_loss,
        discriminator_loss, generator_optimizer, discriminator_optimizer,
        noise_dim, wgan):
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

    # Clip discriminator weights if using WGAN loss
    if wgan:
        for w in discriminator.trainable_variables:
            if "kernel" in w.name or "bias" in w.name:
                w.assign(tf.clip_by_value(w, -0.01, 0.01))
                print("Will clip", w.name)

def train(dataset, generator, discriminator, generator_loss, discriminator_loss,
        generator_optimizer, discriminator_optimizer, wgan, noise_dim, seed,
        steps, checkpoint_manager):
    dataset_iter = iter(dataset)

    for step in range(steps):
        start = time.time()
        image_batch = next(dataset_iter)

        train_step(image_batch, generator, discriminator, generator_loss,
                discriminator_loss, generator_optimizer,
                discriminator_optimizer, noise_dim, wgan)

        if (step+1)%100 == 0:
            display.clear_output(wait=True)
            generate_and_save_images(generator, step+1, seed)

        if (step+1)%400 == 0:
            checkpoint_manager.save(checkpoint_number=step+1)

        print("Time for step {} is {} sec".format(step+1, time.time()-start))

def generate_and_save_images(model, step, test_input):
    predictions = model(test_input, training=False)

    plt.ion()
    fig = plt.figure(1, figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_step_{:04d}.png'.format(step))
    plt.show()
    fig.canvas.flush_events()

if __name__ == "__main__":
    # Params
    wgan = True
    steps = 1200
    noise_dim = 50
    num_examples_to_generate = 16
    seed = tf.random.normal([num_examples_to_generate, noise_dim])

    # Load training data
    train_dataset = load_mnist()

    # Create models, losses, optimizers
    generator = make_generator_model(noise_dim)
    discriminator = make_discriminator_model()
    discriminator_loss, generator_loss = make_losses(wgan)
    generator_optimizer = tf.keras.optimizers.Adam(1e-3, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)

    # Allow loading the model again after training
    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator, discriminator=discriminator)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint,
        directory="./training_checkpoints", max_to_keep=1)
    #checkpoint.restore(checkpoint_manager.latest_checkpoint)

    # Train
    train(train_dataset, generator, discriminator, generator_loss,
        discriminator_loss, generator_optimizer, discriminator_optimizer,
        wgan, noise_dim, seed, steps, checkpoint_manager)
