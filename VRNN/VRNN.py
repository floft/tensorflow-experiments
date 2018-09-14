#!/usr/bin/env python
#
# VRNN
#
import os
import time
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

# Due to this being run on Kamiak, that doesn't have _tkinter, we have to set a
# different backend otherwise it'll error
# https://stackoverflow.com/a/40931739/2698494
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt

# Not-so-pretty code to feed data to TensorFlow.
class IteratorInitializerHook(tf.train.SessionRunHook):
    """Hook to initialise data iterator after Session is created.
    https://medium.com/onfido-tech/higher-level-apis-in-tensorflow-67bfb602e6c0"""
    def __init__(self):
        super(IteratorInitializerHook, self).__init__()
        self.iter_init_func = None

    def after_create_session(self, sess, coord):
        """Initialize the iterator after the session has been created."""
        self.iter_init_func(sess)

def _get_input_fn(features, labels, batch_size, evaluation=False, buffer_size=5000):
    iter_init_hook = IteratorInitializerHook()

    def input_fn():
        # Input images using placeholders to reduce memory usage
        features_placeholder = tf.placeholder(features.dtype, features.shape)
        labels_placeholder = tf.placeholder(labels.dtype, labels.shape)
        dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))

        if evaluation:
            dataset = dataset.batch(batch_size)
        else:
            dataset = dataset.repeat().shuffle(buffer_size).batch(batch_size)

        iterator = dataset.make_initializable_iterator()
        next_data_batch, next_label_batch = iterator.get_next()

        # Need to initialize iterator after creating a session in the estimator
        iter_init_hook.iter_init_func = lambda sess: sess.run(iterator.initializer,
                feed_dict={features_placeholder: features, labels_placeholder: labels})

        return next_data_batch, next_label_batch
    return input_fn, iter_init_hook

# Load a time-series dataset. This is set up to load data in the format of the
# UCR time-series datasets (http://www.cs.ucr.edu/~eamonn/time_series_data/).
# Or, see the generate_trivial_datasets.py for a trivial dataset.
#
# Also runs through one_hot
def load_data(filename):
    """
    Load CSV files in UCR time-series data format

    Returns:
        data - numpy array with data of shape (num_examples, num_features)
        labels - numpy array with labels of shape: (num_examples, 1)
    """
    df = pd.read_csv(filename, header=None)
    df_data = df.drop(0, axis=1).values.astype(np.float32)
    df_labels = df.loc[:, df.columns == 0].values.astype(np.uint8)
    return df_data, df_labels

def one_hot(x, y, num_classes):
    """ Correct dimensions, type, and one-hot encode """
    x = np.expand_dims(x, axis=2).astype(np.float32)
    y = np.eye(num_classes)[np.squeeze(y).astype(np.uint8) - 1] # one-hot encode
    return x, y

# Implementing VRNN. Based on:
#  - https://github.com/phreeza/tensorflow-vrnn/blob/master/model_vrnn.py
#  - https://github.com/kimkilho/tensorflow-vrnn/blob/master/cell.py
#  - https://github.com/kimkilho/tensorflow-vrnn/blob/master/main.py
class VRNNCell(tf.keras.layers.Layer):
    """
    VRNN cell

    Usage:
        cell = VRNNCell(x_dim, h_dim, z_dim, batch_size)
        net = tf.keras.layers.RNN(cell)(net)

    Note: probably want to use vrnn_loss though, otherwise this performs terrible.
    """
    def __init__(self, x_dim, h_dim, z_dim, **kwargs):
        # Dimensions of x input, hidden layers, latent variable (z)
        self.n_x = x_dim
        self.n_h = h_dim
        self.n_z = z_dim

        # Dimensions of phi(z)
        self.n_x_1 = x_dim
        self.n_z_1 = z_dim

        # Dimensions of encoder, decoder, and prior
        self.n_enc_hidden = z_dim
        self.n_dec_hidden = x_dim
        self.n_prior_hidden = z_dim

        # What cell we're going to use internally for the RNN
        self.cell = tf.keras.layers.LSTMCell(self.n_h,
             input_shape=(None, self.n_dec_hidden+self.n_z_1))

        super(VRNNCell, self).__init__(**kwargs)

    @property
    def state_size(self):
        # Note: first two are the state of the LSTM
        return (self.n_h, self.n_h,
                self.n_z, self.n_z,
                self.n_x, self.n_x,
                self.n_z, self.n_z)

    @property
    def output_size(self):
        return self.n_h

    def build(self, input_shape):
        # Input: previous hidden state
        self.prior_h = self.add_weight(
            shape=(self.n_h, self.n_prior_hidden), initializer='glorot_uniform', name='prior_hidden')
        self.prior_mu = self.add_weight(
            shape=(self.n_prior_hidden, self.n_z), initializer='glorot_uniform', name='prior_mu')
        self.prior_sigma = self.add_weight(
            shape=(self.n_prior_hidden, self.n_z), initializer='glorot_uniform', name='prior_sigma')

        self.prior_h_b = self.add_weight(
            shape=(self.n_prior_hidden,), initializer='constant', name='prior_hidden_b')
        self.prior_mu_b = self.add_weight(
            shape=(self.n_z,), initializer='constant', name='prior_mu_b')
        self.prior_sigma_b = self.add_weight(
            shape=(self.n_z,), initializer='constant', name='prior_sigma_b')

        # Input: x
        self.x_1 = self.add_weight(
            shape=(self.n_x, self.n_x_1), initializer='glorot_uniform', name='phi_x')

        self.x_1_b = self.add_weight(
            shape=(self.n_x_1,), initializer='constant', name='phi_x_b')

        # Input: x and previous hidden state
        self.encoder_h = self.add_weight(
            shape=(self.n_x_1+self.n_h, self.n_enc_hidden), initializer='glorot_uniform', name='encoder_hidden')
        self.encoder_mu = self.add_weight(
            shape=(self.n_enc_hidden, self.n_z), initializer='glorot_uniform', name='encoder_mu')
        self.encoder_sigma = self.add_weight(
            shape=(self.n_enc_hidden, self.n_z), initializer='glorot_uniform', name='encoder_sigma')

        self.encoder_h_b = self.add_weight(
            shape=(self.n_enc_hidden,), initializer='constant', name='encoder_hidden_b')
        self.encoder_mu_b = self.add_weight(
            shape=(self.n_z,), initializer='constant', name='encoder_mu_b')
        self.encoder_sigma_b = self.add_weight(
            shape=(self.n_z,), initializer='constant', name='encoder_sigma_b')

        # Input: z = enc_sigma*eps + enc_mu -- i.e. reparameterization trick
        self.z_1 = self.add_weight(
            shape=(self.n_z, self.n_z_1), initializer='glorot_uniform', name='phi_z')

        self.z_1_b = self.add_weight(
            shape=(self.n_z_1,), initializer='constant', name='phi_z_b')

        # Input: latent variable (z) and previous hidden state
        self.decoder_h = self.add_weight(
            shape=(self.n_z+self.n_h, self.n_dec_hidden), initializer='glorot_uniform', name='decoder_hidden')
        self.decoder_mu = self.add_weight(
            shape=(self.n_dec_hidden, self.n_x), initializer='glorot_uniform', name='decoder_mu')
        self.decoder_sigma = self.add_weight(
            shape=(self.n_dec_hidden, self.n_x), initializer='glorot_uniform', name='decoder_sigma')

        self.decoder_h_b = self.add_weight(
            shape=(self.n_dec_hidden,), initializer='constant', name='decoder_hidden_b')
        self.decoder_mu_b = self.add_weight(
            shape=(self.n_x,), initializer='constant', name='decoder_mu_b')
        self.decoder_sigma_b = self.add_weight(
            shape=(self.n_x,), initializer='constant', name='decoder_sigma_b')

        super(VRNNCell, self).build(input_shape)

    def call(self, inputs, states, training=None):
        # Get relevant states
        h = states[0]
        c = states[1] # only passed to the LSTM

        # Determine if training
        if training is None:
            training = K.learning_phase()

        # Input: previous hidden state (h)
        prior_h = K.relu(K.dot(h, self.prior_h) + self.prior_h_b)
        #prior_h = K.dot(h, self.prior_h) + self.prior_h_b # Linear
        prior_mu = K.dot(prior_h, self.prior_mu) + self.prior_mu_b
        prior_sigma = K.softplus(K.dot(prior_h, self.prior_sigma) + self.prior_sigma_b) # >= 0

        # Input: x
        x_1 = K.relu(K.dot(inputs, self.x_1) + self.x_1_b) # >= 0

        # Input: x and previous hidden state
        encoder_input = K.concatenate((x_1, h), 1)
        encoder_h = K.relu(K.dot(encoder_input, self.encoder_h) + self.encoder_h_b)
        #encoder_h = K.dot(encoder_input, self.encoder_h) + self.encoder_h_b # Linear
        encoder_mu = K.dot(encoder_h, self.encoder_mu) + self.encoder_mu_b
        encoder_sigma = K.softplus(K.dot(encoder_h, self.encoder_sigma) + self.encoder_sigma_b)

        # If not training, just copy the prior?
        # https://lirnli.wordpress.com/2017/09/27/variational-recurrent-neural-network-vrnn-with-pytorch/
        #encoder_mu = tf.cond(training, lambda: encoder_mu,    lambda: prior_mu)
        #encoder_mu = tf.cond(training, lambda: encoder_sigma, lambda: prior_sigma)

        # Input: z = enc_sigma*eps + enc_mu -- i.e. reparameterization trick
        batch_size = K.shape(inputs)[0] # https://github.com/tensorflow/tensorflow/issues/373
        eps = K.random_normal((batch_size, self.n_z), dtype=tf.float32)
        z = encoder_sigma*eps + encoder_mu
        z_1 = K.relu(K.dot(z, self.z_1) + self.z_1_b)

        # Input: latent variable (z) and previous hidden state
        decoder_input = K.concatenate((z_1, h), 1)
        decoder_h = K.relu(K.dot(decoder_input, self.decoder_h) + self.decoder_h_b)
        #decoder_h = K.dot(decoder_input, self.decoder_h) + self.decoder_h_b # Linear
        decoder_mu = K.dot(decoder_h, self.decoder_mu) + self.decoder_mu_b
        decoder_sigma = K.softplus(K.dot(decoder_h, self.decoder_sigma) + self.decoder_sigma_b)

        # Pass to cell (e.g. LSTM). Note that the LSTM has both "h" and "c" that are combined
        # into the same next state vector. We'll combine them together to pass in and split them
        # back out after the LSTM returns the next state.
        rnn_cell_input = K.concatenate((x_1, z_1), 1)
        output, (h_next, c_next) = self.cell(rnn_cell_input, [h, c])

        # VRNN state
        next_state = (
            h_next,
            c_next,
            encoder_mu,
            encoder_sigma,
            decoder_mu,
            decoder_sigma,
            prior_mu,
            prior_sigma,
        )

        # TODO save the encoder mu/sigma for the entire history of time steps for the loss?
        # TODO maybe save x and xhat to minimize difference?

        return output, next_state

    def get_config(self):
        """ Save cell config to the model file """
        config = {
            'x_dim': self.n_x,
            'h_dim': self.n_h,
            'z_dim': self.n_z,
        }
        base_config = super(VRNNCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# Then implementing the VRNN loss.
def vrnn_loss(
    encoder_mu, encoder_sigma,
    decoder_mu, decoder_sigma,
    prior_mu, prior_sigma, x):
    """
    Returns function to compute VRNN loss

    First 6 arguments are from the state of the VRNN. Also pass in x
    since we're trying to make the decoder generate x.

    Usage:
        cell = VRNNCell(x_dim, h_dim, z_dim, batch_size)
        vrnn_layer = tf.keras.layers.RNN(cell, return_state=True)
        net, _, _, *vrnn_state = vrnn_layer(x)
        y = ...
        loss = vrnn_loss(*vrnn_state, x)
    """

    def loss_function(y_true, y_pred):
        eps = 1e-9

        # KL divergence
        # https://stats.stackexchange.com/q/7440
        # https://github.com/kimkilho/tensorflow-vrnn/blob/master/main.py
        kl_loss = 0.5*tf.reduce_mean(
                2*tf.log(tf.maximum(eps, prior_sigma)) - 2*tf.log(tf.maximum(eps, encoder_sigma))
                + (tf.square(encoder_sigma) + tf.square(encoder_mu - prior_mu))
                    / tf.maximum(eps, tf.square(prior_sigma))
                - 1,
            axis=1)

        # Reshape [batch_size,time_steps,num_features] -> [batch_size*time_steps,num_features]
        # so that (decoder_mu - x) will broadcast correctly
        x_reshape = tf.transpose(x, [0, 2, 1])

        # Negative log likelihood:
        # https://papers.nips.cc/paper/7219-simple-and-scalable-predictive-uncertainty-estimation-using-deep-ensembles.pdf
        # https://fairyonice.github.io/Create-a-neural-net-with-a-negative-log-likelihood-as-a-loss.html
        likelihood_loss = 0.5*tf.reduce_mean(
            tf.square(decoder_mu - x_reshape) / tf.maximum(eps, tf.square(decoder_sigma))
            + tf.log(tf.maximum(eps, tf.square(decoder_sigma)))
        )

        # We'd also like to classify well
        categorical_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)

        return tf.reduce_mean(kl_loss + likelihood_loss + categorical_loss)
    return loss_function

def build_lstm(x, keep_prob, lstm_sizes):
    """
    Generate multi-layer LSTM
    https://github.com/GarrettHoffman/lstm-oreilly
    """
    lstms = [tf.contrib.rnn.BasicLSTMCell(sz) for sz in lstm_sizes]
    drops = [tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob) for lstm in lstms]
    cell = tf.contrib.rnn.MultiRNNCell(drops)
    batch_size = tf.shape(x)[0]
    initial_state = cell.zero_state(batch_size, tf.float32)

    lstm_outputs, final_state = tf.nn.dynamic_rnn(cell, x, initial_state=initial_state)

    return initial_state, lstm_outputs, cell, final_state

def lstm_model(x, keep_prob, num_classes):
    """ Create model """
    # Build the LSTM
    initial_state, lstm_outputs, lstm_cell, final_state = \
            build_lstm(x, keep_prob, [128, 128])

    # Pass result to fully connected then softmax to get class prediction
    yhat = tf.contrib.layers.fully_connected(
            lstm_outputs[:, -1], num_classes, activation_fn=tf.nn.softmax)

    return yhat

def train(data_info, features, labels,
        features_test, labels_test,
        batch_size=64,
        num_steps=1000,
        num_eval=20,
        model_dir="models",
        log_dir="logs",
        model_save_steps=100,
        log_save_steps=1):

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Data stats
    time_steps, num_features, num_classes = data_info

    # Input training data and noise
    input_fn, input_hook = _get_input_fn(features, labels, batch_size)
    next_data_batch, next_labels_batch = input_fn()

    # Load all the test data in one batch (we'll assume test set is small for now)
    eval_input_fn, eval_input_hook = _get_input_fn(
            features_test, labels_test, features_test.shape[0], evaluation=True)
    next_data_batch_test, next_labels_batch_test = eval_input_fn()

    # Inputs
    keep_prob = tf.placeholder_with_default(1.0, shape=())
    x = tf.placeholder(tf.float32, [None, time_steps, num_features], name='x')
    y = tf.placeholder(tf.float32, [None, num_classes], name='y')

    # Model
    yhat = lstm_model(x, keep_prob, num_classes)

    # Loss
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=yhat)

    # Accuracy -- https://stackoverflow.com/a/42608050/2698494
    accuracy = tf.reduce_mean(tf.cast(
        tf.equal(tf.argmax(y, axis=-1), tf.argmax(yhat, axis=-1)),
    tf.float32))

    # Optimizer
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    # Summaries
    training_summaries = tf.summary.merge([
        tf.summary.scalar("loss", tf.reduce_mean(loss)),
        tf.summary.scalar("training_accuracy", accuracy)
    ])
    evaluation_summaries = tf.summary.merge([
        tf.summary.scalar("validation_accuracy", accuracy)
    ])

    # Allow restoring global_step from past run
    global_step = tf.Variable(0, name="global_step", trainable=False)
    inc_global_step = tf.assign_add(global_step, 1, name='incr_global_step')

    # Keep track of state and summaries
    saver = tf.train.Saver(max_to_keep=num_steps)
    saver_hook = tf.train.CheckpointSaverHook(model_dir,
            save_steps=model_save_steps, saver=saver)
    writer = tf.summary.FileWriter(log_dir)

    # Start training
    with tf.train.SingularMonitoredSession(checkpoint_dir=model_dir,
            hooks=[input_hook, eval_input_hook, saver_hook]) as sess:

        # Get evaluation batch once
        eval_data, eval_labels = sess.run([next_data_batch_test, next_labels_batch_test])

        for i in range(sess.run(global_step),num_steps+1):
            if i == 0:
                writer.add_graph(sess.graph)

            t = time.time()
            data_batch, labels_batch = sess.run([next_data_batch, next_labels_batch])
            _, step = sess.run([optimizer, inc_global_step],
                    feed_dict={x: data_batch, y: labels_batch, keep_prob: 0.8})
            t = time.time() - t

            if i%log_save_steps == 0:
                # Log the step time
                summ = tf.Summary(value=[tf.Summary.Value(tag="step_time", simple_value=t)])
                writer.add_summary(summ, step)

                # Log summaries run on the training data
                summ = sess.run(training_summaries,
                        feed_dict={x: data_batch, y: labels_batch, keep_prob: 1.0})
                writer.add_summary(summ, step)

                # Log summaries run on the evaluation/validation data
                summ = sess.run(evaluation_summaries,
                        feed_dict={x: eval_data, y: eval_labels, keep_prob: 1.0})
                writer.add_summary(summ, step)

                # Log image summary
                #randoms = [np.random.normal(0, 1, n_latent) for _ in range(batch_size)]
                #summ = sess.run(image_summary,
                #        feed_dict={sampled_z: randoms, keep_prob: 1.0})
                #writer.add_summary(summ, step)

                writer.flush()

        writer.flush()

if __name__ == '__main__':
    # Used when training on Kamiak
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', default="models", type=str, help="Directory for saving model files")
    parser.add_argument('--logdir', default="logs", type=str, help="Directory for saving log files")
    parser.add_argument('--eval', default=20, type=int, help="Number of images to use for evaluation")
    args = parser.parse_args()

    # Load dataset
    train_data, train_labels = load_data("trivial/positive_slope_TRAIN")
    test_data, test_labels = load_data("trivial/positive_slope_TEST")

    # Information about dataset
    num_features = 1 # e.g. trivial & Plane datasets only have one feature per time step
    time_steps = train_data.shape[1]
    num_classes = len(np.unique(train_labels))
    data_info = (time_steps, num_features, num_classes)

    # One-hot encoding
    train_data, train_labels = one_hot(train_data, train_labels, num_classes)
    test_data, test_labels = one_hot(test_data, test_labels, num_classes)

    # Train and evaluate
    train(data_info, train_data, train_labels,
            test_data, test_labels,
            num_eval=args.eval,
            model_dir=args.modeldir,
            log_dir=args.logdir)
