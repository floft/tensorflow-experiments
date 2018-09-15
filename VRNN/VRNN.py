#!/usr/bin/env python
#
# VRNN
#
# For VAE aspect of VRNNs, great explanation:
# https://www.youtube.com/watch?v=uaaqyVS9-rM
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
#  - https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/ops/rnn_cell_impl.py
class VRNNCell(tf.contrib.rnn.LayerRNNCell):
    def __init__(self, x_dim, h_dim, z_dim, training, batch_norm=False, **kwargs):
        self.batch_norm = batch_norm
        self.training = training # placeholder for batch norm

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
        self.cell = tf.contrib.rnn.BasicLSTMCell(self.n_h)
             #input_shape=(None, self.n_dec_hidden+self.n_z_1))

        super(VRNNCell, self).__init__(**kwargs)

    @property
    def state_size(self):
        # Note: first two are the state of the LSTM
        return (self.n_h, self.n_h,
                self.n_z, self.n_z,
                self.n_x, self.n_x,
                self.n_z, self.n_z,
                self.n_x_1, self.n_z_1)

    @property
    def output_size(self):
        """ LSTM output is h which is in [0] """
        return (self.n_h, self.n_h,
                self.n_z, self.n_z,
                self.n_x, self.n_x,
                self.n_z, self.n_z,
                self.n_x_1, self.n_z_1)

    def build(self, input_shape):
        # TODO grab input size from input_shape rather than passing in
        # num_features to the constructor

        # Input: previous hidden state
        self.prior_h = self.add_variable('prior/hidden/weights',
            shape=(self.n_h, self.n_prior_hidden), initializer=tf.glorot_uniform_initializer())
        self.prior_mu = self.add_variable('prior/mu/weights',
            shape=(self.n_prior_hidden, self.n_z), initializer=tf.glorot_uniform_initializer())
        self.prior_sigma = self.add_variable('prior/sigma/weights',
            shape=(self.n_prior_hidden, self.n_z), initializer=tf.glorot_uniform_initializer())

        if not self.batch_norm:
            self.prior_h_b = self.add_variable('prior/hidden/bias',
                shape=(self.n_prior_hidden,), initializer=tf.constant_initializer())
            self.prior_sigma_b = self.add_variable('prior/sigma/bias',
                shape=(self.n_z,), initializer=tf.constant_initializer())
        self.prior_mu_b = self.add_variable('prior/mu/bias',
            shape=(self.n_z,), initializer=tf.constant_initializer())

        # Input: x
        self.x_1 = self.add_variable('phi_x/weights',
            shape=(self.n_x, self.n_x_1), initializer=tf.glorot_uniform_initializer())

        if not self.batch_norm:
            self.x_1_b = self.add_variable('phi_x/bias',
                shape=(self.n_x_1,), initializer=tf.constant_initializer())

        # Input: x and previous hidden state
        self.encoder_h = self.add_variable('encoder/hidden/weights',
            shape=(self.n_x_1+self.n_h, self.n_enc_hidden), initializer=tf.glorot_uniform_initializer())
        self.encoder_mu = self.add_variable('encoder/mu/weights',
            shape=(self.n_enc_hidden, self.n_z), initializer=tf.glorot_uniform_initializer())
        self.encoder_sigma = self.add_variable('encoder/sigma/weights',
            shape=(self.n_enc_hidden, self.n_z), initializer=tf.glorot_uniform_initializer())

        if not self.batch_norm:
            self.encoder_h_b = self.add_variable('encoder/hidden/bias',
                shape=(self.n_enc_hidden,), initializer=tf.constant_initializer())
            self.encoder_sigma_b = self.add_variable('encoder/sigma/bias',
                shape=(self.n_z,), initializer=tf.constant_initializer())
        self.encoder_mu_b = self.add_variable('encoder/mu/bias',
            shape=(self.n_z,), initializer=tf.constant_initializer())

        # Input: z = enc_sigma*eps + enc_mu -- i.e. reparameterization trick
        self.z_1 = self.add_variable('phi_z/weights',
            shape=(self.n_z, self.n_z_1), initializer=tf.glorot_uniform_initializer())

        if not self.batch_norm:
            self.z_1_b = self.add_variable('phi_z/bias',
                shape=(self.n_z_1,), initializer=tf.constant_initializer())

        # Input: latent variable (z) and previous hidden state
        self.decoder_h = self.add_variable('decoder/hidden/weights',
            shape=(self.n_z+self.n_h, self.n_dec_hidden), initializer=tf.glorot_uniform_initializer())
        self.decoder_mu = self.add_variable('decoder/mu/weights',
            shape=(self.n_dec_hidden, self.n_x), initializer=tf.glorot_uniform_initializer())
        self.decoder_sigma = self.add_variable('decoder/sigma/weights',
            shape=(self.n_dec_hidden, self.n_x), initializer=tf.glorot_uniform_initializer())

        if not self.batch_norm:
            self.decoder_h_b = self.add_variable('decoder/hidden/bias',
                shape=(self.n_dec_hidden,), initializer=tf.constant_initializer())
            self.decoder_sigma_b = self.add_variable('decoder/sigma/bias',
                shape=(self.n_x,), initializer=tf.constant_initializer())
        self.decoder_mu_b = self.add_variable('decoder/mu/bias',
            shape=(self.n_x,), initializer=tf.constant_initializer())

        super(VRNNCell, self).build(input_shape)

    def call(self, inputs, states):
        # Get relevant states
        h = states[0]
        c = states[1] # only passed to the LSTM

        # Input: previous hidden state (h)
        #
        # Note: update_collections=None from https://github.com/tensorflow/tensorflow/issues/6087
        # And, that's why I'm not using tf.layers.batch_normalization
        if self.batch_norm:
            prior_h = tf.nn.relu(tf.contrib.layers.batch_norm(tf.matmul(h, self.prior_h), is_training=self.training, updates_collections=None))
            prior_sigma = tf.nn.softplus(tf.contrib.layers.batch_norm(tf.matmul(prior_h, self.prior_sigma), is_training=self.training, updates_collections=None)) # >= 0
        else:
            prior_h = tf.nn.relu(tf.matmul(h, self.prior_h) + self.prior_h_b)
            prior_sigma = tf.nn.softplus(tf.matmul(prior_h, self.prior_sigma) + self.prior_sigma_b) # >= 0
        prior_mu = tf.matmul(prior_h, self.prior_mu) + self.prior_mu_b

        # Input: x
        #
        # Note: removed ReLU since in the dataset not all x values are positive
        if self.batch_norm:
            x_1 = tf.contrib.layers.batch_norm(tf.matmul(inputs, self.x_1), is_training=self.training, updates_collections=None)
        else:
            x_1 = tf.matmul(inputs, self.x_1) + self.x_1_b

        # Input: x and previous hidden state
        encoder_input = tf.concat((x_1, h), 1)
        if self.batch_norm:
            encoder_h = tf.nn.relu(tf.contrib.layers.batch_norm(tf.matmul(encoder_input, self.encoder_h), is_training=self.training, updates_collections=None))
            encoder_sigma = tf.nn.softplus(tf.contrib.layers.batch_norm(tf.matmul(encoder_h, self.encoder_sigma), is_training=self.training, updates_collections=None))
        else:
            encoder_h = tf.nn.relu(tf.matmul(encoder_input, self.encoder_h) + self.encoder_h_b)
            encoder_sigma = tf.nn.softplus(tf.matmul(encoder_h, self.encoder_sigma) + self.encoder_sigma_b)
        encoder_mu = tf.matmul(encoder_h, self.encoder_mu) + self.encoder_mu_b

        # Input: z = enc_sigma*eps + enc_mu -- i.e. reparameterization trick
        batch_size = tf.shape(inputs)[0] # https://github.com/tensorflow/tensorflow/issues/373
        eps = tf.random_normal((batch_size, self.n_z), dtype=tf.float32)
        z = encoder_sigma*eps + encoder_mu
        if self.batch_norm:
            z_1 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.matmul(z, self.z_1), is_training=self.training, updates_collections=None))
        else:
            z_1 = tf.nn.relu(tf.matmul(z, self.z_1) + self.z_1_b)

        # Input: latent variable (z) and previous hidden state
        decoder_input = tf.concat((z_1, h), 1)
        if self.batch_norm:
            decoder_h = tf.nn.relu(tf.contrib.layers.batch_norm(tf.matmul(decoder_input, self.decoder_h), is_training=self.training, updates_collections=None))
            decoder_sigma = tf.nn.softplus(tf.contrib.layers.batch_norm(tf.matmul(decoder_h, self.decoder_sigma), is_training=self.training, updates_collections=None))
        else:
            decoder_h = tf.nn.relu(tf.matmul(decoder_input, self.decoder_h) + self.decoder_h_b)
            decoder_sigma = tf.nn.softplus(tf.matmul(decoder_h, self.decoder_sigma) + self.decoder_sigma_b)
        decoder_mu = tf.matmul(decoder_h, self.decoder_mu) + self.decoder_mu_b

        # Pass to cell (e.g. LSTM). Note that the LSTM has both "h" and "c" that are combined
        # into the same next state vector. We'll combine them together to pass in and split them
        # back out after the LSTM returns the next state.
        rnn_cell_input = tf.concat((x_1, z_1), 1)
        output, (c_next, h_next) = self.cell(rnn_cell_input, [c, h]) # Note: (h,c) in Keras (c,h) in contrib
        #output, (h_next, c_next) = self.cell(rnn_cell_input, [h, c]) # Note: (h,c) in Keras (c,h) in contrib

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
            x_1,
            z_1,
        )

        #return output, next_state
        return next_state, next_state

def build_rnn(x, keep_prob, layers):
    """
    Multi-layer LSTM
    https://github.com/GarrettHoffman/lstm-oreilly

    x, keep_prob - placeholders
    layers - cell for each layer, e.g. [LSTMCell(...), LSTMCell(...), ...]
    """

    drops = [tf.contrib.rnn.DropoutWrapper(l, output_keep_prob=keep_prob) for l in layers]
    cell = tf.contrib.rnn.MultiRNNCell(drops)

    batch_size = tf.shape(x)[0]
    initial_state = cell.zero_state(batch_size, tf.float32)

    outputs, final_state = tf.nn.dynamic_rnn(cell, x, initial_state=initial_state)

    return initial_state, outputs, cell, final_state

def lstm_model(x, y, keep_prob, training, num_classes, num_features):
    """ Create an LSTM model as a baseline """
    # Build the LSTM
    with tf.variable_scope("rnn_model"):
        initial_state, outputs, cell, final_state = build_rnn(x, keep_prob, [
            #tf.contrib.rnn.BasicLSTMCell(128), tf.contrib.rnn.BasicLSTMCell(128),
            tf.contrib.rnn.BasicLSTMCell(128),
        ])

    # Pass last output to fully connected then softmax to get class prediction
    yhat = tf.contrib.layers.fully_connected(
            outputs[:, -1], num_classes, activation_fn=tf.nn.softmax)

    # Loss
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=yhat)

    return yhat, loss, []

def vrnn_model(x, y, keep_prob, training, num_classes, num_features, eps=1e-9):
    """ Create the VRNN model """
    #
    # Model
    #
    with tf.variable_scope("rnn_model"):
        initial_state, outputs, cell, final_state = build_rnn(x, keep_prob, [
            #VRNNCell(num_features, 128, 64), VRNNCell(128, 128, 64), # h_dim of l_i must be num_features of l_(i+1)
            VRNNCell(num_features, 128, 10, training, batch_norm=False),
        ])
        # Note: if you try using more than one layer above, then you need to
        # change the loss since for instance if you put an LSTM layer before
        # the VRNN cell, then no longer is the input to the layer x as
        # specified in the loss but now it's the output of the first LSTM layer
        # that the VRNN layer should be learning how to reconstruct. Thus, for
        # now I'll keep it simple and not have multiple layers.

    h, c, \
    encoder_mu, encoder_sigma, \
    decoder_mu, decoder_sigma, \
    prior_mu, prior_sigma, \
    x_1, z_1, \
        = outputs # TODO if multiple layers maybe do loss for each of these

    with tf.variable_scope("classifier"):
        # Pass last output to fully connected then softmax to get class prediction
        output_for_classifier = h # h is the LSTM output
        #output_for_classifier = z_1 # z is the latent variable
        yhat = tf.contrib.layers.fully_connected(
                output_for_classifier[:, -1], num_classes, activation_fn=tf.nn.softmax)

    #
    # Loss
    #
    # KL divergence
    # https://stats.stackexchange.com/q/7440
    # https://github.com/kimkilho/tensorflow-vrnn/blob/master/main.py
    with tf.variable_scope("kl_gaussian"):
        kl_loss = tf.reduce_mean(tf.reduce_mean(
                tf.log(tf.maximum(eps, prior_sigma)) - tf.log(tf.maximum(eps, encoder_sigma))
                + 0.5*(tf.square(encoder_sigma) + tf.square(encoder_mu - prior_mu))
                    / tf.maximum(eps, tf.square(prior_sigma))
                - 0.5,
            axis=1), axis=1)

    # Reshape [batch_size,time_steps,num_features] -> [batch_size*time_steps,num_features]
    # so that (decoder_mu - x) will broadcast correctly
    #x_transpose = tf.transpose(x, [0, 2, 1])
    #decoder_mu_reshape = tf.reshape(decoder_mu, [tf.shape(decoder_mu)[0]*tf.shape(decoder_mu)[1], tf.shape(decoder_mu)[2]])
    #x_reshape = tf.reshape(x, [tf.shape(x)[0]*tf.shape(x)[1], tf.shape(x)[2]])

    # Negative log likelihood:
    # https://papers.nips.cc/paper/7219-simple-and-scalable-predictive-uncertainty-estimation-using-deep-ensembles.pdf
    # https://fairyonice.github.io/Create-a-neural-net-with-a-negative-log-likelihood-as-a-loss.html
    with tf.variable_scope("negative_log_likelihood"):
        #likelihood_loss = tf.reduce_sum(tf.squared_difference(x, x_1), 1)
        likelihood_loss = 0.5*tf.reduce_mean(tf.reduce_mean(
            tf.square(decoder_mu - x) / tf.maximum(eps, tf.square(decoder_sigma))
            + tf.log(tf.maximum(eps, tf.square(decoder_sigma))),
        axis=1), axis=1)

    # We'd also like to classify well
    categorical_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=yhat)

    loss = tf.reduce_mean(kl_loss + likelihood_loss + categorical_loss)

    #
    # Extra summaries
    #
    summaries = [
        tf.summary.scalar("loss/kl", tf.reduce_mean(kl_loss)),
        tf.summary.scalar("loss/likelihood", tf.reduce_mean(likelihood_loss)),
        tf.summary.scalar("loss/categorical", tf.reduce_mean(categorical_loss)),
        tf.summary.histogram("outputs/phi_x", x_1),
        tf.summary.histogram("outputs/phi_z", z_1),
        tf.summary.histogram("encoder/mu", encoder_mu),
        tf.summary.histogram("encoder/sigma", encoder_sigma),
        tf.summary.histogram("decoder/mu", decoder_mu),
        tf.summary.histogram("decoder/sigma", decoder_sigma),
        tf.summary.histogram("prior/mu", prior_mu),
        tf.summary.histogram("prior/sigma", prior_sigma),
    ]

    return yhat, loss, summaries

def train(data_info, features, labels,
        features_test, labels_test,
        model_func=lstm_model,
        batch_size=64,
        num_steps=1000,
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
    training = tf.placeholder(tf.bool, name='training')

    # Model and loss -- e.g. lstm_model
    #
    # Optionally also returns additional summaries to log, e.g. loss components
    yhat, loss, model_summaries = model_func(x, y, keep_prob, training, num_classes, num_features)

    # Accuracy -- https://stackoverflow.com/a/42608050/2698494
    accuracy = tf.reduce_mean(tf.cast(
        tf.equal(tf.argmax(y, axis=-1), tf.argmax(yhat, axis=-1)),
    tf.float32))

    # Optimizer - update ops for batch norm
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, "rnn_model")):
        optimizer = tf.train.AdamOptimizer().minimize(loss)

    # Summaries
    training_summaries = tf.summary.merge([
        tf.summary.scalar("loss/total_loss", tf.reduce_mean(loss)),
        tf.summary.scalar("accuracy/training", accuracy)
    ]+model_summaries)
    evaluation_summaries = tf.summary.merge([
        tf.summary.scalar("accuracy/validation", accuracy)
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
                    feed_dict={x: data_batch, y: labels_batch, keep_prob: 0.8, training: True})
            t = time.time() - t

            if i%log_save_steps == 0:
                # Log the step time
                summ = tf.Summary(value=[tf.Summary.Value(tag="step_time", simple_value=t)])
                writer.add_summary(summ, step)

                # Log summaries run on the training data
                summ = sess.run(training_summaries,
                        feed_dict={x: data_batch, y: labels_batch, keep_prob: 1.0, training: False})
                writer.add_summary(summ, step)

                # Log summaries run on the evaluation/validation data
                summ = sess.run(evaluation_summaries,
                        feed_dict={x: eval_data, y: eval_labels, keep_prob: 1.0, training: False})
                writer.add_summary(summ, step)

                writer.flush()

        writer.flush()

if __name__ == '__main__':
    # Used when training on Kamiak
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', default="models", type=str, help="Directory for saving model files")
    parser.add_argument('--logdir', default="logs", type=str, help="Directory for saving log files")
    args = parser.parse_args()

    # Load dataset
    #train_data, train_labels = load_data("trivial/positive_slope_TRAIN")
    #test_data, test_labels = load_data("trivial/positive_slope_TEST")
    train_data, train_labels = load_data("Plane/Plane_TRAIN")
    test_data, test_labels = load_data("Plane/Plane_TEST")

    # Information about dataset
    num_features = 1 # e.g. trivial & Plane datasets only have one feature per time step
    time_steps = train_data.shape[1]
    num_classes = len(np.unique(train_labels))
    data_info = (time_steps, num_features, num_classes)

    # One-hot encoding
    train_data, train_labels = one_hot(train_data, train_labels, num_classes)
    test_data, test_labels = one_hot(test_data, test_labels, num_classes)

    # Train and evaluate LSTM
    """
    train(data_info, train_data, train_labels,
            test_data, test_labels,
            model_func=lstm_model,
            model_dir="plane-test/lstm-models",
            log_dir="plane-test/lstm-logs")
    tf.reset_default_graph()
    """

    # Train and evaluate VRNN
    train(data_info, train_data, train_labels,
            test_data, test_labels,
            model_func=vrnn_model,
            model_dir="plane-test/vrnn12-models",
            log_dir="plane-test/vrnn12-logs")
            #model_dir=args.modeldir,
            #log_dir=args.logdir)
