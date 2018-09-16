#!/usr/bin/env python
#
# VRADA
#
# See the paper: https://openreview.net/pdf?id=rk9eAFcxg
# Most of this code is from the my VRNN experiment in ../VRNN/VRNN.py
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

from VRNN import VRNNCell
from load_data import IteratorInitializerHook, load_data, one_hot, _get_input_fn
from flip_gradient import flip_gradient

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

def train(data_info,
        features_a, labels_a, test_features_a, test_labels_a,
        features_b, labels_b, test_features_b, test_labels_b,
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

    # Input training data
    input_fn_a, input_hook_a = _get_input_fn(features_a, labels_a, batch_size)
    input_fn_b, input_hook_b = _get_input_fn(features_b, labels_b, batch_size)
    next_data_batch_a, next_labels_batch_a = input_fn_a()
    next_data_batch_b, next_labels_batch_b = input_fn_b()

    # Load all the test data in one batch (we'll assume test set is small for now)
    eval_input_fn_a, eval_input_hook_a = _get_input_fn(
            test_features_a, test_labels_a, test_features_a.shape[0], evaluation=True)
    eval_input_fn_b, eval_input_hook_b = _get_input_fn(
            test_features_b, test_labels_b, test_features_b.shape[0], evaluation=True)
    next_data_batch_test_a, next_labels_batch_test_a = eval_input_fn_a()
    next_data_batch_test_b, next_labels_batch_test_b = eval_input_fn_b()

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

    # Summaries - training and evaluation for both domains A and B
    training_summaries_a = tf.summary.merge([
        tf.summary.scalar("loss/total_loss", tf.reduce_mean(loss)),
        tf.summary.scalar("accuracy/a/training", accuracy)
    ]+model_summaries)
    training_summaries_b = tf.summary.merge([
        tf.summary.scalar("accuracy/b/training", accuracy)
    ])
    evaluation_summaries_a = tf.summary.merge([
        tf.summary.scalar("accuracy/a/validation", accuracy)
    ])
    evaluation_summaries_b = tf.summary.merge([
        tf.summary.scalar("accuracy/b/validation", accuracy)
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
    with tf.train.SingularMonitoredSession(checkpoint_dir=model_dir, hooks=[
                input_hook_a, input_hook_b,
                eval_input_hook_a, eval_input_hook_b,
                saver_hook
            ]) as sess:

        # Get evaluation batch once
        eval_data_a, eval_labels_a, eval_data_b, eval_labels_b = sess.run([
            next_data_batch_test_a, next_labels_batch_test_a,
            next_data_batch_test_b, next_labels_batch_test_b,
        ])

        for i in range(sess.run(global_step),num_steps+1):
            if i == 0:
                writer.add_graph(sess.graph)

            t = time.time()
            data_batch_a, labels_batch_a, data_batch_b, labels_batch_b = sess.run([
                next_data_batch_a, next_labels_batch_a,
                next_data_batch_b, next_labels_batch_b,
            ])
            _, step = sess.run([optimizer, inc_global_step],
                    feed_dict={x: data_batch_a, y: labels_batch_a, keep_prob: 0.8, training: True})
            t = time.time() - t

            if i%log_save_steps == 0:
                # Log the step time
                summ = tf.Summary(value=[tf.Summary.Value(tag="step_time", simple_value=t)])
                writer.add_summary(summ, step)

                # Log summaries run on the training data
                summ = sess.run(training_summaries_a,
                        feed_dict={x: data_batch_a, y: labels_batch_a, keep_prob: 1.0, training: False})
                writer.add_summary(summ, step)
                summ = sess.run(training_summaries_b,
                        feed_dict={x: data_batch_b, y: labels_batch_b, keep_prob: 1.0, training: False})
                writer.add_summary(summ, step)

                # Log summaries run on the evaluation/validation data
                summ = sess.run(evaluation_summaries_a,
                        feed_dict={x: eval_data_a, y: eval_labels_a, keep_prob: 1.0, training: False})
                writer.add_summary(summ, step)
                summ = sess.run(evaluation_summaries_b,
                        feed_dict={x: eval_data_b, y: eval_labels_b, keep_prob: 1.0, training: False})
                writer.add_summary(summ, step)

                writer.flush()

        writer.flush()

if __name__ == '__main__':
    # Used when training on Kamiak
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', default="models", type=str, help="Directory for saving model files")
    parser.add_argument('--logdir', default="logs", type=str, help="Directory for saving log files")
    args = parser.parse_args()

    # Load datasets - domains A & B
    train_data_a, train_labels_a = load_data("trivial/positive_slope_TRAIN")
    test_data_a, test_labels_a = load_data("trivial/positive_slope_TEST")

    train_data_b, train_labels_b = load_data("trivial/positive_sine_TRAIN")
    test_data_b, test_labels_b = load_data("trivial/positive_sine_TEST")

    # Information about dataset - at the moment these are the same for both domains
    num_features = 1
    time_steps = train_data_a.shape[1]
    num_classes = len(np.unique(train_labels_a))
    data_info = (time_steps, num_features, num_classes)

    # One-hot encoding
    train_data_a, train_labels_a = one_hot(train_data_a, train_labels_a, num_classes)
    test_data_a, test_labels_a = one_hot(test_data_a, test_labels_a, num_classes)
    train_data_b, train_labels_b = one_hot(train_data_b, train_labels_b, num_classes)
    test_data_b, test_labels_b = one_hot(test_data_b, test_labels_b, num_classes)

    # Train and evaluate LSTM - i.e. no adaptation
    """
    train(data_info,
            train_data_a, train_labels_a, test_data_a, test_labels_a,
            train_data_b, train_labels_b, test_data_b, test_labels_b,
            model_func=lstm_model,
            model_dir="plane-test/lstm-models",
            log_dir="plane-test/lstm-logs")
    #tf.reset_default_graph()
    """

    # Train and evaluate VRNN - i.e. no adaptation
    train(data_info,
            train_data_a, train_labels_a, test_data_a, test_labels_a,
            train_data_b, train_labels_b, test_data_b, test_labels_b,
            model_func=vrnn_model,
            model_dir="plane-test/vrnn-models",
            log_dir="plane-test/vrnn-logs")

    # Train and evaluate VRADA - i.e. VRNN but with adversarial domain adaptation
    """
    train(data_info,
            train_data_a, train_labels_a, test_data_a, test_labels_a,
            train_data_b, train_labels_b, test_data_b, test_labels_b,
            model_func=vrada_model,
            model_dir="plane-test/vrada-models",
            log_dir="plane-test/vrada-logs")
            #model_dir=args.modeldir,
            #log_dir=args.logdir)
    """
