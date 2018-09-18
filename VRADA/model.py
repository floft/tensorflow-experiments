"""
Create models

This provides the functions like build_lstm and build_vrnn that are used in training.
"""
import tensorflow as tf

from VRNN import VRNNCell
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

def classifier(x, num_classes, keep_prob=None):
    """
    We'll use the same clasifier for task or domain classification

    Returns both output without applying softmax for use in loss function
    and after for use in prediction. See softmax_cross_entropy_with_logits_v2
    documentation: "This op expects unscaled logits, ..."
    """
    classifier_output = tf.contrib.layers.fully_connected(
            x, num_classes, activation_fn=None, scope="classifier")
    if keep_prob is not None:
        classifier_output = tf.nn.dropout(classifier_output, keep_prob)
    softmax_output = tf.nn.softmax(classifier_output, name="softmax")
    
    return classifier_output, softmax_output

def build_model(x, y, domain, grl_lambda, keep_prob, training,
        num_classes, adaptation=True):
    """
    Creates the feature extractor, task classifier, domain classifier

    Inputs:
        x -- fed into feature extractor
        y -- task labels
        domain -- [[1,0], [0,1], ...] for source or target domain
        glr_lambda -- float placeholder for lambda for gradient reversal layer
        keep_prob -- float placeholder for dropout probability
        training -- boolean placeholder for if we're training
        adaptation -- boolean whether we wish to perform adaptation or not
    Outputs:
        task_softmax, domain_softmax -- predictions of classifiers
        task_loss, domain_loss -- losses
        feature_extractor -- output of feature extractor (e.g. for t-SNE)
        summaries -- more summaries to save
    """

    with tf.variable_scope("feature_extractor"):
        # We'll only use the output at the last time step for classification
        feature_extractor = x
        #feature_extractor = tf.reshape(x, [tf.shape(x)[0], 25]) # alternatively, bypass RNN and just use dense
        feature_extractor = tf.contrib.layers.fully_connected(feature_extractor, 50)
        feature_extractor = tf.nn.dropout(feature_extractor, keep_prob)
        feature_extractor = tf.contrib.layers.fully_connected(feature_extractor, 25)
        feature_extractor = tf.nn.dropout(feature_extractor, keep_prob)

        # Use the RNN output for both domain and task inputs
        task_input = feature_extractor
        domain_input = feature_extractor

    # Pass last output to fully connected then softmax to get class prediction
    with tf.variable_scope("task_classifier"):
        task_classifier, task_softmax = classifier(task_input, num_classes, keep_prob)

    # Also pass output to domain classifier
    # Note: always have 2 domains, so set outputs to 2
    with tf.variable_scope("domain_classifier"):
        gradient_reversal_layer = flip_gradient(domain_input, grl_lambda)
        domain_classifier = tf.contrib.layers.fully_connected(gradient_reversal_layer, 15)
        domain_classifier = tf.nn.dropout(domain_classifier, keep_prob)
        domain_classifier, domain_softmax = classifier(domain_classifier, 2, keep_prob)

    # If doing domain adaptation, then we'll need to ignore the second half of the
    # batch for task classification during training since we don't know the labels
    # of the target data
    if adaptation:
        with tf.variable_scope("only_use_source_labels"):
            # Note: this is twice the batch_size in the train() function since we cut
            # it in half there -- this is the sum of both source and target data
            batch_size = tf.shape(task_input)[0]

            # Note: I'm doing this after the classification layers because if you do
            # it before, then fully_connected complains that the last dimension is
            # None (i.e. not known till we run the graph). Thus, we'll do it after
            # all the fully-connected layers.
            #
            # Alternatively, I could do matmul(weights, task_input) + bias and store
            # weights on my own if I do really need to do this at some point.
            #
            # See: https://github.com/pumpikano/tf-dann/blob/master/Blobs-DANN.ipynb
            task_classifier = tf.cond(training,
                lambda: tf.slice(task_classifier, [0, 0], [batch_size // 2, -1]),
                lambda: task_classifier)
            task_softmax = tf.cond(training,
                lambda: tf.slice(task_softmax, [0, 0], [batch_size // 2, -1]),
                lambda: task_softmax)
            y = tf.cond(training,
                lambda: tf.slice(y, [0, 0], [batch_size // 2, -1]),
                lambda: y)

    # Losses
    with tf.variable_scope("task_loss"):
        task_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=y, logits=task_classifier))

    with tf.variable_scope("domain_loss"):
        domain_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=domain, logits=domain_classifier))

    # Extra summaries
    summaries = [
        tf.summary.scalar("loss/task_loss", task_loss),
        tf.summary.scalar("loss/domain_loss", domain_loss),
    ]

    return task_softmax, domain_softmax, task_loss, domain_loss, \
        feature_extractor, summaries

def build_lstm(x, y, domain, grl_lambda, keep_prob, training,
            num_classes, num_features, adaptation):
    """ LSTM for a baseline """
    # Build LSTM
    with tf.variable_scope("rnn_model"):
        initial_state, outputs, cell, final_state = build_rnn(x, keep_prob, [
            tf.contrib.rnn.BasicLSTMCell(100),
        ])

        rnn_output = outputs[:, -1]

    # Other model components passing in output from RNN
    task_softmax, domain_softmax, task_loss, domain_loss, \
        feature_extractor, summaries = build_model(
            rnn_output, y, domain, grl_lambda, keep_prob, training,
            num_classes, adaptation)

    # Total loss is the sum
    with tf.variable_scope("total_loss"):
        total_loss = task_loss

        if adaptation:
            total_loss += domain_loss
    
    summaries += [
        tf.summary.histogram("outputs/feature_extractor", feature_extractor),
        tf.summary.histogram("outputs/task_classifier", task_softmax),
        tf.summary.histogram("outputs/domain_classifier", domain_softmax),
    ]

    # We can't generate with an LSTM
    extra_outputs = None

    return task_softmax, domain_softmax, total_loss, \
        feature_extractor, summaries, extra_outputs

def build_vrnn(x, y, domain, grl_lambda, keep_prob, training,
            num_classes, num_features, adaptation, eps=1e-9):
    """ VRNN model """
    # Build VRNN
    with tf.variable_scope("rnn_model"):
        initial_state, outputs, cell, final_state = build_rnn(x, keep_prob, [
            VRNNCell(num_features, 100, 50, training, batch_norm=False),
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
            = outputs

        rnn_output = z_1[:,-1] # VRADA uses z not h

    # Other model components passing in output from RNN
    task_softmax, domain_softmax, task_loss, domain_loss, \
        feature_extractor, summaries = build_model(
            rnn_output, y, domain, grl_lambda, keep_prob, training,
            num_classes, adaptation)

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

    # Total loss is sum of all of them
    with tf.variable_scope("total_loss"):
        total_loss = task_loss + tf.reduce_mean(kl_loss) + tf.reduce_mean(likelihood_loss)
        
        if adaptation:
            total_loss += domain_loss

    # Extra summaries
    summaries += [
        tf.summary.scalar("loss/kl", tf.reduce_mean(kl_loss)),
        tf.summary.scalar("loss/likelihood", tf.reduce_mean(likelihood_loss)),
        tf.summary.histogram("outputs/phi_x", x_1),
        tf.summary.histogram("outputs/phi_z", z_1),
        tf.summary.histogram("encoder/mu", encoder_mu),
        tf.summary.histogram("encoder/sigma", encoder_sigma),
        tf.summary.histogram("decoder/mu", decoder_mu),
        tf.summary.histogram("decoder/sigma", decoder_sigma),
        tf.summary.histogram("prior/mu", prior_mu),
        tf.summary.histogram("prior/sigma", prior_sigma),
    ]

    # So we can generate sample time-series as well
    #
    # Average over the batch
    extra_outputs = [
        tf.reduce_mean(decoder_mu, axis=0),
        tf.reduce_mean(decoder_sigma, axis=0),
    ]

    return task_softmax, domain_softmax, total_loss, \
        feature_extractor, summaries, extra_outputs