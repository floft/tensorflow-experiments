"""
VRADA implementation

See the paper: https://openreview.net/pdf?id=rk9eAFcxg
See coauthor blog post: https://wcarvalho.github.io/research/2017/04/23/vrada/
"""
import os
import re
import time
import argparse
import pathlib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
#from tensorflow.contrib.tensorboard.plugins import projector

# Due to this being run on Kamiak, that doesn't have _tkinter, we have to set a
# different backend otherwise it'll error
# https://stackoverflow.com/a/40931739/2698494
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt

from plot import plot_embedding, plot_random_time_series
from model import build_lstm, build_vrnn
from load_data import IteratorInitializerHook, \
    load_data, one_hot, \
    domain_labels, _get_input_fn

def train(data_info,
        features_a, labels_a, test_features_a, test_labels_a,
        features_b, labels_b, test_features_b, test_labels_b,
        model_func=build_lstm,
        batch_size=128,
        num_steps=1000,
        learning_rate=0.0003,
        dropout_keep_prob=0.8,
        model_dir="models",
        log_dir="logs",
        embedding_prefix=None,
        model_save_steps=200,
        log_save_steps=5,
        log_extra_save_steps=25,
        adaptation=True):

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Data stats
    time_steps, num_features, num_classes = data_info

    # For adaptation, we'll be concatenating together half source and half target
    # data, so to keep the batch_size about the same, we'll cut it in half
    if adaptation:
        batch_size = batch_size // 2

    # Input training data
    with tf.variable_scope("training_data_a"):
        input_fn_a, input_hook_a = _get_input_fn(features_a, labels_a, batch_size)
        next_data_batch_a, next_labels_batch_a = input_fn_a()
    with tf.variable_scope("training_data_b"):
        input_fn_b, input_hook_b = _get_input_fn(features_b, labels_b, batch_size)
        next_data_batch_b, next_labels_batch_b = input_fn_b()

    # Load all the test data in one batch (we'll assume test set is small for now)
    with tf.variable_scope("evaluation_data_a"):
        eval_input_fn_a, eval_input_hook_a = _get_input_fn(
                test_features_a, test_labels_a, test_features_a.shape[0], evaluation=True)
        next_data_batch_test_a, next_labels_batch_test_a = eval_input_fn_a()
    with tf.variable_scope("evaluation_data_b"):
        eval_input_fn_b, eval_input_hook_b = _get_input_fn(
                test_features_b, test_labels_b, test_features_b.shape[0], evaluation=True)
        next_data_batch_test_b, next_labels_batch_test_b = eval_input_fn_b()

    # Inputs
    keep_prob = tf.placeholder_with_default(1.0, shape=()) # for dropout
    x = tf.placeholder(tf.float32, [None, time_steps, num_features], name='x') # input data
    domain = tf.placeholder(tf.float32, [None, 2], name='domain') # which domain
    y = tf.placeholder(tf.float32, [None, num_classes], name='y') # class 1, 2, etc. one-hot
    training = tf.placeholder(tf.bool, name='training') # whether we're training (batch norm)
    grl_lambda = tf.placeholder_with_default(1.0, shape=()) # gradient multiplier for gradient reversal layer
    lr = tf.placeholder(tf.float32, (), name='learning_rate')

    # Source domain will be [[1,0], [1,0], ...] and target domain [[0,1], [0,1], ...]
    #
    # Size of training batch
    source_domain = domain_labels(0, batch_size)
    target_domain = domain_labels(1, batch_size)
    # Size of evaluation batch - TODO when lots of data, we'll need to batch this
    eval_source_domain = domain_labels(0, test_features_a.shape[0])
    eval_target_domain = domain_labels(1, test_features_b.shape[0])

    # Model, loss, feature extractor output -- e.g. using build_lstm or build_vrnn
    #
    # Optionally also returns additional summaries to log, e.g. loss components
    task_classifier, domain_classifier, total_loss, \
    feature_extractor, model_summaries, extra_model_outputs = \
        model_func(x, y, domain, grl_lambda, keep_prob, training,
            num_classes, num_features, adaptation)

    # Accuracy of the classifiers -- https://stackoverflow.com/a/42608050/2698494
    with tf.variable_scope("task_accuracy"):
        task_accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(y, axis=-1), tf.argmax(task_classifier, axis=-1)),
        tf.float32))
    with tf.variable_scope("domain_accuracy"):
        domain_accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(domain, axis=-1), tf.argmax(domain_classifier, axis=-1)),
        tf.float32))
    
    # Get variables of model
    variables = tf.trainable_variables()
    rnn_vars = [v for v in variables if 'rnn_model' in v.name]
    feature_extractor_vars = [v for v in variables if 'feature_extractor' in v.name]
    task_classifier_vars = [v for v in variables if 'task_classifier' in v.name]
    domain_classifier_vars = [v for v in variables if 'domain_classifier' in v.name]

    # Optimizer - update ops for batch norm (not sure batch norm is working though...)
    #with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, "rnn_model")):
    optimizer = tf.train.AdamOptimizer(lr)
    train_all = optimizer.minimize(total_loss)
    train_notdomain = optimizer.minimize(total_loss,
        var_list=rnn_vars+feature_extractor_vars+task_classifier_vars)

    if adaptation:
        train_domain = optimizer.minimize(total_loss,
            var_list=domain_classifier_vars)

    # Summaries - training and evaluation for both domains A and B
    training_summaries_a = tf.summary.merge([
        tf.summary.scalar("loss/total_loss", total_loss),
        tf.summary.scalar("accuracy/task/source/training", task_accuracy),
        tf.summary.scalar("accuracy/domain/source/training", domain_accuracy),
    ])
    training_summaries_extra_a = tf.summary.merge(model_summaries)
    training_summaries_b = tf.summary.merge([
        tf.summary.scalar("accuracy/task/target/training", task_accuracy),
        tf.summary.scalar("accuracy/domain/target/training", domain_accuracy)
    ])
    evaluation_summaries_a = tf.summary.merge([
        tf.summary.scalar("accuracy/task/source/validation", task_accuracy),
        tf.summary.scalar("accuracy/domain/source/validation", domain_accuracy),
    ])
    evaluation_summaries_b = tf.summary.merge([
        tf.summary.scalar("accuracy/task/target/validation", task_accuracy),
        tf.summary.scalar("accuracy/domain/target/validation", domain_accuracy),
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

            # GRL schedule and learning rate schedule from DANN paper
            grl_lambda_value = 2/(1+np.exp(-10*(i/(num_steps+1))))-1
            #lr_value = 0.001/(1.0+10*i/(num_steps+1))**0.75
            lr_value = learning_rate

            t = time.time()
            step = sess.run(inc_global_step)

            # Get data for this iteration
            data_batch_a, labels_batch_a, data_batch_b, labels_batch_b = sess.run([
                next_data_batch_a, next_labels_batch_a,
                next_data_batch_b, next_labels_batch_b,
            ])

            if adaptation:
                # Concatenate for adaptation - concatenate source labels with all-zero
                # labels for target since we can't use the target labels during
                # unsupervised domain adaptation
                combined_x = np.concatenate((data_batch_a, data_batch_b), axis=0)
                combined_labels = np.concatenate((labels_batch_a, np.zeros(labels_batch_b.shape)), axis=0)
                combined_domain = np.concatenate((source_domain, target_domain), axis=0)

                feed_dict = {
                    x: combined_x, y: combined_labels, domain: combined_domain,
                    grl_lambda: grl_lambda_value,
                    keep_prob: dropout_keep_prob, lr: lr_value, training: True
                }

                # In the paper they split optimization into two parts
                #sess.run(train_notdomain, feed_dict=feed_dict)
                #sess.run(train_domain, feed_dict=feed_dict)

                # Or, train everything in one step which should give a similar result
                sess.run(train_all, feed_dict=feed_dict)
            else:
                # Train task classifier on source domain to be correct
                sess.run(train_notdomain, feed_dict={
                    x: data_batch_a, y: labels_batch_a,
                    keep_prob: dropout_keep_prob, lr: lr_value, training: True
                })

            t = time.time() - t

            if i%log_save_steps == 0:
                # Log the step time
                summ = tf.Summary(value=[
                    tf.Summary.Value(tag="step_time", simple_value=t)
                ])
                writer.add_summary(summ, step)

                # Log summaries run on the training data
                summ = sess.run(training_summaries_a, feed_dict={
                    x: data_batch_a, y: labels_batch_a, domain: source_domain,
                    keep_prob: 1.0, training: False
                })
                writer.add_summary(summ, step)
                summ = sess.run(training_summaries_b, feed_dict={
                    x: data_batch_b, y: labels_batch_b, domain: target_domain,
                    keep_prob: 1.0, training: False
                })
                writer.add_summary(summ, step)

                # Log summaries run on the evaluation/validation data
                summ = sess.run(evaluation_summaries_a, feed_dict={
                    x: eval_data_a, y: eval_labels_a, domain: eval_source_domain,
                    keep_prob: 1.0, training: False
                })
                writer.add_summary(summ, step)
                summ = sess.run(evaluation_summaries_b, feed_dict={
                    x: eval_data_b, y: eval_labels_b, domain: eval_target_domain,
                    keep_prob: 1.0, training: False
                })
                writer.add_summary(summ, step)

                writer.flush()
            
            # Extra stuff only log occasionally, e.g. this is weights and larger stuff
            if i%log_extra_save_steps == 0:
                summ = sess.run(training_summaries_extra_a, feed_dict={
                    x: data_batch_a, y: labels_batch_a, domain: source_domain,
                    keep_prob: 1.0, training: False
                })
                writer.add_summary(summ, step)

                writer.flush()

        writer.flush()

        # Output t-SNE after we've trained everything on the evaluation data
        #
        # Maybe in the future it would be cool to use TensorFlow's projector in TensorBoard
        # https://medium.com/@vegi/visualizing-higher-dimensional-data-using-t-sne-on-tensorboard-7dbf22682cf2
        if embedding_prefix is not None:
            """
            if os.path.exists(tsne_filename+'_tsne_fit.npy'):
                print("Note: generating t-SNE plot using using pre-existing embedding")
                tsne = np.load(tsne_filename+'_tsne_fit.npy')
                pca = np.load(tsne_filename+'_pca_fit.npy')
            else:
                print("Note: did not find t-SNE weights -- recreating")
            
            np.save(tsne_filename+'_tsne_fit', tsne)
            np.save(tsne_filename+'_pca_fit', pca)
            """

            combined_x = np.concatenate((eval_data_a, eval_data_b), axis=0)
            combined_labels = np.concatenate((eval_labels_a, eval_labels_b), axis=0)
            combined_domain = np.concatenate((eval_source_domain, eval_target_domain), axis=0)

            embedding = sess.run(feature_extractor, feed_dict={
                x: combined_x, keep_prob: 1.0, training: False
            })

            tsne = TSNE(n_components=2, init='pca', n_iter=3000).fit_transform(embedding)
            pca = PCA(n_components=2).fit_transform(embedding)

            plot_embedding(tsne, combined_labels.argmax(1), combined_domain.argmax(1),
                title='Domain Adaptation', filename=embedding_prefix+"_tsne.png")
            plot_embedding(pca, combined_labels.argmax(1), combined_domain.argmax(1),
                title='Domain Adaptation', filename=embedding_prefix+"_pca.png")
        
            # Output time-series "reconstructions" from our generator (if VRNN)
            if extra_model_outputs is not None:
                # We'll get the decoder's mu and sigma from the evaluation/validation set since
                # it's much larger than the training batches
                mu, sigma = sess.run(extra_model_outputs, feed_dict={
                    x: eval_data_a, keep_prob: 1.0, training: False
                })

                plot_random_time_series(mu, sigma, title='VRNN Samples (source domain)',
                    filename=embedding_prefix+"_reconstruction_a.png")

                mu, sigma = sess.run(extra_model_outputs, feed_dict={
                    x: eval_data_b, keep_prob: 1.0, training: False
                })

                plot_random_time_series(mu, sigma, title='VRNN Samples (target domain)',
                    filename=embedding_prefix+"_reconstruction_b.png")

def last_modified_number(dir_name, glob):
    """
    Looks in dir_name at all files matching glob and takes number
    from the one last modified
    """
    files = pathlib.Path(dir_name).glob(glob)
    files = sorted(files, key=lambda cp:cp.stat().st_mtime)
    
    if len(files) > 0:
        # Get number from filename
        regex = re.compile(r'\d+')
        numbers = [int(x) for x in regex.findall(str(files[-1]))]
        assert len(numbers) == 1, "Could not determine number from last modified file"
        last = numbers[0]

        return last
    
    return None

if __name__ == '__main__':
    # Used when training on Kamiak
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', default="models", type=str,
        help="Directory for saving model files")
    parser.add_argument('--logdir', default="logs", type=str,
        help="Directory for saving log files")
    args = parser.parse_args()

    # Load datasets - domains A & B

    # Noisy - sine dataset
    # train_data_a, train_labels_a = load_data("trivial/positive_sine_TRAIN")
    # test_data_a, test_labels_a = load_data("trivial/positive_sine_TEST")
    # train_data_b, train_labels_b = load_data("trivial/positive_sine_noise_TRAIN")
    # test_data_b, test_labels_b = load_data("trivial/positive_sine_noise_TEST")

    # Change in y-intercept - sine dataset - doesn't work
    # train_data_a, train_labels_a = load_data("trivial/positive_sine_TRAIN")
    # test_data_a, test_labels_a = load_data("trivial/positive_sine_TEST")
    # train_data_b, train_labels_b = load_data("trivial/positive_sine_low_TRAIN")
    # test_data_b, test_labels_b = load_data("trivial/positive_sine_low_TEST")

    # Change in y-intercept - domain adaptation doesn't work
    train_data_a, train_labels_a = load_data("trivial/positive_slope_TRAIN")
    test_data_a, test_labels_a = load_data("trivial/positive_slope_TEST")
    train_data_b, train_labels_b = load_data("trivial/positive_slope_low_TRAIN")
    test_data_b, test_labels_b = load_data("trivial/positive_slope_low_TEST")
    
    # Noisy - no problem even without adaptation
    # train_data_a, train_labels_a = load_data("trivial/positive_slope_TRAIN")
    # test_data_a, test_labels_a = load_data("trivial/positive_slope_TEST")
    # train_data_b, train_labels_b = load_data("trivial/positive_slope_noise_TRAIN")
    # test_data_b, test_labels_b = load_data("trivial/positive_slope_noise_TEST")

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

    # Train and evaluate LSTM
    attempt = last_modified_number("offset", "lstm-*-logs") + 1
    assert attempt is not None

    train(data_info,
            train_data_a, train_labels_a, test_data_a, test_labels_a,
            train_data_b, train_labels_b, test_data_b, test_labels_b,
            model_func=build_lstm,
            embedding_prefix="images/lstm_da_"+str(attempt),
            model_dir="offset-models/lstm-da"+str(attempt)+"-models",
            log_dir="offset/lstm-da"+str(attempt)+"-logs",
            adaptation=True)

    # train(data_info,
    #         train_data_a, train_labels_a, test_data_a, test_labels_a,
    #         train_data_b, train_labels_b, test_data_b, test_labels_b,
    #         model_func=build_lstm,
    #         embedding_prefix="images/lstm_"+str(attempt),
    #         model_dir="offset-models/lstm-"+str(attempt)+"-models",
    #         log_dir="offset/lstm-"+str(attempt)+"-logs",
    #         adaptation=False)
    
    #tf.reset_default_graph()
    

    # Train and evaluate VRNN
    # attempt = last_modified_number("offset", "vrnn-da*-logs") + 1
    # assert attempt is not None
    
    # train(data_info,
    #         train_data_a, train_labels_a, test_data_a, test_labels_a,
    #         train_data_b, train_labels_b, test_data_b, test_labels_b,
    #         model_func=build_vrnn,
    #         embedding_prefix="images/vrnn_da_"+str(attempt),
    #         model_dir="offset-models/vrnn-da"+str(attempt)+"-models",
    #         log_dir="offset/vrnn-da"+str(attempt)+"-logs",
    #         adaptation=True)

    # Train and evaluate VRADA - i.e. VRNN but with adversarial domain adaptation
    """
    train(data_info,
            train_data_a, train_labels_a, test_data_a, test_labels_a,
            train_data_b, train_labels_b, test_data_b, test_labels_b,
            model_func=vrada_model,
            model_dir="plane-test-models/vrada-models",
            log_dir="plane-test/vrada-logs")
            #model_dir=args.modeldir,
            #log_dir=args.logdir)
    """
