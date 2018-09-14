#!/usr/bin/env python
#
# VRNN
#
import os
import re
import pathlib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K

# Load a time-series dataset. This is set up to load data in the format of the
# UCR time-series datasets (http://www.cs.ucr.edu/~eamonn/time_series_data/).
# Or, see the generate_trivial_datasets.py for a trivial dataset. 
def load_data(fn):
    """
    Load CSV files in UCR time-series data format
    
    Returns:
        data - numpy array with data of shape (num_examples, num_features)
        labels - numpy array with labels of shape: (num_examples, 1)
    """
    df = pd.read_csv(fn, header=None)
    df_data = df.drop(0, axis=1).values.astype(np.float32)
    df_labels = df.loc[:, df.columns == 0].values.astype(np.uint8)
    return df_data, df_labels

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


# Train and test.
def get_dataset(features, labels, num_classes, batch_size, evaluation=False, buffer_size=5000):
    """
    Get the dataset object for feeding into the model
    
    If batch_size==None, then one-hot encode but don't batch (evaluation)
    If batch_size!=None, then repeat, shuffle, and batch (training)
    """
    def map_func(x, y):
        """ One-hot encode y, convert to appropriate data types """
        x_out = tf.cast(tf.expand_dims(x,axis=1), tf.float32)
        y_out = tf.one_hot(tf.squeeze(tf.cast(y, tf.uint8)) - 1, depth=num_classes)
        return [x_out, y_out]
    
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.map(map_func)
    
    if evaluation:
        dataset = dataset.batch(batch_size)
    else:
        dataset = dataset.repeat().shuffle(buffer_size).batch(batch_size)
    
    return dataset

def get_model(time_steps, num_features, num_classes, layer_type):
    """ Define RNN model """
    if layer_type == 'lstm':
        layer1 = tf.keras.layers.LSTM(128, return_sequences=True)
        layer2 = tf.keras.layers.LSTM(128, return_sequences=False)
    elif layer_type == 'rnn':
        layer1 = tf.keras.layers.SimpleRNN(128, return_sequences=True)
        layer2 = tf.keras.layers.SimpleRNN(128, return_sequences=False)
    elif layer_type == 'custom_rnn':
        layer1 = tf.keras.layers.RNN(MinimalRNNCell(128), return_sequences=True)
        layer2 = tf.keras.layers.RNN(MinimalRNNCell(128), return_sequences=False)
    elif layer_type == 'vrnn':
        layer1 = tf.keras.layers.RNN(VRNNCell(num_features, 64, 10),
                                     return_sequences=False, return_state=True)
        # Note: loss when stacking these gets weird.... we're generating an intermediate
        # value meaning we don't have groundtruth for computing the loss. Thus, we'll only
        # have one layer for VRNN and the output is one-hot encoded.
    elif layer_type == 'gru':
        layer1 = tf.keras.layers.GRU(128, return_sequences=True)
        layer2 = tf.keras.layers.GRU(128, return_sequences=False)
    
    x = tf.keras.Input((time_steps,1), dtype=tf.float32)
    
    if layer_type == 'vrnn':
        n, _, _, *vrnn_state = layer1(x)
        #n = tf.keras.layers.Dropout(0.2)(n)
        n = tf.keras.layers.Dense(num_classes)(n)
        y = tf.keras.layers.Activation('softmax')(n)
        model = tf.keras.Model(x, y)
        
        model.compile(loss=vrnn_loss(*vrnn_state, x),
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=['accuracy'])
    else:
        n = layer1(x)
        n = tf.keras.layers.Dropout(0.5)(n)
        n = layer2(n)
        n = tf.keras.layers.Dropout(0.5)(n)
        n = tf.keras.layers.Dense(num_classes)(n)
        y = tf.keras.layers.Activation('softmax')(n)
        model = tf.keras.Model(x, y)
        
        model.compile(loss=tf.keras.losses.categorical_crossentropy,
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=['accuracy'])
    
    return model

def latest_checkpoint(model_file):
    """
    Find latest checkpoint
    https://www.tensorflow.org/tutorials/keras/save_and_restore_models
    """
    model_path = os.path.dirname(model_file)
    checkpoints = pathlib.Path(model_path).glob("*.hdf5")
    checkpoints = sorted(checkpoints, key=lambda cp:cp.stat().st_mtime)
    checkpoints = [cp.with_suffix('.hdf5') for cp in checkpoints]
    
    if len(checkpoints) > 0:
        # Get epoch number from filename
        regex = re.compile(r'\d\d+')
        numbers = [int(x) for x in regex.findall(str(checkpoints[-1]))]
        assert len(numbers) == 1, "Could not determine epoch number from filename since multiple numbers"
        epoch = numbers[0]
        
        return str(checkpoints[-1]), epoch
    
    return None, None

def train(data_info, features, labels,
          batch_size=64,
          num_epochs=5,
          model_file="models/{epoch:04d}.hdf5",
          log_dir="logs",
          layer_type="lstm"):
    
    model_path = os.path.dirname(model_file)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    latest, epoch = latest_checkpoint(model_file)

    # Data stats
    time_steps, num_features, num_classes = data_info

    # Get dataset / model
    dataset = get_dataset(features, labels, num_classes, batch_size)
    
    # Load previous weights if found, if not we'll start at epoch 0
    if latest is not None:
        # Load the entire saved model
        #model = tf.keras.models.load_model(latest, custom_objects={
        #    'MinimalRNNCell': MinimalRNNCell,
        #    'VRNNCell': VRNNCell,
        #})
        
        # Alternatively, recreate model and load only the weights from the model file
        model = get_model(time_steps, num_features, num_classes, layer_type)
        model.load_weights(latest)
    else:
        model = get_model(time_steps, num_features, num_classes, layer_type)
        epoch = 0
    
    # Train
    model.fit(dataset, initial_epoch=epoch, epochs=num_epochs, steps_per_epoch=30, callbacks=[
        # save_weights_only doesn't work for LSTM apparently, the just-trained model.get_weights()
        # don't show up in the model-from-saved-file model.get_weights(), though some are loaded
        # like the last dense layer. This is a saving problem definitely since saving the entire
        # model and then loading just the weights works fine.
        tf.keras.callbacks.ModelCheckpoint(model_file, period=1, verbose=0),
        tf.keras.callbacks.TensorBoard(log_dir),
        tf.keras.callbacks.TerminateOnNaN()
    ])
    
    return model

def evaluate(data_info, features, labels, model=None,
             model_file="models/{epoch:04d}.hdf5",
             layer_type="lstm",
             useTensorFlowDataset=False):
    
    latest, epoch = latest_checkpoint(model_file)
    
    # Data stats
    time_steps, num_features, num_classes = data_info
    
    # Get dataset
    if useTensorFlowDataset:
        dataset = get_dataset(features, labels, num_classes, 1, evaluation=True)
    else:
        x = np.expand_dims(features,axis=2).astype(np.float32)
        y = np.eye(num_classes)[np.squeeze(labels).astype(np.uint8) - 1] # one-hot encode
    
    # Load weights from last checkpoint if model is not given
    if model is None:
        assert latest is not None, "No latest checkpoint to use for evaluation"
        print("Loading model from", latest, "at epoch", epoch)
        
        # Load entire model
        #model = tf.keras.models.load_model(latest, custom_objects={
        #    'MinimalRNNCell': MinimalRNNCell,
        #    'VRNNCell': VRNNCell,
        #})
        
        # Alternatively, recreate model and load only the weights from the model file
        model = get_model(time_steps, num_features, num_classes, layer_type)
        model.load_weights(latest)
    
    # Evaluate
    if useTensorFlowDataset:
        loss, acc = model.evaluate(dataset, steps=len(labels))
    else:
        loss, acc = model.evaluate(x, y)
    
    return acc

if __name__ == '__main__':
    train_data, train_labels = load_data("trivial/positive_slope_TRAIN")
    test_data, test_labels = load_data("trivial/positive_slope_TEST")

    # Information about dataset
    num_features = 1
    time_steps = train_data.shape[1]
    num_classes = len(np.unique(train_labels))
    data_info = (time_steps, num_features, num_classes)

    for layer_type in ['lstm', 'rnn', 'gru', 'custom_rnn', 'vrnn']:
        print("Training model:", layer_type)
        tf.reset_default_graph()
        K.clear_session()
        model = train(data_info, train_data, train_labels,
                      model_file=layer_type+"-models/{epoch:04d}.hdf5",
                      log_dir=layer_type+"-logs", layer_type=layer_type)

    for layer_type in ['lstm', 'rnn', 'gru', 'custom_rnn', 'vrnn']:
        print("Evaluating model:", layer_type)
        print("  Train:", evaluate(data_info, train_data, train_labels,
                                   model_file=layer_type+"-models/{epoch:04d}.hdf5",
                                   layer_type=layer_type))
        print("  Test:", evaluate(data_info, test_data, test_labels,
                                  model_file=layer_type+"-models/{epoch:04d}.hdf5",
                                  layer_type=layer_type))
