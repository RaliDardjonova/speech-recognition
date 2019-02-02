#!/usr/bin/python3.6
from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

items = []
classes = []

def readData(fileName):
    # Read the file, splitting by lines
    f = open(fileName, 'r')
    lines = f.read().splitlines()
    f.close()
    delimiter = ','

    for i in range(0, len(lines)):
        line = lines[i].split(delimiter)
        itemFeatures = []
        classLabel = ''

        for j in range(len(line)):
            if j == 0:
                classLabel = line[j].lower()
            else:
                if len(line[j]) > 0:
                    v = float(line[j])  # Convert feature value to float
                    itemFeatures.append(v)

        if len(itemFeatures) > 0:
            items.append(itemFeatures)
            classNum = 26 if (classLabel == ' ') else ord(classLabel) - ord('a')
            classes.append(classNum)

    # We adapt it to be divisible by batch_size
    del items[-3:]
    del classes[-3:]

fileNames = [ '../process_audio/features/features-kimche3.txt', '../process_audio/features/features-stoyan3.txt', '../process_audio/features/features-radina3.txt' ]
readData(fileNames[0])

matrix_features = tf.convert_to_tensor(items, dtype = tf.float32) # inputs X features_len
matrix_labels = tf.convert_to_tensor(classes, dtype = tf.int32) # inputs X 1

# Data features
num_classes = 27
total_series_length = len(items)
features_len = matrix_features.shape[1] # TODO: How does it cope with multi-dimensional feature set?

# NN parameters
batch_size = 5
num_epochs = 1000
num_units = 128 # internal neurons in the LSTM cell
timesteps = 50
learning_rate = 0.001
num_batches = total_series_length//batch_size//timesteps

batch_features_placeholder = tf.placeholder(tf.float32, [batch_size, timesteps, features_len]) # batch_size X timesteps X features_len
batch_labels_placeholder = tf.placeholder(tf.int32, [batch_size, timesteps]) # [ batch_size X timesteps ]

cell_state = tf.placeholder(tf.float32, [batch_size, num_units]) # timesteps X [batch_size X num_units]
hidden_state = tf.placeholder(tf.float32, [batch_size, num_units]) # timesteps X [batch_size X num_units]
init_state = tf.nn.rnn_cell.LSTMStateTuple(cell_state, hidden_state) # tuple (cell_state, hidden_state)

# Unpack columns
inputs_series = tf.unstack(batch_features_placeholder, axis = 1) # timesteps X [batch_size X features_len]
labels_series = tf.unstack(batch_labels_placeholder, axis = 1) # timesteps X batch_size

# Passes
cell = tf.nn.rnn_cell.LSTMCell(num_units = num_units)
output_series, current_state = tf.nn.static_rnn(cell = cell, inputs = inputs_series, initial_state = init_state) # output_series: timesteps X [batch_size X num_units]
                                                                                                                 # current_state: # [batch_size X num_units]

weights = tf.Variable(np.random.rand(num_units, num_classes), dtype = tf.float32) # [ num_units X num_classes ]
biases = tf.Variable(np.zeros((1, num_classes)), dtype = tf.float32) # [ 1 X num_classes ]

logits_series = [tf.matmul(output, weights) + biases for output in output_series] # every output is [batch_size X num_units] -> logits series is: timesteps X [batch_size X num_units]
predictions_series = [tf.nn.softmax(logits) for logits in logits_series] # every softmax(logits) is [ batch_size X num_classes ] -> prediction_series is: timesteps X [batch_size X num_classes]

losses = [ tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits_, labels = labels_) for logits_, labels_ in zip(logits_series, labels_series) ] # timesteps X [batch_size X num_classes]
total_loss = tf.reduce_mean(losses)
train_step = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(total_loss)

def plot(loss_list, predictions_series, batchX, batchY):
    plt.subplot(2, 3, 1)
    plt.cla()
    plt.plot(loss_list)

    plt.draw()
    plt.pause(0.0001)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    loss_list = []
    features = sess.run(matrix_features)
    labels = sess.run(matrix_labels)
    features = features.reshape((batch_size, -1, features_len))
    labels = labels.reshape((batch_size, -1))

    plt.ion()
    plt.figure()
    plt.show()

    # initialisation
    _current_cell_state = np.zeros((batch_size, num_units))
    _current_hidden_state = np.zeros((batch_size, num_units))

    for epoch_idx in range(num_epochs):

        print("New data, epoch", epoch_idx)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * timesteps
            end_idx = start_idx + timesteps

            cur_batch_features = features[:, start_idx:end_idx, :]
            cur_batch_labels = labels[:, start_idx:end_idx]

            _total_loss, _train_step, _current_state, _predictions_series = sess.run(
                [ total_loss, train_step, current_state, predictions_series ],
                feed_dict = {
                    batch_features_placeholder: cur_batch_features,
                    batch_labels_placeholder: cur_batch_labels,
                    cell_state: _current_cell_state,
                    hidden_state: _current_hidden_state
                }) # TODO: how to save current state in order to continue from it?

            _current_cell_state, _current_hidden_state = _current_state
            loss_list.append(_total_loss)

            if batch_idx % 100 == 0:
                # model evaluation
                pred_series = tf.unstack(tf.convert_to_tensor(np.array(_predictions_series)), axis = 1)
                correct_prediction = tf.equal(tf.argmax(pred_series, axis = 2), cur_batch_labels[:, start_idx:end_idx])
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                _accuracy = sess.run(accuracy)

                print("Step", batch_idx, "Loss", _total_loss)
                print("Accuracy", _accuracy)
                plot(loss_list, _predictions_series, cur_batch_features, cur_batch_labels)

plt.ioff()
plt.show()
