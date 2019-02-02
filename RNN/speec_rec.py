#!/usr/bin/python3.6
#path
import os
from os.path import isdir, join
from pathlib import Path

# Scientific Math
import numpy as np
from scipy.fftpack import fft
from scipy import signal
from scipy.io import wavfile
from sklearn.model_selection import train_test_split

# Visualization
import tensorflow as tf

#Deep learning
'''
import tensorflow.keras as keras
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from tensorflow.python.keras import Input, layers
from tensorflow.python.keras import backend as K
'''
import random
import copy

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

    del items[-3:]
    del classes[-3:]
    # print(items)
    # print(classes)
    # items = items[:len(items)-3]
    # print(len(items))

readData('../process_audio/features/features-stoyan2.txt')

# vec = tf.constant([1.0, 2.0, 3.0])
# vec2 = tf.constant([1.0])
# multiply = tf.constant([100])
# matrix = tf.reshape(tf.tile(vec, multiply), [multiply[0], 3])
# matrix2 = tf.reshape(tf.tile(vec2, multiply), [multiply[0], 1])

matrix = tf.convert_to_tensor(items, dtype = tf.float32)
matrix2 = tf.convert_to_tensor(classes, dtype = tf.int32)
features_len = 6 # 10

num_epochs = 5000
input_size = matrix.shape[1]
total_series_length = len(items)
truncated_backprop_length = 20
state_size = 4
num_classes = 27 # [0; 27)
echo_step = 3
batch_size = 100
num_batches = total_series_length//batch_size//truncated_backprop_length

def generateData(sess):
    x = sess.run(matrix)
    y = sess.run(matrix2)

    x = x.reshape((batch_size, -1, features_len))  # The first index changing slowest, subseries as rows
    y = y.reshape((batch_size, -1))

    return (x, y)

batchX_placeholder = tf.placeholder(matrix.dtype, [batch_size, truncated_backprop_length, features_len]) # batch_size X truncated
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length]) # batch_size X truncated

init_state = tf.placeholder(tf.float32, [batch_size, state_size])  # batch_size X state_size

W2 = tf.Variable(np.random.rand(state_size, num_classes), dtype = tf.float32) # weights, state_size X num_classes
b2 = tf.Variable(np.zeros((1, num_classes)), dtype = tf.float32) # biases, 1 X num_classes

# Unpack columns
inputs_series = tf.unstack(batchX_placeholder, axis=1) # truncated na broi podtensora [batch_size X 1]
labels_series = tf.unstack(batchY_placeholder, axis = 1) # truncated X batch_size

# Forward passes
cell = tf.nn.rnn_cell.BasicRNNCell(state_size)
print(cell)
print(inputs_series)
print(init_state)
states_series, current_state = tf.nn.static_rnn(cell=cell, inputs=inputs_series, initial_state=init_state)

print(states_series)
for state in states_series:
    print("*", state)
print("-", current_state)
logits_series = [tf.matmul(state, W2) + b2 for state in states_series] # Broadcasted addition
predictions_series = [tf.nn.softmax(logits) for logits in logits_series]

print(logits_series)
losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits_, labels = labels_) for logits_, labels_ in zip(logits_series, labels_series)]
total_loss = tf.reduce_mean(losses)

train_step = tf.train.AdagradOptimizer(0.0001).minimize(total_loss)

def plot(loss_list, predictions_series, batchX, batchY):
    #plt.subplot(2, 3, 1)
    plt.cla()
    plt.plot(loss_list)
    '''
    for batch_series_idx in range(5):
        one_hot_output_series = np.array(predictions_series)[:, batch_series_idx, :]
        single_output_series = np.array([(1 if out[0] < 0.5 else 0) for out in one_hot_output_series])

        plt.subplot(2, 3, batch_series_idx + 2)
        plt.cla()
        plt.axis([0, truncated_backprop_length, 0, 2])
        left_offset = range(truncated_backprop_length)
        #plt.bar(left_offset, batchX[batch_series_idx, :, :][0], width=1, color="blue")
        #plt.bar(left_offset, batchY[batch_series_idx, :] * 0.5, width=1, color="red")
        #plt.bar(left_offset, single_output_series * 0.3, width=1, color="green")

    plt.draw()
    '''
    plt.pause(0.0001)


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    plt.ion()
    plt.figure()
    plt.show()
    loss_list = []

    x, y = generateData(sess)
    for epoch_idx in range(num_epochs):
        _current_state = np.zeros((batch_size, state_size))

        print("New data, epoch", epoch_idx)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * truncated_backprop_length
            end_idx = start_idx + truncated_backprop_length

            batchX = x[:, start_idx:end_idx]
            batchY = y[:, start_idx:end_idx]
            #print(batchX)

            _total_loss, _train_step, _current_state, _predictions_series = sess.run(
                [ total_loss, train_step, current_state, predictions_series ],
                feed_dict={
                    batchX_placeholder: batchX,
                    batchY_placeholder: batchY,
                    init_state: _current_state
                })
            #print(_current_state)
            #print(init_state)
            #print(batchX_placeholder)


            loss_list.append(_total_loss)

            if batch_idx%100 == 0:
                print("Step", batch_idx, "Loss", _total_loss)
                plot(loss_list, _predictions_series, batchX, batchY)

plt.ioff()
plt.show()
