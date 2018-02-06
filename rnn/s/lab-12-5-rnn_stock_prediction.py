'''
This script shows how to predict stock prices using a basic RNN
'''
import tensorflow as tf
import numpy as np
import matplotlib
import os

tf.set_random_seed(777)  # reproducibility

if "DISPLAY" not in os.environ:
    # remove Travis CI Error
    matplotlib.use('Agg')

import matplotlib.pyplot as plt


def MinMaxScaler(data):
    ''' Min Max Normalization to [0,1]

    Parameters
    ----------
    data : numpy.ndarray
        input data to be normalized
        shape: [Batch size, dimension]

    Returns
    ----------
    data : numpy.ndarry
        normalized data
        shape: [Batch size, dimension]

    References
    ----------
    .. [1] http://sebastianraschka.com/Articles/2014_about_feature_scaling.html

    '''
    numerator = data - np.min(data, 0) # min record
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7) # normanize the diff

def GetReal(nor,upper,lower):
    return (upper - lower) * nor + lower

# train Parameters
seq_length = 7
data_dim = 3 # colum : open,high,low
hidden_dim = 10
output_dim = 1
learning_rate = 0.01
iterations = 500
shift_size = 1 # make Y be the future value

# Open, High, Low, Volume, Close
#xy = np.loadtxt('bd.csv', delimiter=',')
xy = np.loadtxt('data-02-stock_daily.csv', delimiter=',')
xy = xy[::-1]  # reverse order (chronically ordered)
xyn = MinMaxScaler(xy)
x = xyn[:,0:data_dim]
y = xyn[:, [-1]]  # Close as label : the close price
raw_max = np.max(xy, 0)
raw_min = np.min(xy, 0)
x = x[0:len(y)-shift_size]
y = y[shift_size:]
#print len(x),len(y)

# build a dataset
dataX = []
dataY = []
for i in range(0, len(y) - seq_length):
    _x = x[i:i + seq_length]
    _y = y[i + seq_length]  # Next close price
    #print(_x, "==>", _y)
    dataX.append(_x)
    dataY.append(_y)

# train/test data sets split
train_size = int(len(dataY) * 0.7) # change length to train
test_size = len(dataY) - train_size # left for test
trainX, testX = np.array(dataX[0:train_size]), np.array(
    dataX[train_size:len(dataX)])
trainY, testY = np.array(dataY[0:train_size]), np.array(
    dataY[train_size:len(dataY)])

# input place holders
X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
Y = tf.placeholder(tf.float32, [None, 1])

# build a LSTM network
cell = tf.contrib.rnn.BasicLSTMCell(
    num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)

outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

Y_pred = tf.contrib.layers.fully_connected(
    outputs[:, -1], output_dim, activation_fn=None)  # We use the last cell's output

# cost/loss
loss = tf.reduce_sum(tf.square(Y_pred - Y))  # sum of the squares
# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)

# RMSE
targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # Training step
    for i in range(iterations):
        _, step_loss = sess.run([train, loss], feed_dict={
                                X: trainX, Y: trainY})
        #print("[step: {}] loss: {}".format(i, step_loss))

    writer = tf.summary.FileWriter('/home/teric/dev/tf/logdir', sess.graph)
    # Test step
    test_predict = sess.run(Y_pred, feed_dict={X: testX})
    rmse_val = sess.run(rmse, feed_dict={
                    targets: testY, predictions: test_predict})
    print("RMSE: {}".format(rmse_val))
    print len(test_predict)
    writer.close()

    # Plot predictions
    plt.plot(testY)
    plt.plot(test_predict)
    plt.xlabel("Time Period")
    plt.ylabel("Stock Price")
    print "Last: ",testY[-1], test_predict[-1]
    print "Pridict: ",GetReal(testY[-1],raw_max[-1],raw_min[-1]), GetReal(test_predict[-1],raw_max[-1],raw_min[-1])
    #plt.show()
