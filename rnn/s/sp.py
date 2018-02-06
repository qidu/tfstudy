import pandas as pd
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import os
import time
import numpy as np

if "DISPLAY" not in os.environ:
    # remove Travis CI Error
    matplotlib.use('Agg')

# import data
data = pd.read_csv('data_stocks.csv')
# dimensions of dataset
n = data.shape[0]
p = data.shape[1]
#
#plt.plot(data['SP500'][0:1000])
#plt.show()
print n,p

#time.sleep(10)

#split training & testing sets
train_start = 0
train_end = int(np.floor(0.9*n))
test_start = train_end + 1
test_end = n
#print np.arange(train_start,train_end)
#print test_start,test_end

# ix:lines
data_train = data.ix[np.arange(train_start, train_end), :]
data_test = data.ix[np.arange(test_start, test_end), :]
#print data_test.columns
#print data_test.dtypes

#scale normalized
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(data_train)
data_train = scaler.transform(data_train)
#scaler.transform(data_train)
data_test = scaler.transform(data_test)
#scaler.transform(data_test)

#build
X_train = data_train[:, 2:]
Y_train = data_train[:, 1]
X_test = data_test[:, 2:]
Y_test = data_test[:, 1]

# Model architecture parameters
n_stocks = 500
n_neurons_1 = 1024
n_neurons_2 = 512
n_neurons_3 = 256
n_neurons_4 = 128
n_neurons_5 = 16
n_target = 1

# Placeholder
X = tf.placeholder(dtype=tf.float32, shape=[None, n_stocks])
Y = tf.placeholder(dtype=tf.float32, shape=[None])

# Initializers
sigma = 1
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
bias_initializer = tf.zeros_initializer()


# Layer 1: Variables for hidden weights and biases
W_hidden_1 = tf.Variable(weight_initializer([n_stocks, n_neurons_1]))
bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
# Layer 2: Variables for hidden weights and biases
W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))
# Layer 3: Variables for hidden weights and biases
W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))
# Layer 4: Variables for hidden weights and biases
W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))
W_hidden_5 = tf.Variable(weight_initializer([n_neurons_4, n_neurons_5]))
bias_hidden_5 = tf.Variable(bias_initializer([n_neurons_5]))
# Output layer: Variables for output weights and biases
W_out = tf.Variable(weight_initializer([n_neurons_5, n_target]))
bias_out = tf.Variable(bias_initializer([n_target]))

# Hidden layer
hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
hidden_3 = tf.nn.elu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
hidden_4 = tf.nn.relu6(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))
hidden_5 = tf.nn.relu(tf.add(tf.matmul(hidden_4, W_hidden_5), bias_hidden_5))

# Output layer (must be transposed)
out = tf.transpose(tf.add(tf.matmul(hidden_5, W_out), bias_out))

# Cost function
#mse = tf.reduce_sum(tf.squared_difference(out, Y))
#mse = tf.reduce_mean(tf.squared_difference(out, Y))
#mse = tf.reduce_sum(tf.where(tf.greater(out,Y),(out-Y)*1,(Y-out)*1))
mse = tf.reduce_mean(tf.where(tf.greater(out,Y),(out-Y)*1,(Y-out)*3/2))
#mse = tf.losses.mean_squared_error(out, Y)
#mse = tf.nn.softmax_cross_entropy_with_logits(out, Y)
opt = tf.train.AdamOptimizer().minimize(mse)

# Make Session
sess = tf.Session()
# Run initializer
sess.run(tf.global_variables_initializer())

# Setup interactive plot
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(111)
line1, = ax1.plot(Y_test)
line2, = ax1.plot(Y_test*0.5)
plt.show()

# Number of epochs and batch size
epochs = 9
batch_size = 512

for e in range(epochs):
    # Shuffle training data
    shuffle_indices = np.random.permutation(np.arange(len(Y_train)))
    #print len(shuffle_indices)
    #print "before ",len(X_train),len(Y_train)
    X_train = X_train[shuffle_indices]
    Y_train = Y_train[shuffle_indices]
    #print "after ",len(X_train),len(Y_train)

    # Minibatch training
    for i in range(0, len(Y_train) // batch_size):
        start = i * batch_size
        batch_x = X_train[start:(start + batch_size)]
        batch_y = Y_train[start:(start + batch_size)]
        #print "batch: ",start,batch_size,len(batch_x)
        #print tf.shape(batch_y.values)

        writer = tf.summary.FileWriter('/home/teric/dev/tf/logdir', sess.graph) 
        # Run optimizer with batch
        sess.run(opt, feed_dict={X: batch_x, Y: batch_y})
        writer.close()
        

        # Show progress
        if np.mod(i, 5) == 0:
            # Prediction
            pred = sess.run(out, feed_dict={X: X_test})
            #pred_mse_diff = (np.sum(Y_test) - np.sum(pred[0]))/len(pred[0])
            #pred_mse_diff = (np.sum(Y_test.values) - np.sum(pred[0]))/len(pred[0])
            pred_mse_diff = sess.run(tf.round(tf.reduce_sum(tf.squared_difference(Y_test, pred[0]))))
            print pred_mse_diff
            line2.set_ydata(pred)
            plt.title('Epoch ' + str(e) + ', Batch ' + str(i) + ' Diff ' + str(pred_mse_diff))
            file_name = 'img/d_' + str(pred_mse_diff) + '_e_' + str(e) + '_ba_' + str(i) + '.png'
            plt.savefig(file_name)
            plt.pause(0.01)
