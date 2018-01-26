#!/usr/bin/env python 

import input_data
import tensorflow as tf
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("/home/teric/dev/tf/mnist", one_hot=True)

x = tf.placeholder("float", [None, 784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y_ = tf.placeholder("float", [None,10])

y = tf.nn.softmax(tf.matmul(x,W) + b)
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
#train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)

#init = tf.initialize_all_variables()
init = tf.global_variables_initializer();
sess = tf.Session()
sess.run(init)

fig,ax = plt.subplots(nrows=5,ncols=5,sharex=True,sharey=True,)
for i in range(5):
    for j in range(5):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        img = batch_xs[1].reshape(28,28)
        ax[i%5][j%5].imshow(img,cmap='Greys',interpolation='nearest')

ax[0][0].set_xticks([])
ax[0][0].set_yticks([])
plt.tight_layout()
print "showing sample figures. close it to continue..."
plt.show()

for i in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(1000)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

writer = tf.summary.FileWriter('/home/teric/dev/tf/logdir', sess.graph) 

print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})

writer.close()
sess.close()
