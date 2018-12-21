import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

def convert_to_one_hot(labels, depth):
    one_hot_matrix = tf.one_hot(labels, depth, axis=0)
    sess = tf.Session()
    one_hot_labels = sess.run(one_hot_matrix)
    sess.close()
    return one_hot_labels

def load_dataset():
    x_train = pd.read_csv('train.csv').as_matrix()
    x_test = np.divide(pd.read_csv('test.csv').as_matrix(), 255).T
    y_train = x_train[:, 0].reshape(1, x_train.shape[0])
    y_train = np.squeeze(convert_to_one_hot(y_train, int(np.amax(y_train) + 1)))
    x_train = np.divide(x_train[:, 1:], 255).T
    validation_images = x_train[:, 0:1024]
    validation_labels = y_train[:, 0:1024]
    x_train = x_train[:, 1024:]
    y_train = y_train[:, 1024:]
    return x_train, y_train, validation_images, validation_labels, x_test


def compute_cost(Z, Y, W1, W2, W3, batch_size):
    logits = tf.transpose(Z)
    labels = tf.transpose(Y)
    unregularized_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    L2_regularization = tf.multiply(L2_lambda / batch_size, tf.add(tf.add(tf.nn.l2_loss(W1), tf.nn.l2_loss(W2)), tf.nn.l2_loss(W3)))
    cost = tf.add(unregularized_cost, L2_regularization)
    return cost


def forward_propagation(X, W1, b1, W2, b2, W3, b3):
    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)
    return Z3


def initialize_parameters(n_in, n_1, n_2, n_out):
    W1 = tf.get_variable('W1', (n_1, n_in), initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable('b1', (n_1, 1), initializer=tf.zeros_initializer())
    W2 = tf.get_variable('W2', (n_2, n_1), initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable('b2', (n_2, 1), initializer=tf.zeros_initializer())
    W3 = tf.get_variable('W3', (n_out, n_2), initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable('b3', (n_out, 1), initializer=tf.zeros_initializer())
    return W1, b1, W2, b2, W3, b3


train_images, train_labels, validation_images, validation_labels, test_images = load_dataset()

learning_rate = 0.001
L2_lambda = 0.05
epoch = 50
batchsize = 256
n_in = train_images.shape[0]
n_1 = 256
n_2 = 128
n_out = train_labels.shape[0]
m = train_images.shape[1]
m_test = test_images.shape[0]
total_batches = np.ceil(m / batchsize).astype(np.int32)

X = tf.placeholder(tf.float32, (n_in, None), name='X')
Y = tf.placeholder(tf.float32, (n_out, None), name='Y')

W1, b1, W2, b2, W3, b3 = initialize_parameters(n_in, n_1, n_2, n_out)

Z3 = forward_propagation(X, W1, b1, W2, b2, W3, b3)

cost = compute_cost(Z3, Y, W1, W2, W3, batchsize)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(Z3), tf.argmax(Y))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

initialize = tf.global_variables_initializer()

costs = []
costs_val = []
accuracies = []
accuracies_val = []
session = tf.Session()
session.run(initialize)
for i in range(epoch):
    epoch_cost = 0.
    for j in range(total_batches):
        index_front = j * batchsize
        index_end = (j + 1) * batchsize if (j + 1) * batchsize < m else m
        X_batch = train_images[:, index_front:index_end]
        Y_batch = train_labels[:, index_front:index_end]
        _, batch_cost = session.run([optimizer, cost], feed_dict={X: X_batch, Y: Y_batch})
        epoch_cost += batch_cost / total_batches
    epoch_cost_val = session.run(cost, feed_dict={X: validation_images, Y: validation_labels})
    epoch_acc = session.run(accuracy, feed_dict={X: train_images, Y: train_labels})
    epoch_acc_val = session.run(accuracy, feed_dict={X: validation_images, Y: validation_labels})
    print('Epoch ' + str(i + 1) + ' Training Cost/Validation Cost/Training Accuracy/Validation Accuracy: ' + str(epoch_cost) + '/' + str(epoch_cost_val) + '/' + str(epoch_acc) + '/' + str(epoch_acc_val))
    costs.append(epoch_cost)
    costs_val.append(epoch_cost_val)
    accuracies.append(epoch_acc)
    accuracies_val.append(epoch_acc_val)

plt.plot(np.squeeze(costs), label='Training Cost')
plt.plot(np.squeeze(costs_val), label='Validation Cost')
plt.ylabel('Cost')
plt.xlabel('Epoch')
plt.title('Cost - Epoch')
plt.legend(loc='best')
plt.show()

plt.plot(np.squeeze(accuracies), label='Training Accuracy')
plt.plot(np.squeeze(accuracies_val), label='Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.title('Cost - Accuracy')
plt.legend(loc='best')
plt.show()

Z3_test = forward_propagation(X, W1, b1, W2, b2, W3, b3)
A3_test = tf.nn.softmax(Z3_test, dim=0)
predict = np.argmax(session.run(A3_test, feed_dict={X: test_images}), axis=0)
predict = predict.reshape(predict.shape[0], 1)
image_id = np.arange(1, len(predict) + 1, 1)
image_id = image_id.reshape(len(predict), 1)
np.savetxt('amato.csv', np.c_[image_id, predict], delimiter=',', header='ImageId,Label', comments='', fmt='%d')

sess.close()
