import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

img_size = 28
num_classes = 10
img_shape = (img_size, img_size)
img_size_flat = img_size*img_size
num_channels = 1
batch_size = 128
l2_regulizer = 0.001

filter_size1 = 3
num_filters1 = 128
filter_size2 = 3
num_filters2 = 128
filter_size3 = 3
num_filters3 = 128

filter_size4 = 3
num_filters4 = 256
filter_size5 = 3
num_filters5 = 256
filter_size6 = 3
num_filters6 = 256
filter_size7 = 3
num_filters7 = 256

filter_size8 = 3
num_filters8 = 512
filter_size9 = 3
num_filters9 = 512
filter_size10 = 3
num_filters10 = 512
filter_size11 = 3
num_filters11 = 512
filter_size12 = 3
num_filters12 = 512

filter_size13 = 1
num_filters13 = 1024


fc_size1 = 4096
fc_size2 = 1024


def new_weights(shape, name):
    return tf.get_variable(name=name, initializer=tf.contrib.layers.xavier_initializer(), shape=shape)


def new_biases(length, name):
    return tf.get_variable(name, shape=[length], initializer=tf.zeros_initializer())


def new_conv_layer(input, num_input_channels, filter_size, num_filters, use_pooling=True):
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    s = np.random.randint(0, 100000)
    weights = new_weights(shape, 'conv'+str(s))
    biases = new_biases(num_filters, 'bias'+str(s))
    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
    layer += biases
    if use_pooling:
        layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    layer = tf.nn.relu(layer)
    return layer, weights


def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = np.array(layer_shape[1:4], dtype=int).prod()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features


def new_fc_layer(input, num_inputs, num_outputs, use_relu=True):
    s = np.random.randint(0, 100000)
    weights = new_weights(shape=[num_inputs, num_outputs], name='w'+str(s))
    biases = new_biases(length=num_outputs, name='b'+str(s))
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)
    return layer, weights


x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)


layer_conv1, weights_conv1 = new_conv_layer(input=x_image, num_input_channels=num_channels, filter_size=filter_size1, num_filters=num_filters1, use_pooling=False)

print(layer_conv1)

layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1, num_input_channels=num_filters1, filter_size=filter_size2, num_filters=num_filters2, use_pooling=True)

print(layer_conv2)
dropped1 = tf.nn.dropout(layer_conv2, 0.75)

layer_conv3, weights_conv3 = new_conv_layer(input=dropped1, num_input_channels=num_filters2, filter_size=filter_size3, num_filters=num_filters3, use_pooling=False)

print(layer_conv3)

layer_conv4, weights_conv4 = new_conv_layer(input=layer_conv3, num_input_channels=num_filters3, filter_size=filter_size4, num_filters=num_filters4, use_pooling=False)

print(layer_conv4)
##
layer_conv5, weights_conv5 = new_conv_layer(input=layer_conv4, num_input_channels=num_filters4, filter_size=filter_size5, num_filters=num_filters5, use_pooling=False)

print(layer_conv5)

layer_conv6, weights_conv6 = new_conv_layer(input=layer_conv5, num_input_channels=num_filters5, filter_size=filter_size6, num_filters=num_filters6, use_pooling=True)

print(layer_conv6)
dropped2 = tf.nn.dropout(layer_conv6, 0.75)

layer_conv7, weights_conv7 = new_conv_layer(input=dropped2, num_input_channels=num_filters6, filter_size=filter_size7, num_filters=num_filters7, use_pooling=False)

print(layer_conv7)

layer_conv8, weights_conv8 = new_conv_layer(input=layer_conv7, num_input_channels=num_filters7, filter_size=filter_size8, num_filters=num_filters8, use_pooling=False)

print(layer_conv8)

layer_conv9, weights_conv9 = new_conv_layer(input=layer_conv8, num_input_channels=num_filters8, filter_size=filter_size9, num_filters=num_filters9, use_pooling=False)

print(layer_conv9)

layer_conv10, weights_conv10 = new_conv_layer(input=layer_conv9, num_input_channels=num_filters9, filter_size=filter_size10, num_filters=num_filters10, use_pooling=True)

print(layer_conv10)
dropped3 = tf.nn.dropout(layer_conv10, 0.75)

layer_conv11, weights_conv11 = new_conv_layer(input=dropped3, num_input_channels=num_filters10, filter_size=filter_size11, num_filters=num_filters11, use_pooling=False)


print(layer_conv11)

layer_conv12, weights_conv12 = new_conv_layer(input=layer_conv11, num_input_channels=num_filters11, filter_size=filter_size12, num_filters=num_filters12, use_pooling=False)

print(layer_conv12)

layer_conv13, weights_conv13 = new_conv_layer(input=layer_conv12, num_input_channels=num_filters12, filter_size=filter_size13, num_filters=num_filters13, use_pooling=True)

print(layer_conv13)

layer_flat, num_features = flatten_layer(layer_conv13)

print(layer_flat)
print(num_features)
dropped4 = tf.nn.dropout(layer_flat, 0.75)

layer_fc1, weights14 = new_fc_layer(input=dropped4, num_inputs=num_features, num_outputs=fc_size1, use_relu=True)

print(layer_fc1)
dropped5 = tf.nn.dropout(layer_fc1, 0.75)


layer_fc2, weights15 = new_fc_layer(input=dropped5, num_inputs=fc_size1, num_outputs=fc_size2, use_relu=True)

print(layer_fc2)

layer_fc3, weights16 = new_fc_layer(input=layer_fc2, num_inputs=fc_size2, num_outputs=num_classes, use_relu=False)

print(layer_fc3)

y_pred = tf.nn.softmax(layer_fc3)
y_pred_cls = tf.argmax(y_pred, dimension=1)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc3, labels=y_true)
cost = tf.add(tf.reduce_mean(cross_entropy), tf.multiply(l2_regulizer/batch_size, tf.nn.l2_loss(weights_conv1)+tf.nn.l2_loss(weights_conv2)+tf.nn.l2_loss(weights_conv3)+tf.nn.l2_loss(weights_conv4)+tf.nn.l2_loss(weights_conv5)+tf.nn.l2_loss(weights_conv6)+tf.nn.l2_loss(weights_conv7)+tf.nn.l2_loss(weights_conv8)+tf.nn.l2_loss(weights_conv9)+tf.nn.l2_loss(weights_conv10)+tf.nn.l2_loss(weights_conv11)+tf.nn.l2_loss(weights_conv12)+tf.nn.l2_loss(weights_conv13)+tf.nn.l2_loss(weights14)+tf.nn.l2_loss(weights15)+tf.nn.l2_loss(weights16)))
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session = tf.Session()
session.run(tf.global_variables_initializer())
saver = tf.train.Saver()


def optimize(num_iterations):
    accu = []
    vali = []
    print(num_iterations)
    for i in range(num_iterations):
        print("Optimization Iteration: {0:>6}".format(i+1))
        for j in range(total_batches):
            print(str(j).zfill(len(str(total_batches)))+"/"+str(total_batches))
            index_front = j * batch_size
            index_end = (j + 1) * batch_size if (j + 1) * batch_size < x_train.shape[0] else x_train.shape[0]
            X_batch = x_train[index_front:index_end, :]
            Y_batch = y_train[index_front:index_end, :]
            # print("bitch")
            feed_dict_train = {x: X_batch, y_true: Y_batch}
            session.run(optimizer, feed_dict=feed_dict_train)
        acc = session.run(accuracy, feed_dict=feed_dict_train)
        msg = "Training Accuracy: {0:>6.1%}"
        print(msg.format(acc))
        feed_dict_val = {x: x_val, y_true: y_val}
        val = session.run(accuracy, feed_dict=feed_dict_val)
        msg = "Validation Accuracy: {0:>6.1%}"
        print(msg.format(val))
        accu.append(acc)
        vali.append(val)
        save_path = saver.save(session, "E:/Code/Python/Intern/tmp/model"+str(i+1)+".ckpt")
        print("Model saved in path: %s" % save_path)
        print("====================================================")

    plt.plot(np.arange(0, len(accu)), accu, 'r')
    plt.plot(np.arange(0, len(vali)), vali, 'b')
    plt.show()


def convert_to_one_hot(labels, depth):
    one_hot_matrix = tf.one_hot(labels, depth, axis=0)
    sess = tf.Session()
    one_hot_labels = sess.run(one_hot_matrix)
    sess.close()
    return one_hot_labels


def load_dataset():
    x_train = pd.read_csv('train.csv').as_matrix()
    x_test = np.divide(pd.read_csv('test.csv').as_matrix(), 255)
    y_train = x_train[:, 0].reshape(1, x_train.shape[0])
    print(y_train.shape)
    y_train = np.squeeze(convert_to_one_hot(y_train, int(np.amax(y_train) + 1))).T
    print(y_train.shape)
    x_train = np.divide(x_train[:, 1:], 255)
    validation_images = x_train[0:512, :]
    validation_labels = y_train[0:512, :]
    x_train = x_train[512:, :]
    y_train = y_train[512:, :]
    return x_train, y_train, validation_images, validation_labels, x_test


x_train, y_train, x_val, y_val, x_test = load_dataset()
print(x_train.shape)
print(y_train.shape)
total_batches = np.ceil(x_train.shape[0] / batch_size).astype(np.int32)
optimize(50)
print(x_test.shape)
print(x_train.shape)
predict = session.run(y_pred_cls, feed_dict={x: x_test})
print(predict.shape)
print(predict)
image_id = np.arange(1, len(predict) + 1, 1)
image_id = image_id.reshape(len(predict), 1)
np.savetxt('kosagam.csv', np.c_[image_id, predict], delimiter=',', header='ImageId,Label', comments='', fmt='%d')
