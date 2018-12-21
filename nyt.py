import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

img_size = 28
num_classes = 10
img_shape = (img_size, img_size)
img_size_flat = img_size*img_size
num_channels = 1
batch_size = 64
learning_rate = 1e-4
l2_regulizer = 0.001

filter_size1 = 5
num_filters1 = 32

filter_size2 = 5
num_filters2 = 32

filter_size3 = 3
num_filters3 = 64

filter_size4 = 3
num_filters4 = 64

fc_size = 256


def new_weights(shape, name):
    return tf.get_variable(name=name, initializer=tf.contrib.layers.xavier_initializer(), shape=shape)


def new_biases(length, name):
    return tf.get_variable(name, shape=[length])


def new_conv_layer(input, num_input_channels, filter_size, num_filters, use_pooling=True):
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    s = np.random.randint(0, 100)
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


def new_fc_layer(input, num_inputs, num_outputs, use_relu=True, tanh=False):
    s = np.random.randint(0, 100)
    weights = new_weights(shape=[num_inputs, num_outputs], name='w'+str(s))
    biases = new_biases(length=num_outputs, name='b'+str(s))
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)
    elif tanh:
        layer = tf.nn.tanh(layer)
    return layer, weights


x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)


layer_conv1, weights_conv1 = new_conv_layer(input=x_image, num_input_channels=num_channels, filter_size=filter_size1, num_filters=num_filters1, use_pooling=False)

print(layer_conv1)

layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1, num_input_channels=num_filters1, filter_size=filter_size2, num_filters=num_filters2, use_pooling=True)

print(layer_conv2)

dropped = tf.nn.dropout(layer_conv2, 0.75)

print(dropped)

layer_conv3, weights_conv3 = new_conv_layer(input=dropped, num_input_channels=num_filters2, filter_size=filter_size3, num_filters=num_filters3, use_pooling=False)

print(layer_conv3)

layer_conv4, weights_conv4 = new_conv_layer(input=layer_conv3, num_input_channels=num_filters3, filter_size=filter_size4, num_filters=num_filters4, use_pooling=True)

print(layer_conv4)

layer_flat, num_features = flatten_layer(layer_conv4)

print(layer_flat)
print(num_features)

layer_fc1, weights3 = new_fc_layer(input=layer_flat, num_inputs=num_features, num_outputs=fc_size, use_relu=True)

print(layer_fc1)

dropped = tf.nn.dropout(layer_fc1, 0.75)

layer_fc2, weights4 = new_fc_layer(input=dropped, num_inputs=fc_size, num_outputs=num_classes, use_relu=False)

print(layer_fc2)

y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, dimension=1)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session = tf.Session()
session.run(tf.global_variables_initializer())

saver = tf.train.Saver()


def optimize(num_iterations, learning_rate):
    accu = []
    vali = []
    l = 0
    loss_train = []
    loss_val = []
    for i in range(num_iterations):
        print("Optimization Iteration: {0:>6}".format(i+1))
        save_path = saver.save(session, "E:/Code/Python/Intern/tmp/model"+str(i+1)+".ckpt")
        print("Model saved in path: %s" % save_path)
        epoch_cost = 0
        if i % 10 is 0 and i is not 0:
            learning_rate /= 2
        for j in range(total_batches):
            if j % 100 is 0:
                print(str(j).zfill(len(str(total_batches)))+"/"+str(total_batches))
                print("=========================================")
            index_front = j * batch_size
            index_end = (j + 1) * batch_size if (j + 1) * batch_size < x_train.shape[0] else x_train.shape[0]
            X_batch = x_train[index_front:index_end, :]
            Y_batch = y_train[index_front:index_end, :]
            # print("bitch")
            feed_dict_train = {x: X_batch, y_true: Y_batch}
            _, batch_cost = session.run([optimizer, cost], feed_dict=feed_dict_train)
            epoch_cost += batch_cost / total_batches

        acc = session.run(accuracy, feed_dict=feed_dict_train)
        msg = "Training Accuracy: {0:>6.1%}, Training cost:{1}"
        print(msg.format(acc, epoch_cost))
        feed_dict_val = {x: x_val, y_true: y_val}
        val = session.run(accuracy, feed_dict=feed_dict_val)
        epoch_cost_val = session.run(cost, feed_dict=feed_dict_val)
        msg = "Validation Accuracy: {0:>6.1%}, Validation cost:{1}"
        print(msg.format(val, epoch_cost_val))
        loss_train.append(epoch_cost)
        loss_val.append(epoch_cost_val)
        accu.append(acc)
        vali.append(val)
        print(val)
        if val >= 0.992 and l is 0:
            learning_rate /= 2
            l += 1
        if val >= 0.999:
            break
    a, = plt.plot(np.arange(0, len(accu)), accu, 'r', label="train accuracy")
    b, = plt.plot(np.arange(0, len(vali)), vali, 'b', label="validation accuracy")
    c, = plt.plot(np.arange(0, len(loss_train)), loss_train, 'r', label="train cost")
    d, = plt.plot(np.arange(0, len(loss_val)), loss_val, 'b', label="validation cost")
    plt.legend(handles=[a, b, c, d])
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
    validation_images = x_train[0:1024, :]
    validation_labels = y_train[0:1024, :]
    x_train = x_train[1024:, :]
    y_train = y_train[1024:, :]
    return x_train, y_train, validation_images, validation_labels, x_test


x_train, y_train, x_val, y_val, x_test = load_dataset()


def rotate_images(X_imgs, start_angle, end_angle, n_images):
    X_rotate = []
    iterate_at = (end_angle - start_angle) / (n_images - 1)

    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=(None, img_size, img_size, num_channels))
    radian = tf.placeholder(tf.float32, shape=(len(X_imgs)))
    tf_img = tf.contrib.image.rotate(X, radian)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for index in range(n_images):
            degrees_angle = start_angle + index * iterate_at
            radian_value = degrees_angle * math.pi / 180  # Convert to radian
            radian_arr = [radian_value] * len(X_imgs)
            rotated_imgs = sess.run(tf_img, feed_dict={X: X_imgs, radian: radian_arr})
            X_rotate.extend(rotated_imgs)

    X_rotate = np.array(X_rotate, dtype=np.float32)
    return X_rotate


# Start rotation at -90 degrees, end at 90 degrees and produce totally 14 images

def get_translate_parameters(index):
    if index == 0:  # Translate left 20 percent
        offset = np.array([0.0, 0.1], dtype=np.float32)
        size = np.array([img_size, math.ceil(0.9 * img_size)], dtype=np.int32)
        w_start = 0
        w_end = int(math.ceil(0.9 * img_size))
        h_start = 0
        h_end = img_size
    elif index == 1:  # Translate right 20 percent
        offset = np.array([0.0, -0.1], dtype=np.float32)
        size = np.array([img_size, math.ceil(0.9 * img_size)], dtype=np.int32)
        w_start = int(math.floor((1 - 0.9) * img_size))
        w_end = img_size
        h_start = 0
        h_end = img_size
    elif index == 2:  # Translate top 20 percent
        offset = np.array([0.1, 0.0], dtype=np.float32)
        size = np.array([math.ceil(0.9 * img_size), img_size], dtype=np.int32)
        w_start = 0
        w_end = img_size
        h_start = 0
        h_end = int(math.ceil(0.9 * img_size))
    else:  # Translate bottom 20 percent
        offset = np.array([-0.1, 0.0], dtype=np.float32)
        size = np.array([np.ceil(0.9 * img_size), img_size], dtype=np.int32)
        w_start = 0
        w_end = img_size
        h_start = int(math.floor((1 - 0.9) * img_size))
        h_end = img_size

    return offset, size, w_start, w_end, h_start, h_end


def translate_images(X_imgs):
    offsets = np.zeros((len(X_imgs), 2), dtype=np.float32)
    n_translations = 4
    X_translated_arr = []

    tf.reset_default_graph()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(n_translations):
            X_translated = np.zeros((len(X_imgs), img_size, img_size, num_channels),
                                    dtype=np.float32)
            X_translated.fill(1.0)  # Filling background color
            base_offset, size, w_start, w_end, h_start, h_end = get_translate_parameters(i)
            offsets[:, :] = base_offset
            glimpses = tf.image.extract_glimpse(X_imgs, size, offsets)

            glimpses = sess.run(glimpses)
            X_translated[:, h_start: h_start + size[0], \
            w_start: w_start + size[1], :] = glimpses
            X_translated_arr.extend(X_translated)
    X_translated_arr = np.array(X_translated_arr, dtype=np.float32)
    return X_translated_arr


translated_imgs = translate_images(x_train.reshape((x_train.shape[0], img_size, img_size, num_channels)))
print("translated")
print(translated_imgs.shape)
rotated_imgs = rotate_images(x_train.reshape((x_train.shape[0], img_size, img_size, num_channels)), -5, 10, 5)
print(rotated_imgs.shape)
rotated_imgs = np.reshape(rotated_imgs, (rotated_imgs.shape[0], img_size_flat))
translated_imgs = np.reshape(translated_imgs, (translated_imgs.shape[0], img_size_flat))
print(rotated_imgs.shape)
print(x_train.shape)
x_train = np.concatenate((x_train, rotated_imgs, translated_imgs))
print(x_train.shape)
print("================================")
y_trainy = y_train
while not(y_train.shape[0] == x_train.shape[0]):
    print(y_train.shape)
    print(x_train.shape)
    print(x_train.shape[0])
    print(y_train.shape[0])
    print("======================")
    print("**********************")
    y_train = np.concatenate((y_train, y_trainy))
print(x_train.shape)
print(y_train.shape)
total_batches = np.ceil(x_train.shape[0] / batch_size).astype(np.int32)
optimize(30, learning_rate)
print(x_test.shape)
print(x_train.shape)
n = []
n = np.array(n)
for i in range(np.ceil(x_test.shape[0] / batch_size).astype(np.int32)):
    index_front = i * batch_size
    index_end = (i + 1) * batch_size if (i + 1) * batch_size < x_test.shape[0] else x_test.shape[0]
    X_batch = x_test[index_front:index_end, :]
    predict = session.run(y_pred_cls, feed_dict={x: X_batch})
    n = np.concatenate((n, predict))
# 0.998
print(n)
print(n.shape)
print(predict.shape)
print(predict)
image_id = np.arange(1, len(n) + 1, 1)
image_id = image_id.reshape(len(n), 1)
np.savetxt('kosagam.csv', np.c_[image_id, n], delimiter=',', header='ImageId,Label', comments='', fmt='%d')
print("done")
