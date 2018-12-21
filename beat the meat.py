import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
import cv2 as cv
from skimage import feature


image_size = 28
number_class = 10
image_shape = (image_size, image_size)
image_size_flat = image_size*image_size


def read_data():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    label = train['label']
    train = train.drop("label", axis=1)
    # print(train.shape, label.shape)
    return label, train, test


def random_batch_normal(train_data, permutation, start=0, completed=False, batch_size=256):
    class_train_data, image_train_data = train_data
    if start==0:
        np.random.shuffle(permutation)
    # print(image_train_data.shape)
    # print(class_train_data.shape)
    # print(permutation)
    # im = []
    # for i in permutation:
    #     im.append(image_train_data.iloc[i, :])
    # _image = np.array(im)
    # print(_image.shape)
    _image = image_train_data.iloc[permutation, :]
    _class = class_train_data[permutation]

    if start+batch_size > len(class_train_data):
        completed = True
        train_batch = _image[start:], _class[start:]
    else:
        train_batch = _image[start:start + batch_size], _class[start:start+batch_size]
    return train_batch, start+batch_size + 1, completed, permutation


def random_batch_pca(train_data, permutation, start=0, completed=False, batch_size=1024):
    class_train_data, image_train_data = train_data
    if start==0:
        np.random.shuffle(permutation)
    # print(image_train_data.shape)
    # print(class_train_data.shape)
    # print(permutation)
    # im = []
    # for i in permutation:
    #     im.append(image_train_data.iloc[i, :])
    # _image = np.array(im)
    # print(_image.shape)
    _image = image_train_data[permutation, :]
    _class = class_train_data[permutation]

    if start+batch_size > len(class_train_data):
        completed = True
        train_batch = _image[start:], _class[start:]
    else:
        train_batch = _image[start:start + batch_size], _class[start:start+batch_size]
    return train_batch, start+batch_size + 1, completed, permutation


def weights(shape, mu=0, std=0.05):
    return tf.Variable(tf.truncated_normal(shape, mu, std), dtype=tf.float32)


def biases(length, value=0):
    return tf.Variable(tf.constant(value, dtype=tf.float32, shape=[length]), dtype=tf.float32)


def new_fc_layer(input, num_inputs, num_outputs, activation="no"):
    Wi = weights(shape=[num_inputs, num_outputs])
    Bi = biases(length=num_outputs)
    layer = tf.matmul(input, Wi) + Bi
    if activation=="softmax":
        layer = tf.nn.softmax(input)
    elif activation=="relu":
        layer = tf.nn.relu(layer)
    elif activation=="sigmoid":
        layer = tf.nn.sigmoid(layer)
    elif activation=="tanh":
        layer = tf.nn.tanh(layer)
    elif activation=="leaky":
        layer = tf.nn.leaky_relu(layer)
    return layer


def model_pixel(x_train, y_train, image_flatten_size, number_classes, layer_size=[512, 256, 128, 64, 10, 10], epoch=2, learning_rate=0.75, activation_list=["sigmoid", "leaky", "signmoid", "relu"]):
    x = tf.placeholder(dtype=tf.float32, name="x", shape=[None, image_flatten_size])
    y_true = tf.placeholder(tf.float32, [None, number_classes])
    layer1 = new_fc_layer(input=x, num_inputs=image_flatten_size, num_outputs=layer_size[0], activation="tanh")
    layer = []
    layer.append(layer1)
    for i in range(1, len(layer_size)):
        if i is not len(layer_size)-1:
            layer.append(new_fc_layer(input=layer[i-1], num_inputs=layer_size[i-1], num_outputs=layer_size[i], activation=activation_list[i-1]))
        else:
            layer.append(new_fc_layer(input=layer[i-1], num_inputs=layer_size[i-1], num_outputs=layer_size[i], activation="softmax"))

    # layer2 = new_fc_layer(input=layer1, num_inputs=layer_size[0], num_outputs=layer_size[1], activation="sigmoid")
    # layer3 = new_fc_layer(input=layer2, num_inputs=layer_size[1], num_outputs=layer_size[2], activation="sigmoid")
    # layer5 = new_fc_layer(input=layer3, num_inputs=layer_size[2], num_outputs=layer_size[3], activation="relu")
    # layer4 = new_fc_layer(input=layer5, num_inputs=layer_size[3], num_outputs=layer_size[4], activation="softmax")
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer[len(layer)-2], labels=y_true)
    cost = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    correct_prediction = tf.equal(layer[len(layer)-1], y_true)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def print_accuracy(feed):
        acc = session.run(accuracy, feed_dict=feed)
        print("Accuracy on train-set: {0:.3%}".format(acc))
        return acc

    def optimize(num_iterations, learning_rate):
        acc = []
        for i in range(num_iterations):
            print("===============================================")
            print("epoch ", i+1)
            print("learning_rate", learning_rate)
            print("===============================================")
            completed = False
            start = 0
            permutation = np.arange(len(y_train))
            while not completed:
                train_batch, start, completed, permutation = random_batch_normal((y_train, x_train), permutation, start, completed)
                x_img, y_class = train_batch
                # print(x_img.shape)
                # print(y_class.shape)
                # print(y_class[0])
                feed_dict_train = {x: x_img, y_true: y_class}
                # print(session.run(layer4, feed_dict=feed_dict_train)[0])
                session.run(optimizer, feed_dict=feed_dict_train)
                acc.append(print_accuracy(feed_dict_train))
            learning_rate /= 2
        plt.plot(np.arange(0, len(acc)), acc)
        plt.show()
    optimize(epoch, learning_rate)
    session.close()


def model_hog(x_train, y_train, image_flatten_size, number_classes, layer_size=[512, 256, 128, 64, 32, 10], epoch=512,
                learning_rate=0.875, activation_list=["sigmoid", "leaky", "signmoid", "relu"]):
    x = tf.placeholder(dtype=tf.float32, name="x", shape=[None, image_flatten_size])
    y_true = tf.placeholder(tf.float32, [None, number_classes])
    layer1 = new_fc_layer(input=x, num_inputs=image_flatten_size, num_outputs=layer_size[0], activation="sigmoid")
    layer = []
    layer.append(layer1)
    for i in range(1, len(layer_size)):
        if i is not len(layer_size) - 1:
            layer.append(new_fc_layer(input=layer[i - 1], num_inputs=layer_size[i - 1], num_outputs=layer_size[i],
                                      activation=activation_list[i - 1]))
        else:
            layer.append(new_fc_layer(input=layer[i - 1], num_inputs=layer_size[i - 1], num_outputs=layer_size[i],
                                      activation="softmax"))

    # layer2 = new_fc_layer(input=layer1, num_inputs=layer_size[0], num_outputs=layer_size[1], activation="sigmoid")
    # layer3 = new_fc_layer(input=layer2, num_inputs=layer_size[1], num_outputs=layer_size[2], activation="sigmoid")
    # layer5 = new_fc_layer(input=layer3, num_inputs=layer_size[2], num_outputs=layer_size[3], activation="relu")
    # layer4 = new_fc_layer(input=layer5, num_inputs=layer_size[3], num_outputs=layer_size[4], activation="softmax")
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer[len(layer) - 1], labels=y_true)
    cost = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    correct_prediction = tf.equal(layer[len(layer) - 1], y_true)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def print_accuracy(feed):
        # print(session.run((layer[len(layer) - 1]), feed_dict=feed))
        # print(session.run((y_true), feed_dict=feed))

        acc = session.run(accuracy, feed_dict=feed)
        print("Accuracy on train-set: {0:.3%}".format(acc))
        return acc

    def optimize(num_iterations, learning_rate):
        acc = []
        for i in range(num_iterations):
            print("===============================================")
            print("epoch ", i + 1)
            print("learning_rate", learning_rate)
            print("===============================================")
            completed = False
            start = 0
            permutation = np.arange(len(y_train))
            while not completed:
                train_batch, start, completed, permutation = random_batch_pca((y_train, x_train), permutation, start,
                                                                          completed)
                x_img, y_class = train_batch
                # print(x_img.shape)
                # print(y_class.shape)
                # print(y_class[0])
                feed_dict_train = {x: x_img, y_true: y_class}
                # print(session.run(layer4, feed_dict=feed_dict_train)[0])
                session.run(optimizer, feed_dict=feed_dict_train)
                acc.append(print_accuracy(feed_dict_train))
            if (i+1) % 10 is 0:
                learning_rate /= 2
        plt.plot(np.arange(0, len(acc)), acc)
        plt.show()

    optimize(epoch, learning_rate)
    session.close()


def one_hot(y):
    m = []
    print(y)
    for i in y:
        xx = np.zeros((10))
        xx[i] = 1
        m.append(xx)
    return np.array(m)


def main():
    class_label, train_image, test_image = read_data()
    X_train = train_image.values
    Y_train = class_label.values
    X_test = test_image.values
    print(X_train)
    # mu = np.mean(train_image, 1).reshape(42000)
    # std = np.std(train_image, 1).reshape(42000,1)
    # train_image = np.subtract(train_image, mu) / std
    model_pixel(train_image, one_hot(class_label), train_image.shape[1], 10)
    # permutation = np.arange(len(class_label))
    # train_batch, start, completed, permutation = random_batch((class_label, train_image), permutation, 0, False)
    # x, y = train_batch

    # model_hog(train_image, one_hot(class_label), train_image.shape[1], 10)

    # img=train_image.iloc[0]
    # print(img.shape)
    # print(pd.DataFrame.as_matrix(img))
    # aks = np.reshape(pd.DataFrame.as_matrix(img), (28, 28))
    # plt.imshow(aks)
    # plt.show()
    # features = []
    # for i in range(len(train_image)):
    #     features.append(feature.hog(np.reshape(pd.DataFrame.as_matrix(train_image.iloc[i]), (28, 28)), orientations=9, pixels_per_cell=(9, 9), cells_per_block=(3, 3), block_norm='L2-Hys'))
    # features = np.array(features)
    # features = np.subtract(features, np.mean(features, axis=0))/np.std(features, axis=0)
    # print(features.shape)
    # model_hog(features, one_hot(class_label), features.shape[1], 10)
    # X_std = StandardScaler().fit_transform(features)
    # pca = PCA(n_components=32)
    # pca.fit(X_std)
    # X_32d = pca.transform(X_std)
    # model_hog(X_32d, one_hot(class_label), X_32d.shape[1], 10)
    # results = clf.predict(test_data[0:5000])
    # df = pd.DataFrame(results)
    # df.index.name = 'ImageId'
    # df.index += 1
    # df.columns = ['Label']
    # df.to_csv('results.csv', header=True)
    # X_std = StandardScaler().fit_transform(train_image)
    # pca = PCA(n_components=5)
    # pca.fit(X_std)
    # X_5d = pca.transform(X_std)
    # print(X_5d)
    # print(X_5d.shape)
    # model_pixel(train_image, one_hot(class_label), train_image.shape[1], 10)
    # model_hog(X_5d, one_hot(class_label), X_5d.shape[1], 10)
    # pca = PCA(n_components=15)
    # pca.fit(X_std)
    # X_15d = pca.transform(X_std)
    # model_hog(X_15d, one_hot(class_label), X_15d.shape[1], 10)
    # pca = PCA(n_components=150)
    # pca.fit(X_std)
    # X_150d = pca.transform(X_std)
    # model_hog(X_150d, one_hot(class_label), X_150d.shape[1], 10)
    # pca = PCA(n_components=256)
    # pca.fit(X_std)
    # X_256d = pca.transform(X_std)
    # model_hog(X_256d, one_hot(class_label), X_256d.shape[1], 10)






if __name__ == '__main__':
    main()