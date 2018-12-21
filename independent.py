import keras
import numpy
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd

number_classes = 10
image_size = 28
image_flatten = image_size * image_size
number_channels = 1

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

Y_train = train["label"]
X_train = train.drop(labels=['label'], axis=1)
X_train = X_train / 255.0
test = test / 255.0
X_train = X_train.values.reshape(-1, image_size, image_size, number_channels)
test = test.values.reshape(-1, image_size, image_size, number_channels)
Y_train = keras.utils.np_utils.to_categorical(Y_train, num_classes=number_classes)

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=2)


model = keras.Sequential()
model.add(keras.layers.Conv2D(filters=32, kernel_size=(5, 5), padding='Same', kernel_initializer='he_normal', activation='relu', input_shape=(image_size, image_size, number_channels)))
model.add(keras.layers.Conv2D(filters=32, kernel_size=(5, 5), padding='Same', kernel_initializer='he_normal', activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='Same', kernel_initializer='he_normal', activation='relu'))
model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='Same', kernel_initializer='he_normal', activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='Same', kernel_initializer='he_normal', activation='relu'))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Dense(number_classes, activation='softmax'))

optimizer = keras.optimizers.RMSprop(lr=0.0003, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
save = keras.callbacks.ModelCheckpoint('tmp/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val-acc', patience=3, verbose=1, factor=0.5, min_lr=1e-6)
epoch = 30
batch_size = 64

augmentation = keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=15,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,
        vertical_flip=False)

augmentation.fit(X_train)

history = model.fit_generator(augmentation.flow(X_train, Y_train, batch_size=batch_size), epochs=epoch, steps_per_epoch=X_train.shape[0], verbose=1,
                              callbacks=[learning_rate_reduction, save], validation_data=(X_val, Y_val))

fig, ax = plt.subplots(2, 1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss", axes=ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r', label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)
