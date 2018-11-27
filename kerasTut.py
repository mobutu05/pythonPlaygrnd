import keras
from keras.datasets import mnist
from keras import models, Sequential
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
from keras.utils import to_categorical
import numpy as np
from numpy.core.multiarray import ndarray


def example1():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    network = models.Sequential()
    network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
    network.add(layers.Dense(10, activation='softmax'))

    network.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    train_images = train_images.reshape((60000, 28 * 28))
    train_images = train_images.astype('float32') / 255
    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype('float32') / 255

    network.fit(train_images, train_labels, epochs=5, batch_size=128)

    test_loss, test_acc = network.evaluate(test_images, test_labels)
    print('test_acc:', test_acc)
    #
    # print (train_images)

    keras.layers.Dense(512, activation='relu')


def vectorize_sequences(sequences: ndarray, dimension: int = 10000) -> ndarray:
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


from keras.datasets import imdb

test_labels: ndarray
train_data: ndarray
train_labels: ndarray
test_data: ndarray
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
x_train: ndarray = vectorize_sequences(train_data)
x_test: int = vectorize_sequences(test_data)
y_train: ndarray = np.asarray(train_labels).astype('float32')
y_test: ndarray = np.asarray(test_labels).astype('float32')
model: Sequential = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
# model.compile(optimizer='rmsprop',
#               metrics=['accuracy'],
#               loss='binary_crossentropy')
# model.compile(optimizer=optimizers.RMSprop(lr=0.001),
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
