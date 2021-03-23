from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.losses import MSE

import numpy

import dataset


def main():
    X_train, Y_train, X_test, Y_test = dataset.load_data('../Problema1-Petrobras', mode='binary')

    input_shape = X_train[0].shape[0] * X_train[0].shape[1]
    output_shape = Y_train[0].shape[0] * Y_train[0].shape[1]

    X_train = numpy.reshape(X_train, (len(X_train), input_shape))
    Y_train = numpy.reshape(Y_train, (len(Y_train), output_shape))
    X_test = numpy.reshape(X_test, (len(X_test), input_shape))
    Y_test = numpy.reshape(Y_test, (len(Y_test), output_shape))

    model = Sequential()
    model.add(Input(shape=(input_shape,)))
    model.add(Dense(units=output_shape, activation='relu'))

    model.compile(optimizer='adam', loss=MSE, metrics=['mse'])

    print(model.summary())

    model.fit(X_train,
              Y_train,
              batch_size=32,
              epochs=30,
              callbacks=[
                  EarlyStopping(), ModelCheckpoint('model.h5')
              ],
              validation_data=(X_test, Y_test))


if __name__ == '__main__':
    main()
