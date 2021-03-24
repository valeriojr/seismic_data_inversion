from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Input, Flatten, Reshape
from keras.losses import MSE

import numpy

import dataset


def main():
    X_train, Y_train, X_test, Y_test = dataset.load_data('../Problema1-Petrobras', mode='binary')

    # Nesse caso o shape de entrada/saída será igual ao shape de X[i] e Y[i]
    input_shape = X_train.shape[1:]
    output_shape = Y_train.shape[1:]

    model = Sequential()

    model.add(Flatten(input_shape=input_shape))

    # Aqui são adicionadas as camadas que interessam
    model.add(Dense(units=output_shape[0] * output_shape[1], activation='relu'))

    # Can a neural network be configured to output a matrix in Keras?
    # https://stackoverflow.com/a/55976308
    # Faz um reshape na saída da rede para ter o mesmo shape dos dados
    model.add(Reshape(output_shape))

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
