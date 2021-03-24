# Disable Tensorflow debugging information
# https://stackoverflow.com/a/42121886
# Desablita as mensagens de erro do tensorflow
import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Flatten, Reshape, Conv2D, MaxPooling2D, Input
from keras.losses import MAE
import numpy
from sklearn.model_selection import train_test_split

import dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('--dataset_opening_mode', default='binary')
    parser.add_argument('--model_filename', default='model.h5')
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--epochs', default=50)
    parser.add_argument('--patience', default=2)

    args = parser.parse_args()

    X_train, Y_train, X_test, Y_test = dataset.load_data(args.data_path, mode=args.dataset_opening_mode)
    X_train = numpy.reshape(X_train, (*X_train.shape, 1))
    X_test = numpy.reshape(X_test, (*X_test.shape, 1))

    print(X_train.shape)

    # X = numpy.concatenate((X_train, X_test))
    # Y = numpy.concatenate((Y_train, Y_test))

    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.5, random_state=42)
    # X_test, X_validation, Y_test, Y_validation = train_test_split(X_test, Y_test, train_size=0.5)

    # Nesse caso o shape de entrada/saída será igual ao shape de X[i] e Y[i]
    input_shape = X_train.shape[1:]
    output_shape = Y_train.shape[1:]

    model = Sequential()
    # model.add(Input(shape=(*input_shape, 1)))

    # Aqui são adicionadas as camadas que interessam
    model.add(Conv2D(filters=32, kernel_size=(5, 5), input_shape=input_shape, activation='tanh'))
    model.add(MaxPooling2D())
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='tanh'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(units=2048, activation='tanh'))
    model.add(Dense(units=2048 , activation='tanh'))
    model.add(Dense(units=output_shape[0] * output_shape[1]))

    # Can a neural network be configured to output a matrix in Keras?
    # https://stackoverflow.com/a/55976308
    # Faz um reshape na saída da rede para ter o mesmo shape dos dados
    model.add(Reshape(output_shape))
    model.compile(optimizer='adam', loss=MAE, metrics=['mae'])

    print(model.summary())

    model.fit(X_train,
              Y_train,
              batch_size=args.batch_size,
              epochs=args.epochs,
              callbacks=[
                  EarlyStopping(patience=args.patience), ModelCheckpoint(args.model_filename, save_best_only=True)
              ],
              validation_data=(X_test, Y_test))

    # print(model.evaluate(X_validation, Y_validation))


if __name__ == '__main__':
    main()
