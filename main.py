# Disable Tensorflow debugging information
# https://stackoverflow.com/a/42121886
# Desablita as mensagens de erro do tensorflow
import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Flatten, Reshape
from keras.losses import MAE

import dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('dataset_opening_mode', default='binary')
    parser.add_argument('model_filename', default='model.h5')
    parser.add_argument('batch_size', default=32)
    parser.add_argument('epochs', default=50)
    parser.add_argument('patience', default=2)

    args = parser.parse_args()

    X_train, Y_train, X_test, Y_test = dataset.load_data(args.data_path, mode=args.dataset_opening_mode)

    # Nesse caso o shape de entrada/saída será igual ao shape de X[i] e Y[i]
    input_shape = X_train.shape[1:]
    output_shape = Y_train.shape[1:]

    model = Sequential()
    model.add(Flatten(input_shape=input_shape))

    # Aqui são adicionadas as camadas que interessam
    model.add(Dense(units=2, activation='tanh'))
    model.add(Dense(units=2, activation='tanh'))
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
                  EarlyStopping(patience=args.patience), ModelCheckpoint(args.model_filename)
              ],
              validation_data=(X_test, Y_test))


if __name__ == '__main__':
    main()
