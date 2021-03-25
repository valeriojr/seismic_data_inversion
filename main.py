# Disable Tensorflow debugging information
# https://stackoverflow.com/a/42121886
# Desablita as mensagens de erro do tensorflow
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from matplotlib import pyplot, gridspec
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Flatten, Reshape, Conv2D, MaxPooling2D, Input
from keras.losses import MAE
import numpy

import dataset


def plot_results(X, Y, predict, path, format):
    relative_errors = numpy.zeros(predict.shape[0])
    for i in tqdm(range(predict.shape[0])):
        figure = pyplot.figure(figsize=(10, 8))
        figure.suptitle('Comparação entre o modelo esperado e o modelo obtido')

        grid_size = (2, 4)
        seismogram_ax = pyplot.subplot2grid(grid_size, (0, 0))
        model_ax = pyplot.subplot2grid(grid_size, (0, 1))
        predict_ax = pyplot.subplot2grid(grid_size, (0, 2))
        predict_mean_ax = pyplot.subplot2grid(grid_size, (0, 3))
        error_ax = pyplot.subplot2grid(grid_size, (1, 0), colspan=4, ymargin=1.0)

        seismogram_ax.set_title('Sismograma')
        seismogram_ax.imshow(X[i], cmap='gray', aspect='auto')

        model_ax.set_title('Modelo esperado')
        model_ax.imshow(Y[i], aspect='auto')

        predict_ax.set_title('Modelo obtido')
        predict_ax.imshow(predict[i], aspect='auto')

        mean = numpy.mean(predict[i], axis=1)
        mean_tiled = numpy.zeros(predict.shape[1:])
        for j in range(predict.shape[2]):
            mean_tiled[:, j] = mean

        predict_mean_ax.set_title('Modelo médio obtido')
        predict_mean_ax.imshow(mean_tiled, aspect='auto')

        # Erro
        ground_truth = Y[i][:, 0]
        relative_error = 100.0 * numpy.fabs((ground_truth - mean)/ground_truth).sum()
        relative_errors[i] = relative_error
        error_ax.set_title(f'Erro relativo = {relative_error:.2f}%')
        error_ax.set_ylabel('Velocidade')
        error_ax.plot(ground_truth, 'k', label='Modelo esperado')
        error_ax.plot(mean, 'r', label='Modelo médio obtido')
        error_ax.legend()

        pyplot.savefig(path / f'prediction_{i}.{format}')
        pyplot.close(figure)

    figure = pyplot.figure(figsize=(8, 6))
    ax = figure.add_subplot(111)
    ax.plot(relative_errors, 'k')
    ax.hlines([relative_errors.mean()], 0, len(predict), 'r')
    ax.set_xlabel('Exemplo')
    ax.set_ylabel('Erro relativo')
    pyplot.savefig(path / f'relative_error.{format}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('--dataset_opening_mode', type=str, default='binary')
    parser.add_argument('--model_name', type=str, default='sequential')
    parser.add_argument('--overwrite_model', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--print_summary', type=bool, default=True)
    parser.add_argument('--fit_verbose', type=int, default=1)
    parser.add_argument('--plot_test_predict', type=bool, default=True)
    parser.add_argument('--plot_format', type=str, default='png')

    args = parser.parse_args()

    results_dir = Path('resultados')
    if not results_dir.exists():
        results_dir.mkdir()

    model_dir = results_dir / (args.model_name + datetime.now().strftime('_%d_%m_%Y__%H_%M_%S'))

    # Cria a pasta onde os resultados deste modelo serão salvos
    if not model_dir.exists():
        model_dir.mkdir()

    # Cria a pasta onde os gráficos para este modelo serão salvos
    figures_dir = model_dir / 'figures'
    if not (figures_dir).exists():
        figures_dir.mkdir()

    X_train, Y_train, X_test, Y_test = dataset.load_data(args.data_path, mode=args.dataset_opening_mode)
    # Pré processa os dados de entrada, adicionando um novo axis ao array (quantidade de canais da imagem, nesse caso, 1)
    X_train = numpy.reshape(X_train, (*X_train.shape, 1))
    X_test = numpy.reshape(X_test, (*X_test.shape, 1))

    # Nesse caso o shape de entrada/saída será igual ao shape de X[i] e Y[i]
    input_shape = X_train.shape[1:]
    output_shape = Y_train.shape[1:]

    model = Sequential(name=args.model_name)
    model.add(Input(shape=input_shape))

    # Aqui são adicionadas as camadas que interessam
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(units=2048, activation='relu'))
    model.add(Dense(units=2048, activation='relu'))
    model.add(Dense(units=output_shape[0] * output_shape[1]))

    # Can a neural network be configured to output a matrix in Keras?
    # https://stackoverflow.com/a/55976308
    # Faz um reshape na saída da rede para ter o mesmo shape dos dados
    model.add(Reshape(output_shape))
    model.compile(optimizer='adam', loss=MAE, metrics=['mae'])

    if args.print_summary:
        model.summary()
    # Salva a arquitetura da rede num arquivo txt
    with open(model_dir / 'summary.txt', 'w') as fp:
        # Keras model.summary() object to string
        # https://stackoverflow.com/a/45546663
        model.summary(print_fn=lambda line: fp.write(line + '\n'))

    history = model.fit(X_train,
                        Y_train,
                        batch_size=args.batch_size,
                        epochs=args.epochs,
                        callbacks=[
                            EarlyStopping(patience=args.patience),
                            ModelCheckpoint(model_dir / 'model.h5', save_best_only=True)
                        ],
                        validation_data=(X_test, Y_test),
                        verbose=args.fit_verbose)

    # Salva o histórico do treino num arquivo json
    with open(model_dir / 'history.json', 'w') as fp:
        json.dump(history.history, fp, indent=4)

    # Plota os resultados obtidos no conjunto de teste
    predict = model.predict(X_test)
    plot_results(X_test, Y_test, predict, figures_dir, args.plot_format)


if __name__ == '__main__':
    main()
