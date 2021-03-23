import argparse
import os
from pathlib import Path

from keras.models import load_model
from matplotlib import pyplot
import numpy
from tqdm import tqdm

import dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Caminho para carregar a rede')
    parser.add_argument('data_path', help='Caminho para os dados')
    parser.add_argument('format', help="Formato para salvar os resultados ('png' ou 'pdf')")
    parser.add_argument('output_path', help='Caminho para o diretório em que os resultados devem ser salvos')

    args = parser.parse_args()

    _, _, X_test, Y_test = dataset.load_data(args.data_path, mode='binary')
    X_test = numpy.reshape(X_test, (X_test.shape[0], X_test.shape[1] * X_test.shape[2]))
    Y_test = numpy.reshape(Y_test, (Y_test.shape[0], Y_test.shape[1] * Y_test.shape[2]))
    model = load_model(args.model_path)
    output_path = Path(args.output_path)

    predict = model.predict(X_test)

    X_test = numpy.reshape(X_test, (X_test.shape[0], 201, 51))
    Y_test = numpy.reshape(Y_test, (Y_test.shape[0], 101, 51))
    predict = numpy.reshape(predict, (X_test.shape[0], 101, 51))

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for i in tqdm(range(predict.shape[0])):
        fig = pyplot.figure()
        seismogram_ax = fig.add_subplot(131)
        model_ax = fig.add_subplot(132)
        predict_ax = fig.add_subplot(133)

        seismogram_ax.set_xlabel('Sismograma')
        seismogram_ax.imshow(X_test[i], 'gray')
        model_ax.set_xlabel('Modelo')
        model_ax.imshow(Y_test[i])
        predict_ax.set_xlabel('Resultado da rede')
        predict_ax.imshow(predict[i])

        pyplot.savefig(output_path / f'predict_{i}.{args.format}')
        pyplot.close(fig)


if __name__ == '__main__':
    main()