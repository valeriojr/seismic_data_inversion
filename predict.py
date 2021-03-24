import argparse
import os
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.models import load_model
from matplotlib import pyplot
from matplotlib import gridspec
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
    X_test = numpy.reshape(X_test, (*X_test.shape, 1))
    model = load_model(args.model_path)
    output_path = Path(args.output_path)

    predict = model.predict(X_test)

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for i in tqdm(range(predict.shape[0])):
        fig = pyplot.figure(figsize=(10, 5))
        gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 4])

        seismogram_ax = fig.add_subplot(gs[0, 0])
        model_ax = fig.add_subplot(gs[0, 1])
        predict_ax = fig.add_subplot(gs[0, 2])
        error_ax = fig.add_subplot(gs[0, 3:])

        seismogram_ax.set_title('Sismograma')
        seismogram_ax.set_adjustable('datalim')
        seismogram_ax.imshow(X_test[i], cmap='gray', aspect='auto')

        model_ax.set_title('Modelo')
        model_ax.set_adjustable('datalim')
        model_ax.imshow(Y_test[i])

        predict_ax.set_title('Resultado da rede')
        predict_ax.set_adjustable('datalim')
        predict_ax.imshow(predict[i])

        # Erro
        mean = numpy.mean(predict[i], axis=1)
        ground_truth = Y_test[i][:, 0]
        # relative_error = numpy.fabs((ground_truth - mean)/ground_truth)
        error_ax.set_title('Erro')
        error_ax.plot(ground_truth, 'k')
        error_ax.plot(mean, 'r')

        pyplot.savefig(output_path / f'predict_{i}.{args.format}')
        pyplot.close(fig)


if __name__ == '__main__':
    main()
