from pathlib import Path

import numpy


def load_data(path, mode='text', n_samples_train=2000, n_samples_test=50, seismogram_shape=(201, 51),
              model_shape=(101, 51)):
    """
    Carrega os dados como array numpy

    :param path: Caminho para a pasta que contém os arquivos de treino/teste
    :param mode: Modo que os dados devem ser carregados. 'text' para carregar os arquivos originais, 'binary' para
    carregar arquivos salvos com save_data
    :param n_samples_train: Número de exemplos usados no treino
    :param n_samples_test: Número de exemplos usados no teste
    :param seismogram_shape: Dimensão do vetor de entrada
    :param model_shape: Dimensão do vetor de saída

    :return: tuple (seismogram_train, model_train, seismogram_test, model_test)
    """

    data_path = Path(path)

    if mode == 'text':
        seismogram_train = numpy.loadtxt(data_path / 'sismo_treino')
        model_train = numpy.loadtxt(data_path / 'modelo_treino')
        seismogram_test = numpy.loadtxt(data_path / 'sismo_teste')
        model_test = numpy.loadtxt(data_path / 'modelo_teste')

        seismogram_train = numpy.reshape(seismogram_train, (n_samples_train, *seismogram_shape))
        model_train = numpy.reshape(model_train, (n_samples_train, *model_shape))
        seismogram_test = numpy.reshape(seismogram_test, (n_samples_test, *seismogram_shape))
        model_test = numpy.reshape(model_test, (n_samples_test, *model_shape))
    elif mode == 'binary':
        seismogram_train = numpy.load(data_path / 'sismo_treino.npy')
        model_train = numpy.load(data_path / 'modelo_treino.npy')
        seismogram_test = numpy.load(data_path / 'sismo_teste.npy')
        model_test = numpy.load(data_path / 'modelo_teste.npy')
    else:
        raise ValueError(f"mode '{mode}' must be 'text' or 'binary'")

    return seismogram_train, model_train, seismogram_test, model_test


def save_data(path, seismogram_train, model_train, seismogram_test, model_test):
    """
    Salva o dataset em formato binário

    :param path: Caminho onde os arquivos devem ser salvos
    :param seismogram_train: Numpy array contendo os sismogramas de treino
    :param model_train: Numpy array contendo os modelos de treino
    :param seismogram_test: Numpy array contendo os sismogramas de teste
    :param model_test: Numpy array contendo os modelos de teste
    """
    data_path = Path(path)

    numpy.save(data_path / 'sismo_treino.npy', seismogram_train)
    numpy.save(data_path / 'modelo_treino.npy', model_train)
    numpy.save(data_path / 'sismo_teste.npy', seismogram_test)
    numpy.save(data_path / 'modelo_teste.npy', model_test)
