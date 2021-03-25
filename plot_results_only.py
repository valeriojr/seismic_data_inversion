import argparse
from pathlib import Path

from keras.models import load_model

import dataset
from main import plot_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', type=str)
    parser.add_argument('data_dir', type=str)
    parser.add_argument('--plot_format', type=str, default='png')

    args = parser.parse_args()

    _, _, X_test, Y_test = dataset.load_data(args.data_dir, mode='binary')
    X_test = X_test.reshape((*X_test.shape, 1))
    Y_test = Y_test.reshape((*Y_test.shape, 1))

    model_dir = Path(args.model_dir)
    model = load_model(model_dir / 'model.h5')

    predict = model.predict(X_test)
    plot_results(X_test, Y_test, predict, model_dir/'figures', args.plot_format)


if __name__ == '__main__':
    main()
