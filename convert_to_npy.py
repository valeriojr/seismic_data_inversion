import argparse

import dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', help='Caminho para o dataset')

    args = parser.parse_args()

    X_train, Y_train, X_test, Y_test = dataset.load_data(args.dir)
    dataset.save_data(args.dir, X_train, Y_train, X_test, Y_test)


if __name__ == '__main__':
    main()