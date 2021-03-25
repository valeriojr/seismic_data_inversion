from pathlib import Path
import shutil

from tqdm import tqdm


def main():
    # TODO: Adicionar argumentos para filtrar quais resultados devem ser apagados
    for dir in tqdm(Path('resultados').iterdir()):
        shutil.rmtree(dir)


if __name__ == '__main__':
    main()