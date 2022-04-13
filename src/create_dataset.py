import os

from dataset import create_train_dataset


if __name__ == '__main__':
    os.makedirs('./data', exist_ok=True)

    create_train_dataset()
