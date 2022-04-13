import numpy as np
import os
import pandas as pd
import pickle
import tensorflow as tf

from funcy import cat, concat, pairwise, repeat
from glob import glob
from itertools import starmap
from math import ceil


rng = np.random.default_rng(0)


def read_csv(filename):
    return pd.read_csv(os.path.join('..', 'input', 'optiver-realized-volatility-prediction', filename))


def read_parquet(type, stock_id):
    return pd.read_parquet(glob(os.path.join('..', 'input', 'optiver-realized-volatility-prediction', f'{type}.parquet', f'stock_id={stock_id}', '*.parquet'))[0])


def read_pickle(filename):
    with open(os.path.join('.', 'data', filename), mode='rb') as f:
        return pickle.load(f)


def get_x1(book, time_id):
    bucket = book[book['time_id'] == time_id]

    seconds = concat(bucket['seconds_in_bucket'],
                     (600,))

    items = bucket[['bid_price1', 'bid_size1', 'bid_price2', 'bid_size2', 'ask_price1', 'ask_size1', 'ask_price2', 'ask_size2']].values

    return np.reshape(np.array(tuple(cat(starmap(lambda repeat_count, item: repeat(item, repeat_count),
                                                 zip(starmap(lambda prev_second, next_second: next_second - prev_second,
                                                             pairwise(seconds)),
                                                     items)))),
                               dtype=np.float32),
                      (600, 4, 2))


def get_x2(trade, time_id):
    bucket = trade[trade['time_id'] == time_id]

    if len(bucket) == 0:
        return np.zeros((600, 1, 3), dtype=np.float32)

    seconds = concat((-1,),
                     bucket['seconds_in_bucket'],
                     (599,) if bucket['seconds_in_bucket'].iloc[-1] != 599 else ())

    items = concat(bucket[['price', 'size', 'order_count']].values,
                   ((0, 0, 0),) if bucket['seconds_in_bucket'].iloc[-1] != 599 else ())

    return np.reshape(np.array(tuple(cat(starmap(lambda gap_count, item: concat(repeat((0, 0, 0), gap_count),
                                                                                (item,)),
                                                 zip(starmap(lambda prev_second, next_second: next_second - prev_second - 1,
                                                             pairwise(seconds)),
                                                     items)))),
                               dtype=np.float32),
                      (600, 1, 3))


def create_train_dataset():
    ys = []

    volatilities = read_csv('train.csv')

    books, trades = map(lambda type: dict(map(lambda stock_id: (stock_id, read_parquet(type, stock_id)), volatilities['stock_id'].drop_duplicates())),
                        ('book_train', 'trade_train'))

    for _, stock_id, time_id, target in volatilities[['stock_id', 'time_id', 'target']].itertuples():
        with open(os.path.join('.', 'data', f'x-{stock_id}-{time_id}-1.pickle'), mode='wb') as f:
            pickle.dump(get_x1(books[stock_id], time_id), f)

        with open(os.path.join('.', 'data', f'x-{stock_id}-{time_id}-2.pickle'), mode='wb') as f:
            pickle.dump(get_x2(trades[stock_id], time_id), f)

        ys.append((stock_id, time_id, target))

    with open(os.path.join('.', 'data', 'ys.pickle'), mode='wb') as f:
        pickle.dump(ys, f)


class Generator(tf.keras.utils.Sequence):
    def __init__(self, ys, batch_size):
        self.ys = ys
        self.batch_size = batch_size

        self.on_epoch_end()

    def __len__(self):
        return ceil(len(self.ys) / self.batch_size)

    def __getitem__(self, i):
        ys = self.ys[i * self.batch_size:(i + 1) * self.batch_size]

        return ((np.reshape(np.array(tuple(map(lambda y: y[0], ys)), dtype=np.int32), (-1, 1)),
                 np.reshape(np.array(tuple(map(lambda y: read_pickle(f'x-{y[0]}-{y[1]}-1.pickle'), ys)), dtype=np.float32), (-1, 600, 4 * 2)),
                 np.reshape(np.array(tuple(map(lambda y: read_pickle(f'x-{y[0]}-{y[1]}-2.pickle'), ys)), dtype=np.float32), (-1, 600, 1 * 3))),
                np.array(tuple(map(lambda y: y[2], ys)), dtype=np.float32))

    def on_epoch_end(self):
        rng.shuffle(self.ys)


def get_dataset(ys):
    return ((np.reshape(np.array(tuple(map(lambda y: y[0], ys)), dtype=np.int32), (-1, 1)),
             np.reshape(np.array(tuple(map(lambda y: read_pickle(f'x-{y[0]}-{y[1]}-1.pickle'), ys)), dtype=np.float32), (-1, 600, 4 * 2)),
             np.reshape(np.array(tuple(map(lambda y: read_pickle(f'x-{y[0]}-{y[1]}-2.pickle'), ys)), dtype=np.float32), (-1, 600, 1 * 3))),
            np.array(tuple(map(lambda y: y[2], ys)), dtype=np.float32))
