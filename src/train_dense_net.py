import numpy as np
import tensorflow as tf

from dataset import Generator, get_xs_from_pickle, get_ys, read_pickle
from dense_net import dense_net
from funcy import concat, identity, juxt, partial, repeat, take
from operator import getitem


BATCH_SIZE = 128
EPOCH_SIZE = 100

rng = np.random.default_rng(0)


def prepare(x0_vocab_size):
    def op(x):
        x0, x1, x2 = x

        return (tf.keras.layers.Embedding(x0_vocab_size, 512)(x0),
                tf.keras.layers.Concatenate(axis=2)((x1, x2)))

    return op


def regress():
    def op(x):
        x0, x1 = x

        result = tf.keras.layers.Concatenate()((tf.keras.layers.Flatten()(x0), x1))
        result = tf.keras.layers.Dense(512, activation='gelu', use_bias=False)(result)
        result = tf.keras.layers.Dropout(0.5)(result)
        result = tf.keras.layers.Dense(256, activation='gelu', use_bias=False)(result)
        result = tf.keras.layers.Dropout(0.5)(result)
        result = tf.keras.layers.Dense(1, use_bias=False)(result)

        return result

    return op


def op(x):
    result_0, result_1 = prepare(127)(x)

    return regress()((result_0, dense_net(32)(result_1)))


def rmspe(y_true, y_pred):
    return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square((y_true - y_pred) / y_true)))


def get_datasets():
    ys = read_pickle('ys.pickle')

    rng.shuffle(ys)

    return Generator(ys[50000:], BATCH_SIZE), (get_xs_from_pickle(ys[:50000]), get_ys(ys[:50000]))


model = tf.keras.Model(*juxt(identity, op)((tf.keras.Input(shape=(1,)), tf.keras.Input(shape=(600, 4 * 2)), tf.keras.Input(shape=(600, 1 * 3)))))
model.compile('adam', loss=rmspe)
model.summary()

train_generator, valid_dataset = get_datasets()

model.fit(train_generator,
          epochs=EPOCH_SIZE,
          callbacks=(tf.keras.callbacks.LearningRateScheduler(partial(getitem, tuple(take(EPOCH_SIZE, concat(repeat(0.01, EPOCH_SIZE // 2), repeat(0.01 / 10, EPOCH_SIZE // 4), repeat(0.01 / 100))))))),
          validation_data=valid_dataset)

model.save('model', include_optimizer=False)
