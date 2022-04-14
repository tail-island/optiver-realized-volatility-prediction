import numpy as np
import tensorflow as tf

from dataset import Generator, get_xs_from_pickle, get_ys, read_pickle
from funcy import identity, juxt, rcompose
from transformer_param import NUM_BLOCKS, D_MODEL, NUM_HEADS, D_FF, DROPOUT_RATE
from transformer import LearningRateSchedule, transformer


BATCH_SIZE = 64
EPOCH_SIZE = 100

rng = np.random.default_rng(0)


def prepare(x0_vocab_size):
    def op(x):
        x0, x1, x2 = x

        return tf.keras.layers.Concatenate(axis=1)((tf.keras.layers.Embedding(x0_vocab_size, D_MODEL)(x0),
                                                    tf.keras.layers.ReLU()(tf.keras.layers.BatchNormalization()(tf.keras.layers.Conv1D(D_MODEL, 1, padding='same')(tf.keras.layers.Concatenate(axis=2)((x1, x2)))))))

    return op


def regress():
    return rcompose(tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(512, activation='gelu', use_bias=False),
                    tf.keras.layers.Dropout(0.5),
                    tf.keras.layers.Dense(256, activation='gelu', use_bias=False),
                    tf.keras.layers.Dropout(0.5),
                    tf.keras.layers.Dense(1, use_bias=False))


def rmspe(y_true, y_pred):
    return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square((y_true - y_pred) / y_true)))


def get_datasets():
    ys = read_pickle('ys.pickle')

    rng.shuffle(ys)

    return Generator(ys[50000:], BATCH_SIZE), (get_xs_from_pickle(ys[:50000]), get_ys(ys[:50000]))


op = rcompose(prepare(127),
              transformer(NUM_BLOCKS, D_MODEL, NUM_HEADS, D_FF, (1 + 600), DROPOUT_RATE),
              regress())

model = tf.keras.Model(*juxt(identity, op)((tf.keras.Input(shape=(1,)), tf.keras.Input(shape=(600, 4 * 2)), tf.keras.Input(shape=(600, 1 * 3)))))
model.compile(tf.keras.optimizers.Adam(LearningRateSchedule(D_MODEL), beta_1=0.9, beta_2=0.98, epsilon=1e-9), loss=rmspe)
model.summary()

train_generator, valid_dataset = get_datasets()

model.fit(train_generator, epochs=EPOCH_SIZE, validation_data=valid_dataset)
model.save('model', include_optimizer=False)
