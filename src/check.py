import matplotlib.pyplot as plot
import numpy as np
import tensorflow as tf

from dataset import get_xs_from_pickle, get_ys, read_pickle


rng = np.random.default_rng(0)


def get_dataset():
    ys = read_pickle('ys.pickle')

    rng.shuffle(ys)

    return get_xs_from_pickle(ys[:40000]), get_ys(ys[:40000])


model = tf.keras.models.load_model('model')

xs, ys_true = get_dataset()

ys_pred = model.predict(xs).flatten()

# TODO: 株価の単位が$0.01であることを利用した補正処理を入れる。

plot.figure(figsize=(15, 15))
plot.scatter(ys_true, ys_pred, s=10, alpha=0.1)
plot.xlim((0, 0.05))
plot.ylim((0, 0.05))
plot.show()

print(np.sqrt(np.mean(np.square((ys_true - ys_pred) / ys_true))))
