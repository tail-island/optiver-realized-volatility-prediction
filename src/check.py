import matplotlib.pyplot as plot
import numpy as np
import tensorflow as tf

from dataset import get_dataset, read_pickle


rng = np.random.default_rng(0)


def get_ys():
    result = read_pickle('ys.pickle')

    rng.shuffle(result)

    return result[:40000]


model = tf.keras.models.load_model('model')

xs, ys_true = get_dataset(get_ys())

ys_pred = model.predict(xs).flatten()

# TODO: 株価の単位が$0.01であることを利用した補正処理を入れる。

plot.scatter(ys_true, ys_pred, s=10, alpha=0.1)
plot.show()

print(np.sqrt(np.mean(np.square((ys_true - ys_pred) / ys_true))))
