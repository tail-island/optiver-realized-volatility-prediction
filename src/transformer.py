import numpy as np
import tensorflow as tf

from funcy import func_partial, rcompose


def transformer(num_blocks, d_model, num_heads, d_ff, x_maximum_position, dropout_rate):
    # KerasやTensorflowのレイヤーや関数をラップします。

    def dense(units):
        return tf.keras.layers.Dense(units)

    def dropout(rate):
        return tf.keras.layers.Dropout(rate)

    def embedding(input_dim, output_dim):
        return tf.keras.layers.Embedding(input_dim, output_dim)

    def gelu():
        return tf.keras.activations.gelu

    def layer_normalization():
        return tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def reshape(target_shape):
        return tf.keras.layers.Reshape(target_shape)

    def transpose(perm):
        return func_partial(tf.transpose, perm=perm)

    # Transformerに必要な演算を定義します。

    def scaled_dot_product_attention(x):
        query, key, value = x

        return tf.matmul(tf.nn.softmax(tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(tf.cast(tf.shape(key)[-1], tf.float32)), axis=-1), value)

    def multi_head_attention(d_model, num_heads):
        split  = rcompose(reshape((-1, num_heads, d_model // num_heads)),  # noqa: E221
                          transpose((0, 2, 1, 3)))
        concat = rcompose(transpose((0, 2, 1, 3)),
                          reshape((-1, d_model)))

        def op(inputs):
            q, k, v = inputs

            o = scaled_dot_product_attention((split(dense(d_model)(q)),
                                              split(dense(d_model)(k)),
                                              split(dense(d_model)(v))))
            o = concat(o)
            o = dense(d_model)(o)

            return o

        return op

    def point_wise_feed_forward(d_model, d_ff):
        return rcompose(dense(d_ff),
                        gelu(),
                        dense(d_model))

    def encoder_block(d_model, num_heads, d_ff, dropout_rate):
        def op(inputs):
            x = inputs

            o = layer_normalization()(x)
            o = dropout(dropout_rate)(multi_head_attention(d_model, num_heads)((o, o, o))) + o
            o = layer_normalization()(o)
            o = dropout(dropout_rate)(point_wise_feed_forward(d_model, d_ff)(o)) + o

            return o

        return op

    def get_positional_encoding(maximum_position, d_model):
        result = embedding(maximum_position, d_model)(tf.range(0, maximum_position))

        return result[np.newaxis, ...]

    def encoder(num_blocks, d_model, num_heads, d_ff, maximum_position, dropout_rate):
        positional_encoding = get_positional_encoding(maximum_position, d_model)

        def op(inputs):
            x = inputs

            o = dropout(dropout_rate)(x + positional_encoding)

            for _ in range(num_blocks):
                o = encoder_block(d_model, num_heads, d_ff, dropout_rate)((o))

            return o

        return op

    def op(inputs):
        x = inputs

        return encoder(num_blocks, d_model, num_heads, d_ff, x_maximum_position, dropout_rate)(x)

    return op


class LearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(LearningRateSchedule, self).__init__()

        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        return self.d_model ** -0.5 * tf.math.minimum(step ** -0.5, step * self.warmup_steps ** -1.5)
