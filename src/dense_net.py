import tensorflow as tf

from funcy import rcompose


def dense_net(growth_rate):
    # KerasやTensorflowのレイヤーや関数をラップします。

    def average_pooling_1d(pool_size, strides=None):
        return tf.keras.layers.AveragePooling1D(pool_size, strides=strides)

    def concatenate():
        return tf.keras.layers.Concatenate()

    def batch_normalization():
        return tf.keras.layers.BatchNormalization(epsilon=1.001e-5)

    def conv_1d(filters, kernel_size):
        return tf.keras.layers.Conv1D(filters, kernel_size, padding='same', use_bias=False)

    def global_average_pooling_1d():
        return tf.keras.layers.GlobalAveragePooling1D()

    def relu():
        return tf.keras.layers.ReLU()

    # DenseNetに必要な演算を定義します。

    def dense_block(blocks):
        def op(inputs):
            result = inputs

            for _ in range(blocks):
                result_ = batch_normalization()(result)
                result_ = relu()(result_)
                result_ = conv_1d(4 * growth_rate, 1)(result_)
                result_ = batch_normalization()(result_)
                result_ = relu()(result_)
                result_ = conv_1d(growth_rate, 2)(result_)

                result = concatenate()((result, result_))

            return result

        return op

    def transition_block():
        def op(inputs):
            result = batch_normalization()(inputs)
            result = relu()(result)
            result = conv_1d(int(tf.keras.backend.int_shape(inputs)[2] * 0.5), 1)(result)
            result = average_pooling_1d(2)(result)

            return result

        return op

    # DenseNet-121を作成します。

    return rcompose(dense_block(6),
                    transition_block(),
                    dense_block(12),
                    transition_block(),
                    dense_block(24),
                    transition_block(),
                    dense_block(16),
                    global_average_pooling_1d())
