import tensorflow as tf


class PreNet(tf.keras.layers.Layer):
    def __init__(self, units, drop_rate, activation, name='PreNet', **kwargs):
        super(PreNet, self).__init__(name=name, **kwargs)
        self.dense1 = tf.keras.layers.Dense(
            units=units, activation=activation, name='dense_1')
        self.dense2 = tf.keras.layers.Dense(
            units=units, activation=activation, name='dense_2')
        self.dropout_layer = tf.keras.layers.Dropout(rate=drop_rate)

    def call(self, inputs, training=None):
        dense1_out = self.dense1(inputs)
        dense1_out = self.dropout_layer(dense1_out, training=training)
        dense2_out = self.dense2(dense1_out)
        dense2_out = self.dropout_layer(dense2_out, training=training)
        return dense2_out


class ConvPreNet(tf.keras.layers.Layer):
    def __init__(self, nconv, hidden, conv_kernel, drop_rate,
                 activation=tf.nn.relu, bn_before_act=True, name='ConvPrenet', **kwargs):
        super(ConvPreNet, self).__init__(name=name, **kwargs)
        self.conv_stack = []
        for i in range(nconv):
            conv = Conv1D(filters=hidden, kernel_size=conv_kernel, activation=activation,
                          drop_rate=drop_rate, bn_before_act=bn_before_act,
                          name='PreNetConv{}'.format(i))
            self.conv_stack.append(conv)
        self.projection = tf.keras.layers.Dense(units=hidden)

    def call(self, inputs, training=None):
        conv_outs = inputs
        for conv in self.conv_stack:
            conv_outs = conv(conv_outs, training=training)
        projections = self.projection(conv_outs)
        return projections


class FFN(tf.keras.layers.Layer):
    def __init__(self, hidden1, hidden2, name='PositionalFeedForward', **kwargs):
        super(FFN, self).__init__(name=name, **kwargs)
        self.dense1 = tf.keras.layers.Dense(units=hidden1, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=hidden2, activation=None)
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, inputs, training=None):
        dense1_outs = self.dense1(inputs)
        dense2_outs = self.dense2(dense1_outs)
        outs = dense2_outs + inputs
        outs = self.layer_norm(outs, training=training)
        return outs


class Conv1DLayerNorm(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, activation, drop_rate,
                 padding='SAME', strides=1, name='Conv1D_with_dropout_IN'):
        super(Conv1DLayerNorm, self).__init__(name=name)
        self.filters = filters
        self.kernel_size = kernel_size
        self.drop_rate = drop_rate
        self.padding = padding
        self.conv1d = tf.keras.layers.Conv1D(filters=filters,
                                             kernel_size=kernel_size,
                                             strides=strides,
                                             padding=padding,
                                             activation=None,
                                             name='conv1d')
        self.activation = activation if activation is not None else tf.identity
        self.layer_norm = tf.keras.layers.LayerNormalization(name='layerNorm')
        self.dropout = tf.keras.layers.Dropout(rate=drop_rate, name='dropout')

    def call(self, inputs, training=None):
        conv_outs = self.conv1d(inputs)
        conv_outs = self.layer_norm(conv_outs, training=training)
        conv_outs = self.activation(conv_outs)
        dropouts = self.dropout(conv_outs, training=training)
        return dropouts


class Conv1D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, activation, drop_rate,
                 bn_before_act=False, padding='SAME', strides=1,
                 name='Conv1D_with_dropout_BN', **kwargs):
        super(Conv1D, self).__init__(name=name, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.drop_rate = drop_rate
        self.padding = padding
        self.conv1d = tf.keras.layers.Conv1D(filters=filters,
                                             kernel_size=kernel_size,
                                             strides=strides,
                                             padding=padding,
                                             activation=None,
                                             name='conv1d')
        self.activation = activation if activation is not None else tf.identity
        self.bn = tf.keras.layers.BatchNormalization(name='batch_norm')
        self.dropout = tf.keras.layers.Dropout(rate=drop_rate, name='dropout')
        self.bn_before_act = bn_before_act

    def call(self, inputs, training=None):
        conv_outs = self.conv1d(inputs)
        if self.bn_before_act:
            conv_outs = self.bn(conv_outs, training=training)
            conv_outs = self.activation(conv_outs)
        else:
            conv_outs = self.activation(conv_outs)
            conv_outs = self.bn(conv_outs, training=training)
        dropouts = self.dropout(conv_outs, training=training)
        return dropouts

    def get_config(self):
        config = super(Conv1D, self).get_config()
        config.update({'filters': self.filters,
                       'kernel_size': self.kernel_size,
                       'padding': self.padding,
                       'activation': self.activation,
                       'dropout_rate': self.drop_rate,
                       'bn_before_act': self.bn_before_act})
        return config


class PostNet(tf.keras.layers.Layer):
    def __init__(self, n_conv, conv_filters, conv_kernel,
                 drop_rate, name='PostNet', **kwargs):
        super(PostNet, self).__init__(name=name, **kwargs)
        self.conv_stack = []
        self.batch_norm_stack = []
        activations = [tf.math.tanh] * (n_conv - 1) + [tf.identity]
        for i in range(n_conv):
            conv = Conv1D(filters=conv_filters, kernel_size=conv_kernel,
                          padding='same', activation=activations[i],
                          drop_rate=drop_rate, name='conv_{}'.format(i))
            self.conv_stack.append(conv)

    def call(self, inputs, training=None):
        conv_out = inputs
        for conv in self.conv_stack:
            conv_out = conv(conv_out, training)
        return conv_out


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, name='PositionalEncoding'):
        super(PositionalEncoding, self).__init__(name=name)

    @staticmethod
    def positional_encoding(len, dim, step=1.):
        """
        :param len: int scalar
        :param dim: int scalar
        :param step:
        :return: position embedding
        """
        pos_mat = tf.tile(
            tf.expand_dims(
                tf.range(0, tf.cast(len, dtype=tf.float32), dtype=tf.float32) * step,
                axis=-1),
            [1, dim])
        dim_mat = tf.tile(
            tf.expand_dims(
                tf.range(0, tf.cast(dim, dtype=tf.float32), dtype=tf.float32),
                axis=0),
            [len, 1])
        dim_mat_int = tf.cast(dim_mat, dtype=tf.int32)
        pos_encoding = tf.where(  # [time, dims]
            tf.math.equal(tf.math.mod(dim_mat_int, 2), 0),
            x=tf.math.sin(pos_mat / tf.pow(10000., dim_mat / tf.cast(dim, tf.float32))),
            y=tf.math.cos(pos_mat / tf.pow(10000., (dim_mat - 1) / tf.cast(dim, tf.float32))))
        return pos_encoding


class EncBlk(tf.keras.layers.Layer):
    """
    1D convolutional block for time-invariant feature extraction
    """

    def __init__(self, out_channels, kernel_size, activation, pooling_kernel):
        super(EncBlk, self).__init__()
        self.conv1 = tf.keras.layers.Conv1D(out_channels, kernel_size=kernel_size, padding='same')
        self.conv2 = tf.keras.layers.Conv1D(out_channels, kernel_size=kernel_size, padding='same')
        self.conv_sc = tf.keras.layers.Conv1D(out_channels, kernel_size=1, padding='valid')
        self.downsample = tf.keras.layers.AveragePooling1D(pool_size=pooling_kernel)
        self.activation = activation

    def call(self, x):
        """
        :param x: [batch, channels, time-length]
        :return:
        """
        h = self.activation(x)
        h = self.activation(self.conv1(h))
        h = self.conv2(h)
        h = self.downsample(h)
        sc = self.downsample(self.conv_sc(x))
        out = h + sc
        return out


def get_activation(act_str):
    return {'relu': tf.nn.relu, 'leaky_relu': tf.nn.leaky_relu, 'tanh': tf.math.tanh}[act_str]


@tf.custom_gradient
def grad_reverse(x):
    y = tf.identity(x)

    def custom_grad(dy):
        return -dy

    return y, custom_grad


class GradReverse(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, x):
        return grad_reverse(x)
