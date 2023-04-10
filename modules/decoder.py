import tensorflow as tf
from modules.utils import Conv1D, PostNet, ConvPreNet
from modules.attention import SelfAttentionBLK


class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self, pre_n_conv, pre_conv_kernel, pre_drop_rate, nblk, attention_dim,
                 attention_heads, temperature, attention_causality, attention_window,
                 ffn_hidden, post_n_conv, post_conv_filters, post_conv_kernel, post_drop_rate,
                 out_dim, reduction_factor, name='TransformerDecoder'):
        super(TransformerDecoder, self).__init__(name=name)
        self.reduction_factor = reduction_factor
        self.out_dim = out_dim
        self.pre_projection = ConvPreNet(
            nconv=pre_n_conv, hidden=attention_dim, conv_kernel=pre_conv_kernel, drop_rate=pre_drop_rate)
        self.attentions = []
        for i in range(nblk):
            attention = SelfAttentionBLK(
                input_dim=attention_dim,
                attention_dim=attention_dim,
                attention_heads=attention_heads,
                attention_temperature=temperature,
                causality=attention_causality,
                attention_window=attention_window,
                ffn_hidden=ffn_hidden, name='decoder-attention-{}'.format(i))
            self.attentions.append(attention)
        self.out_projection = tf.keras.layers.Dense(units=out_dim * self.reduction_factor,
                                                    name='linear_outputs')
        self.postnet = PostNet(n_conv=post_n_conv, conv_filters=post_conv_filters,
                               conv_kernel=post_conv_kernel, drop_rate=post_drop_rate,
                               name='postnet')
        self.residual_projection = tf.keras.layers.Dense(units=out_dim, name='residual_outputs')

    def call(self, inputs, lengths=None, training=None):
        print('Tracing back at Self-attention decoder')
        batch_size = tf.shape(inputs)[0]
        max_len = tf.shape(inputs)[1]
        att_outs = self.pre_projection(inputs, training=training)
        alignments = {}
        for att in self.attentions:
            att_outs, ali = att(
                inputs=att_outs, memory=att_outs, query_lengths=lengths,
                memory_lengths=lengths, training=training)
            alignments[att.name] = ali
        initial_outs = self.out_projection(att_outs)
        initial_outs = tf.reshape(
            initial_outs, [batch_size, max_len * self.reduction_factor, self.out_dim])
        residual = self.postnet(initial_outs, training=training)
        residual = self.residual_projection(residual)
        outputs = residual + initial_outs
        return initial_outs, outputs, alignments


class PitchPredictor(tf.keras.layers.Layer):
    def __init__(self, n_conv, conv_filters, conv_kernel, drop_rate, name='PitchPredictor'):
        super(PitchPredictor, self).__init__(name=name)
        self.conv_stack = [Conv1D(filters=conv_filters, kernel_size=conv_kernel,
                                  activation=tf.nn.relu, drop_rate=drop_rate)
                           for _ in range(n_conv)]
        self.out_proj = tf.keras.layers.Dense(units=4)

    def call(self, inputs, training=None):
        conv_outs = inputs
        for conv in self.conv_stack:
            conv_outs = conv(conv_outs, training=training)
        outs = self.out_proj(conv_outs)
        return outs
