import tensorflow as tf
from typing import Tuple
from .utils import FFN


class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, attention_dim, name='BaseAttention', **kwargs):
        super(BaseAttention, self).__init__(name=name, **kwargs)
        self.attention_dim = attention_dim

    def call(self, inputs, memory, memory_lengths, query_lengths) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        :param inputs: query, [batch, q_time, q_dim]
        :param memory: [batch, m_time, m_dim]
        :param memory_lengths: [batch,]
        :param query_lengths: [batch,]
        :return: (tensor1, tensor2)
            tensor1: contexts, [batch, q_time, attention_dim]
            tensor2: alignments, probabilities, [batch, q_time, m_time]
        """
        raise NotImplementedError

    @staticmethod
    def _get_key_mask(batch_size, memory_max_time, query_max_time, memory_lengths, query_lengths):
        memory_lengths = (memory_lengths if memory_lengths is not None
                          else tf.ones([batch_size], dtype=tf.int32) * memory_max_time)
        memeory_mask = tf.sequence_mask(memory_lengths, maxlen=memory_max_time)
        memeory_mask = tf.tile(tf.expand_dims(memeory_mask, axis=1),  # [batch, 1, m_max_time]
                               [1, query_max_time, 1])  # [batch, q_max_time, m_max_time]
        query_lengths = (query_lengths if query_lengths is not None
                         else tf.ones([batch_size], dtype=tf.int32) * query_max_time)
        query_mask = tf.sequence_mask(query_lengths, maxlen=query_max_time)  # [batch, q_max_time]
        query_mask = tf.tile(tf.expand_dims(query_mask, axis=2),  # [batch, q_max_time, 1]
                             [1, 1, memory_max_time])  # [batch, q_max_time, m_max_time]
        length_mask = tf.logical_and(memeory_mask, query_mask)
        return length_mask

    def get_config(self):
        config = super(BaseAttention, self).get_config()
        config.update({'attention_dim': self.attention_dim})
        return config


class BahdanauAttention(BaseAttention):
    def __init__(self, attention_dim, temperature=1.0, name='attention', **kwargs):
        """
        :param attention_dim:
        :param name:
        :param kwargs:
        """
        super(BahdanauAttention, self).__init__(attention_dim=attention_dim,
                                                name=name, **kwargs)
        self.query_layer = tf.keras.layers.Dense(
            units=attention_dim, use_bias=False, name='query_layer')
        self.memory_layer = tf.keras.layers.Dense(
            units=attention_dim, use_bias=False, name='memory_layer')
        self.score_v = tf.Variable(tf.random.normal([attention_dim, ]),
                                   name='attention_v', trainable=True,
                                   dtype=tf.float32)
        self.score_b = tf.Variable(tf.zeros([attention_dim, ]),
                                   name='attention_b', trainable=True,
                                   dtype=tf.float32)
        self.temperature = temperature

    def _bahdanau_score(self, w_queries, w_keys):
        """
        :param w_queries: [batch, q_max_time, attention_dim]
        :param w_keys: [batch, k_max_time, attention_dim]
        :return: [batch, q_max_time, k_max_time], attention score (energy)
        """
        # [batch, q_max_time, 1, attention_dim]
        expanded_queries = tf.expand_dims(w_queries, axis=2)
        # [batch, 1, k_max_time, attention_dim]
        expanded_keys = tf.expand_dims(w_keys, axis=1)
        # [batch, q_max_time, k_max_time]
        return tf.math.reduce_sum(
            self.score_v * tf.nn.tanh(
                # [batch, q_max_time, k_max_time, attention_dim]
                expanded_keys + expanded_queries + self.score_b),
            axis=3)

    def call(self, inputs, memory, memory_lengths=None, query_lengths=None):
        # TODO: add query lengths and query mask
        """
        :param inputs: query: [batch, q_max_time, query_depth]
        :param memory: [batch, m_max_time, memory_depth]
        :param memory_lengths: [batch, ]
        :param query_lengths: [batch, ]
        :return: contexts: [batch, q_max_time, attention_dim],
                 alignments: [batch, q_max_time, m_max_time]
        """
        processed_query = self.query_layer(inputs)
        values = self.memory_layer(memory)
        # energy: [batch, q_max_time, m_max_time]
        energy = self._bahdanau_score(processed_query, values) / self.temperature

        # apply mask
        batch_size = tf.shape(memory)[0]
        memory_max_time = tf.shape(memory)[1]
        query_max_time = tf.shape(inputs)[1]
        length_mask = self._get_key_mask(
            batch_size, memory_max_time, query_max_time, memory_lengths, query_lengths)
        # [batch, q_max_time, m_max_time]
        paddings = tf.ones_like(energy) * (-2. ** 32 + 1)
        energy = tf.where(length_mask, energy, paddings)
        # alignments shape = energy shape = [batch, q_max_time, m_max_time]
        alignments = tf.math.softmax(energy, axis=2)

        # compute context vector
        # context: [batch, q_max_time, attention_dim]
        contexts = tf.linalg.matmul(alignments, values)

        return contexts, alignments


class ScaledDotProductAttention(BaseAttention):
    def __init__(self, attention_dim, value_dim, temperature=1.0,
                 name='ScaledDotProductAttention', **kwargs):
        super(ScaledDotProductAttention, self).__init__(
            attention_dim=attention_dim, name=name, **kwargs)
        self.query_layer = tf.keras.layers.Dense(
            units=attention_dim, use_bias=False, name='query_layer')
        self.key_layer = tf.keras.layers.Dense(
            units=attention_dim, use_bias=False, name='key_layer')
        self.value_layer = tf.keras.layers.Dense(
            units=value_dim, use_bias=False, name='value_layer')
        self.temperature = temperature

    def call(self, inputs, memory, memory_lengths=None, query_lengths=None):
        queries = self.query_layer(inputs)  # [batch, Tq, D]
        keys = self.key_layer(memory)  # [batch, Tk, D]
        values = self.key_layer(memory)  # [batch, Tk, Dv]
        logits = tf.linalg.matmul(queries, keys, transpose_b=True)  # [batch, Tq, Tk]
        logits = logits / tf.math.sqrt(tf.cast(self.attention_dim, tf.float32))  # scale
        logits = logits / self.temperature  # temperature
        # apply mask
        batch_size = tf.shape(memory)[0]
        memory_max_time = tf.shape(memory)[1]
        query_max_time = tf.shape(inputs)[1]
        length_mask = self._get_key_mask(
            batch_size, memory_max_time, query_max_time, memory_lengths, query_lengths)
        paddings = tf.ones_like(logits) * (-2. ** 32 + 1)
        logits = tf.where(length_mask, logits, paddings)
        alignments = tf.math.softmax(logits, axis=2)
        contexts = tf.linalg.matmul(alignments, values)
        return contexts, alignments


class MultiHeadScaledProductAttention(BaseAttention):
    def __init__(self, attention_dim, num_head, temperature=1.0, causality=None,
                 attention_window=-1, name='attention', **kwargs):
        assert attention_dim % num_head == 0
        super(MultiHeadScaledProductAttention, self).__init__(
            attention_dim=attention_dim, name=name, **kwargs)
        self.causality = causality
        self.attention_window = attention_window
        self.query_layer = tf.keras.layers.Dense(
            units=attention_dim, use_bias=False, name='query_layer')
        self.key_layer = tf.keras.layers.Dense(
            units=attention_dim, use_bias=False, name='key_layer')
        self.value_layer = tf.keras.layers.Dense(
            units=attention_dim, use_bias=False, name='value_layer')
        self.num_head = num_head
        self.temperature = temperature

    def _split_head(self, inputs):
        """
        :param inputs: [batch, time, dim]
        :return: [batch, num_head, time, dim // head]
        """
        batch = tf.shape(inputs)[0]
        max_time = tf.shape(inputs)[1]
        dim = tf.shape(inputs)[2]
        reshaped = tf.reshape(inputs,
                              [batch, max_time, self.num_head,
                               dim // self.num_head])
        # [batch, time, num_head, dim // head]
        transposed = tf.transpose(reshaped, [0, 2, 1, 3])
        # [batch, num_head, time, dim // head]
        return transposed

    def _merge_head(self, inputs):
        """
        :param inputs: [batch, num_head, time, dim]
        :return: [batch, time, attention_dim]
        """
        batch = tf.shape(inputs)[0]
        time = tf.shape(inputs)[2]
        head_dim = tf.shape(inputs)[3]
        transposed = tf.transpose(inputs, [0, 2, 1, 3])
        # [batch, time, num_head, dim]
        reshaped = tf.reshape(transposed, [batch, time, self.num_head * head_dim])
        return reshaped

    def _get_key_mask(self, batch_size, memory_max_time, query_max_time,
                      memory_lengths, query_lengths):
        memory_lengths = (memory_lengths if memory_lengths is not None
                          else tf.ones([batch_size], dtype=tf.int32) * memory_max_time)
        memory_mask = tf.sequence_mask(memory_lengths, maxlen=memory_max_time,
                                       name='length_mask')  # [batch, m_max_time]
        memory_mask = tf.tile(tf.expand_dims(memory_mask, axis=1),  # [batch, 1, m_max_time]
                              [1, query_max_time, 1])  # [batch, q_max_time, m_max_time]
        query_lengths = (query_lengths if query_lengths is not None
                         else tf.ones([batch_size], dtype=tf.int32) * query_max_time)
        query_mask = tf.sequence_mask(query_lengths, maxlen=query_max_time)  # [batch, q_max_time]
        query_mask = tf.tile(tf.expand_dims(query_mask, axis=2),  # [batch, q_max_time, 1]
                             [1, 1, memory_max_time])  # [batch, q_max_time, m_max_time]
        length_mask = tf.logical_and(memory_mask, query_mask)
        length_mask = tf.tile(tf.expand_dims(length_mask, axis=1),
                              [1, self.num_head, 1, 1])
        # [batch, num_head, q_max_time, m_max_time]
        return length_mask

    def get_window_mask(self, logits):
        win_mask = tf.ones(tf.shape(logits), dtype=tf.bool)
        if self.causality:
            win_mask = tf.linalg.band_part(win_mask, self.attention_window, 0, name='causal_mask')
        else:
            win_mask = tf.linalg.band_part(
                win_mask, self.attention_window, self.attention_window, name='causal_mask')
        return win_mask

    def call(self, inputs, memory, memory_lengths=None, query_lengths=None):
        queries = self.query_layer(inputs)  # [batch, Tq, D]
        keys = self.key_layer(memory)  # [batch, Tk, D]
        values = self.value_layer(memory)  # [batch, Tk, Dv]
        headed_queries = self._split_head(queries)  # [batch, num_head, Tq, head_dim]
        headed_keys = self._split_head(keys)  # [batch, num_head, Tk, head_dim]
        headed_values = self._split_head(values)  # [batch, num_head, Tk, head_dim]
        logits = tf.linalg.matmul(headed_queries,
                                  headed_keys,
                                  transpose_b=True)  # [batch, num_head, Tq, Tk]
        logits = logits / tf.sqrt(
            tf.cast(self.attention_dim // self.num_head, dtype=tf.float32))  # scale
        logits = logits / self.temperature  # temperature
        # apply mask
        batch_size = tf.shape(memory)[0]
        memory_max_time = tf.shape(memory)[1]
        query_max_time = tf.shape(inputs)[1]
        length_mask = self._get_key_mask(
            batch_size, memory_max_time, query_max_time, memory_lengths, query_lengths)
        window_mask = self.get_window_mask(logits)
        length_mask = tf.math.logical_and(length_mask, window_mask)
        # [batch, num_head, q_max_time, m_max_time]
        paddings = tf.ones_like(logits, dtype=tf.float32) * (-2. ** 32 + 1)
        logits = tf.where(length_mask, logits, paddings)
        alignments = tf.math.softmax(logits, axis=3)  # [batch, num_head, Tq, Tk]
        contexts = tf.linalg.matmul(alignments, headed_values)
        # [batch, num_head, Tq, head_dim]
        contexts = self._merge_head(contexts)  # [batch, Tq, attention_dim]
        return contexts, alignments


class SelfAttentionBLK(tf.keras.layers.Layer):
    def __init__(self, input_dim, attention_dim, attention_heads, attention_temperature, ffn_hidden,
                 causality=None, attention_window=-1, name='self_attention_blk', **kwargs):
        super(SelfAttentionBLK, self).__init__(name=name, **kwargs)
        self.input_dim = input_dim
        self.attention_dim = attention_dim
        self.attention = MultiHeadScaledProductAttention(attention_dim=attention_dim,
                                                         num_head=attention_heads,
                                                         temperature=attention_temperature,
                                                         causality=causality,
                                                         attention_window=attention_window)
        self.att_proj = tf.keras.layers.Dense(units=input_dim)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.ffn = FFN(hidden1=ffn_hidden, hidden2=input_dim)

    def call(self, inputs, memory, query_lengths, memory_lengths, training=None):
        att_outs, alignments = self.attention(inputs=inputs, memory=memory,
                                              query_lengths=query_lengths,
                                              memory_lengths=memory_lengths)
        contexts = tf.concat([inputs, att_outs], axis=-1)
        contexts.set_shape([None, None, self.attention_dim + self.input_dim])
        att_outs = self.att_proj(contexts)
        att_outs = self.layer_norm(inputs + att_outs, training=training)
        ffn_outs = self.ffn(att_outs, training=training)
        return ffn_outs, alignments


class CrossAttentionBLK(tf.keras.layers.Layer):
    def __init__(self, input_dim, attention_dim, attention_heads, attention_temperature,
                 ffn_hidden, causality=None, attention_window=-1, name='cross_attention_blk', **kwargs):
        super(CrossAttentionBLK, self).__init__(name=name, **kwargs)
        self.input_dim = input_dim
        self.attention_dim = attention_dim
        self.self_attention = MultiHeadScaledProductAttention(
            attention_dim=attention_dim, num_head=attention_heads,
            temperature=attention_temperature, name='self_attention',
            causality=causality, attention_window=attention_window)
        self.att_proj1 = tf.keras.layers.Dense(units=input_dim)
        self.layer_norm1 = tf.keras.layers.LayerNormalization(name='layerNorm1')
        self.cross_attention = MultiHeadScaledProductAttention(
            attention_dim=attention_dim, num_head=attention_heads,
            temperature=attention_temperature, causality=False,
            attention_window=-1, name='cross_attention')
        self.att_proj2 = tf.keras.layers.Dense(units=attention_dim)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(name='layerNorm2')
        self.ffn = FFN(hidden1=ffn_hidden, hidden2=attention_dim)

    def call(self, inputs, memory, query_lengths, memory_lengths, training=None):
        self_att_outs, self_ali = self.self_attention(
            inputs=inputs, memory=inputs, query_lengths=query_lengths,
            memory_lengths=query_lengths)
        contexts = tf.concat([inputs, self_att_outs], axis=-1)
        contexts.set_shape([None, None, self.attention_dim + self.input_dim])
        self_att_outs = self.att_proj1(contexts)
        self_att_outs = self.layer_norm1(self_att_outs + inputs, training=training)
        att_outs, cross_ali = self.cross_attention(
            inputs=self_att_outs, memory=memory, query_lengths=query_lengths,
            memory_lengths=memory_lengths)
        contexts = tf.concat([self_att_outs, att_outs], axis=-1)
        contexts.set_shape([None, None, self.attention_dim * 2])
        att_outs = self.att_proj2(contexts)
        att_outs = self.layer_norm2(att_outs + self_att_outs, training=training)
        ffn_outs = self.ffn(att_outs, training=training)
        return ffn_outs, cross_ali
