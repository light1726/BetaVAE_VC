import tensorflow as tf
from modules.decoder import TransformerDecoder
from modules.posterior import TransformerPosterior, ConvSpkEncoder


class BetaVAEVC(tf.keras.Model):
    def __init__(self, hps, name='BetaVAEVC', **kwargs):
        super(BetaVAEVC, self).__init__(name=name, **kwargs)
        self.hps = hps
        self.n_sample = hps.Train.num_samples
        self.reduction_factor = hps.Common.reduction_factor
        self.chunk_size = hps.Dataset.chunk_size
        self.segment_size = hps.Dataset.segment_size
        assert self.chunk_size % self.segment_size == 0
        self.spk_posterior = ConvSpkEncoder(
            hidden_channels=hps.SpkPosterior.ConvSpkEncoder.hidden_channels,
            conv_kernels=hps.SpkPosterior.ConvSpkEncoder.conv_kernels,
            out_channels=hps.Common.latent_dim,
            activation=hps.SpkPosterior.ConvSpkEncoder.activation)
        self.decoder = TransformerDecoder(
            pre_n_conv=hps.Decoder.Transformer.pre_n_conv,
            pre_conv_kernel=hps.Decoder.Transformer.pre_conv_kernel,
            pre_drop_rate=hps.Decoder.Transformer.pre_drop_rate,
            nblk=hps.Decoder.Transformer.nblk,
            attention_dim=hps.Decoder.Transformer.attention_dim,
            attention_heads=hps.Decoder.Transformer.attention_heads,
            temperature=hps.Decoder.Transformer.attention_temperature,
            attention_causality=hps.Decoder.Transformer.attention_causality,
            attention_window=hps.Decoder.Transformer.attention_window,
            ffn_hidden=hps.Decoder.Transformer.ffn_hidden,
            post_n_conv=hps.Decoder.Transformer.post_n_conv,
            post_conv_filters=hps.Decoder.Transformer.post_conv_filters,
            post_conv_kernel=hps.Decoder.Transformer.post_conv_kernel,
            post_drop_rate=hps.Decoder.Transformer.post_drop_rate,
            out_dim=hps.Common.output_dim,
            reduction_factor=hps.Common.reduction_factor,
            name='transformer_decoder')
        self.posterior = TransformerPosterior(
            pre_n_conv=hps.ContentPosterior.Transformer.pre_n_conv,
            pre_conv_kernel=hps.ContentPosterior.Transformer.pre_conv_kernel,
            pre_hidden=hps.ContentPosterior.Transformer.pre_hidden,
            pre_drop_rate=hps.ContentPosterior.Transformer.pre_drop_rate,
            pos_drop_rate=hps.ContentPosterior.Transformer.pos_drop_rate,
            nblk=hps.ContentPosterior.Transformer.nblk,
            attention_dim=hps.ContentPosterior.Transformer.attention_dim,
            attention_heads=hps.ContentPosterior.Transformer.attention_heads,
            temperature=hps.ContentPosterior.Transformer.temperature,
            attention_causality=hps.ContentPosterior.Transformer.attention_causality,
            attention_window=hps.ContentPosterior.Transformer.attention_window,
            ffn_hidden=hps.ContentPosterior.Transformer.ffn_hidden,
            latent_dim=hps.Common.latent_dim)

    def _compute_l2_loss(self, reconstructed, targets, lengths=None, reduce=None):
        max_time = tf.shape(reconstructed)[1]
        dim = tf.shape(reconstructed)[2]
        r = tf.reshape(reconstructed, [-1, self.n_sample, max_time, dim])
        t = tf.reshape(targets, [-1, self.n_sample, max_time, dim])
        if lengths is not None:
            seq_mask = tf.sequence_mask(lengths, max_time, dtype=tf.float32)
            seq_mask = tf.reshape(seq_mask, [-1, self.n_sample, max_time])
            reshaped_lens = tf.reshape(lengths, [-1, self.n_sample])
            l2_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.reduce_mean(tf.square(r - t) + tf.abs(r - t), axis=-1) * seq_mask,
                    axis=-1) / tf.cast(reshaped_lens, tf.float32),
                axis=-1)
        else:
            l2_loss = tf.math.reduce_mean(tf.square(r - t) + tf.abs(r - t), axis=[1, 2, 3])
        if reduce:
            return tf.math.reduce_mean(l2_loss)
        else:
            return l2_loss

    @staticmethod
    def kl_with_normal(logvar, mu, reduce=None):
        kl = -0.5 * tf.reduce_sum(1. + logvar - mu ** 2 - tf.exp(logvar), axis=1)
        if reduce:
            return tf.reduce_mean(kl)
        else:
            return kl

    @staticmethod
    def kl_between_normal2d(mu1, logvar1, mu2, logvar2, reduce=None):
        var1 = tf.math.exp(logvar1)
        var2 = tf.math.exp(logvar2)
        kl = 0.5 * tf.reduce_sum(
            logvar2 - logvar1 - 1. + var1 / var2 + (mu2 - mu1) ** 2. / var2, axis=1)
        if reduce:
            kl = tf.reduce_mean(kl)
        return kl

    @staticmethod
    def kl_between_normal3d(mu1, logvar1, mu2, logvar2, lengths=None, reduce=None):
        var1 = tf.math.exp(logvar1)
        var2 = tf.math.exp(logvar2)
        kl = 0.5 * tf.reduce_sum(
            logvar2 - logvar1 - 1. + var1 / var2 + (mu2 - mu1) ** 2. / var2, axis=2)
        if lengths is None:
            kl = tf.reduce_mean(kl, axis=1)
        else:
            max_len = tf.shape(mu1)[1]
            mask = tf.sequence_mask(lengths, maxlen=max_len, dtype=tf.float32)
            kl = tf.reduce_sum(kl * mask, axis=1) / tf.cast(lengths, tf.float32)
        if reduce:
            kl = tf.reduce_mean(kl)
        return kl

    def random_shuffle(self, inputs):
        """
        :param inputs: [batch, chunk_size, dim]
        :return:
        """
        bs = tf.shape(inputs)[0]
        mel_dim = self.hps.Audio.num_mels
        reshaped = tf.reshape(
            inputs, [bs, self.chunk_size // self.segment_size, self.segment_size * mel_dim])
        transposed = tf.transpose(reshaped, perm=[1, 0, 2])
        shuffled = tf.random.shuffle(transposed)
        shuffled = tf.transpose(shuffled, perm=[1, 0, 2])
        shuffled = tf.reshape(shuffled, [bs, self.chunk_size, mel_dim])
        return shuffled

    def call(self, inputs, mel_lengths, inp_ext, training=None, reduce_loss=None):
        """
        :param inputs: [batch, mel_max_time, mel_dim]
        :param mel_lengths: [batch, ]
        :param inp_ext: [batch, chunk_size, mel_dim]
        :param training: bool
        :param reduce_loss: bool
        :return: predicted mel: [batch, mel_max_time, mel_dim]
                 loss: float32
        """
        print('Tracing back at Model.call')
        # shape info
        mel_max_len = tf.shape(inputs)[1]
        # reduce the mels
        reduced_mels = inputs[:, ::self.reduction_factor, :]
        reduced_mels.set_shape([None, None, self.hps.Audio.num_mels])
        reduced_mel_lens = (mel_lengths + self.reduction_factor - 1) // self.reduction_factor
        reduced_max_len = tf.shape(reduced_mels)[1]

        # text encoding
        content_mu, content_logvar, _ = self.posterior(
            reduced_mels, lengths=reduced_mel_lens, training=training)
        # samples, eps: [batch, n_sample, mel_max_time, dim]
        samples, eps = self.posterior.reparameterize(
            content_mu, content_logvar, self.n_sample)
        posterior_samples = tf.squeeze(samples, axis=1)

        # compute speaker identity prior
        ref_mel = self.random_shuffle(inp_ext)
        spk_posterior_mu, spk_posterior_logvar = self.spk_posterior(ref_mel)
        spk_posterior_embd = self.spk_posterior.sample(spk_posterior_mu, spk_posterior_logvar)
        spk_kl = self.kl_between_normal2d(
            spk_posterior_mu, spk_posterior_logvar, tf.zeros_like(spk_posterior_mu),
            tf.zeros_like(spk_posterior_logvar), reduce=reduce_loss)

        content_kl = self.kl_between_normal3d(
            content_mu, content_logvar, tf.zeros_like(content_mu),
            tf.zeros_like(content_logvar), reduced_mel_lens, reduce_loss)
        posterior_samples_spk = tf.concat(
            [posterior_samples, tf.tile(tf.expand_dims(spk_posterior_embd, axis=1),
                                        [1, reduced_max_len, 1])], axis=2)
        decoded_initial, decoded_post, _ = self.decoder(
            posterior_samples_spk, reduced_mel_lens, training=training)
        decoded_initial = decoded_initial[:, :mel_max_len, :]
        decoded_post = decoded_post[:, :mel_max_len, :]
        initial_l2_loss = self._compute_l2_loss(
            decoded_initial, inputs, mel_lengths, reduce_loss)
        post_l2_loss = self._compute_l2_loss(
            decoded_post, inputs, mel_lengths, reduce_loss)
        l2_loss = initial_l2_loss + post_l2_loss
        return decoded_post, l2_loss, content_kl, spk_kl

    def post_inference(self, inputs, mel_lengths, ref_mels):
        spk_mu, spk_logs = self.spk_posterior(ref_mels)
        inputs = inputs[:, ::self.reduction_factor, :]
        mel_lengths = (mel_lengths + self.reduction_factor - 1) // self.reduction_factor
        mu, _, _ = self.posterior(
            inputs, lengths=mel_lengths, training=False)
        reduced_max_len = tf.shape(mu)[1]
        mu = tf.concat(
            [mu, tf.tile(tf.expand_dims(spk_mu, axis=1), [1, reduced_max_len, 1])], axis=2)
        _, predicted_mel, dec_alignments = self.decoder(
            inputs=mu, lengths=mel_lengths, training=False)
        return predicted_mel, dec_alignments
