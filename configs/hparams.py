class CNENHPS:
    class Train:
        random_seed = 123
        epochs = 1000
        train_batch_size = 32
        test_batch_size = 8
        test_interval = 100
        ckpt_interval = 50
        shuffle_buffer = 128
        shuffle = True
        num_samples = 1
        length_weight = 1.
        content_kl_weight = 5e-3
        spk_kl_weight = 1e-5
        learning_rate = 1.25e-4

    class Dataset:
        buffer_size = 65536
        num_parallel_reads = 64
        dev_set_rate = 0.05
        test_set_rate = 0.05
        chunk_size = 256  # the length of the mel-sepctrogram that is required by speaker encoder
        segment_size = 16  # the length of the smallest unit to segment and shuffle the chunk
        n_record_split = 32
        preprocess_n_jobs = 16

        # define dataset specifics below
        class VCTK:
            corpus_dir = '/path/to/extracted/vctk'
            val_spks = ['p225', 'p243', 'p231', 'p251', 'p258', 'p271', 'p284', 'p326', 'p374', 'p334']
            test_spks = ['p274', 'p293', 'p360', 'p262', 'p314', 'p239',  'p273', 'p302', 'p270', 'p340']

        class AiShell3:
            corpus_dir = '/path/to/extracted/data_aishell3'
            train_spk_file = './train-speakers.txt'
            val_test_spk_file = './test-speakers.txt'

        dataset_dir = '/path/to/save/features'

    class Audio:
        num_mels = 80
        num_freq = 1025
        min_mel_freq = 0.
        max_mel_freq = 8000.
        sample_rate = 16000
        frame_length_sample = 800
        frame_shift_sample = 200
        preemphasize = 0.97
        sil_trim_db = 20.
        min_level_db = -100.0
        ref_level_db = 20.0
        max_abs_value = 1
        symmetric_specs = False
        griffin_lim_iters = 60
        power = 1.5
        center = True

    class Common:
        latent_dim = 128
        output_dim = 80
        reduction_factor = 2

    class Decoder:
        class Transformer:
            pre_n_conv = 2
            pre_conv_kernel = 3
            pre_drop_rate = 0.2
            nblk = 2
            attention_dim = 256
            attention_heads = 4
            ffn_hidden = 1024
            attention_temperature = 1.
            attention_causality = False
            attention_window = 16
            post_n_conv = 5
            post_conv_filters = 256
            post_conv_kernel = 5
            post_drop_rate = 0.2

    class ContentPosterior:
        class Transformer:
            pre_n_conv = 2
            pre_conv_kernel = 3
            pre_hidden = 256
            pre_drop_rate = 0.2
            pos_drop_rate = 0.2
            nblk = 2
            attention_dim = 256
            attention_heads = 4
            temperature = 1.0
            attention_causality = False
            attention_window = 8
            ffn_hidden = 1024

    class SpkPosterior:
        class ConvSpkEncoder:
            hidden_channels = 256
            conv_kernels = [3, 3, 5, 5]
            activation = 'relu'
