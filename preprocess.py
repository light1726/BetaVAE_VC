import os
import random
import numpy as np
import warnings
from configs import CNENHPS
from datasets import VCTK, AiShell3, MelPreprocessor, TFRecordWriter

warnings.filterwarnings("ignore")


def main():
    hps = CNENHPS()
    random.seed(hps.Train.random_seed)
    np.random.seed(hps.Train.random_seed)
    vctk_writer = VCTK(
        data_dir=hps.Dataset.VCTK.corpus_dir,
        out_dir=hps.Dataset.dataset_dir,
        val_spks=hps.Dataset.VCTK.val_spks,
        test_spks=hps.Dataset.VCTK.test_spks)
    vctk_writer.write_summary()
    aishell_writer = AiShell3(
        data_dir=hps.Dataset.AiShell3.corpus_dir,
        out_dir=hps.Dataset.dataset_dir,
        train_spk_file=None,
        val_test_spk_file=hps.Dataset.AiShell3.val_test_spk_file)
    aishell_writer.write_summary()
    feats_extractor = MelPreprocessor(
        [vctk_writer.summary_file, aishell_writer.summary_file],
        save_dir=hps.Dataset.dataset_dir, hps=hps)
    feats_extractor.feature_extraction()
    tfrecord_save_dir = os.path.join(hps.Dataset.dataset_dir, 'tfrecords')
    if not os.path.exists(tfrecord_save_dir):
        os.makedirs(tfrecord_save_dir)
    tfrecord_writer = TFRecordWriter(
        train_split=hps.Dataset.n_record_split,
        data_dir=hps.Dataset.dataset_dir,
        save_dir=tfrecord_save_dir,
        chunk_size=hps.Dataset.chunk_size)
    tfrecord_writer.write_all()

    # test
    print('TFRecord test...')
    tf_dataset = tfrecord_writer.create_dataset(
        buffer_size=hps.Dataset.buffer_size,
        num_parallel_reads=hps.Dataset.num_parallel_reads,
        batch_size=hps.Train.test_batch_size,
        num_mels=hps.Audio.num_mels,
        shuffle_buffer=hps.Train.shuffle_buffer,
        shuffle=hps.Train.shuffle,
        tfrecord_files=tfrecord_writer.get_tfrecords_list('test'))
    for epoch in range(2):
        for i, data in enumerate(tf_dataset):
            print('epoch {}, step: {}'.format(epoch, i))
            fid, mel, mel_len, mel_ext = data
            print(fid.numpy(), mel.shape, mel_len.numpy(), mel_ext.shape)


if __name__ == '__main__':
    main()
