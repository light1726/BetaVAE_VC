import os
import argparse
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from configs import CNENHPS
from models import BetaVAEVC
from datasets import TFRecordWriter


def main(args):
    hparams = CNENHPS()
    data_records = TFRecordWriter(
        save_dir=args.data_dir, chunk_size=hparams.Dataset.chunk_size)
    val_set = data_records.create_dataset(
        buffer_size=hparams.Dataset.buffer_size,
        num_parallel_reads=hparams.Dataset.num_parallel_reads,
        batch_size=hparams.Train.test_batch_size,
        num_mels=hparams.Audio.num_mels,
        shuffle_buffer=hparams.Train.shuffle_buffer,
        shuffle=hparams.Train.shuffle,
        tfrecord_files=data_records.get_tfrecords_list('val'),
        seed=hparams.Train.random_seed,
        drop_remainder=False)
    test_set = data_records.create_dataset(
        buffer_size=hparams.Dataset.buffer_size,
        num_parallel_reads=hparams.Dataset.num_parallel_reads,
        batch_size=hparams.Train.test_batch_size,
        num_mels=hparams.Audio.num_mels,
        shuffle_buffer=hparams.Train.shuffle_buffer,
        shuffle=hparams.Train.shuffle,
        tfrecord_files=data_records.get_tfrecords_list('test'),
        seed=hparams.Train.random_seed,
        drop_remainder=False)

    ckpt_path = args.ckpt_path
    save_dir = args.save_dir
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    cn_dir = os.path.join(save_dir, 'CN')
    en_dir = os.path.join(save_dir, 'EN')
    os.makedirs(cn_dir, exist_ok=True)
    os.makedirs(en_dir, exist_ok=True)
    # setup model
    model = BetaVAEVC(hparams)
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(ckpt_path).expect_partial()

    def save_spk_npy(arrs, fids):
        for a, n in zip(arrs.numpy(), fids.numpy()):
            n = n.decode('utf-8') if type(n) is bytes else n
            if n.startswith('SSB'):
                save_name = os.path.join(cn_dir, '{}-spk.npy'.format(n))
            else:
                save_name = os.path.join(en_dir, '{}-spk.npy'.format(n))
            np.save(save_name, a)
        return

    def save_content_npy(arrs, lens, fids):
        for a, l, n in zip(arrs.numpy(), lens.numpy(), fids.numpy()):
            n = n.decode('utf-8') if type(n) is bytes else n
            if n.startswith('SSB'):
                save_name = os.path.join(cn_dir, '{}-content.npy'.format(n))
            else:
                save_name = os.path.join(en_dir, '{}-content.npy'.format(n))
            np.save(save_name, a[:l, :])
        return

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, hparams.Audio.num_mels], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, hparams.Audio.num_mels], dtype=tf.float32),
        tf.TensorSpec(shape=[None], dtype=tf.int32)])
    def inference(mels, mels_ext, mel_lengths):
        spk_mu, _ = model.spk_posterior(mels_ext)
        reduced_mels = mels[:, ::model.reduction_factor, :]
        reduced_lens = (mel_lengths + model.reduction_factor - 1) // model.reduction_factor
        content_mu, _, _ = model.posterior(reduced_mels, lengths=reduced_lens, training=False)
        return spk_mu, content_mu, reduced_lens

    for dataset in [val_set, test_set]:
        for _fids, _mels, _m_lengths, _mels_ext in tqdm(dataset):
            spk_emb, content_emb, reduced_lengths = inference(_mels, _mels_ext, _m_lengths)
            save_spk_npy(spk_emb, _fids)
            save_content_npy(content_emb, reduced_lengths, _fids)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--data_dir', type=str, help='Tf-Records directory')
    parser.add_argument('--ckpt_path', type=str, help='path to the model ckpt')
    parser.add_argument('--save_dir', type=str, help='directory to save test results')
    main_args = parser.parse_args()
    main(main_args)
