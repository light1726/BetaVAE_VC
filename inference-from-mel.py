import os
import argparse
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from configs import CNENHPS
from audio import TestUtils
from models import BetaVAEVC


def read_mels(mel_list_f):
    mels = []
    mel_names = []
    with open(mel_list_f, 'r', encoding='utf-8') as f:
        for line in f:
            mel = np.load(line.strip()).astype(np.float32)
            mels.append(mel)
            name = line.strip().split('/')[-1].split('.')[0]
            mel_names.append(name)
    return mels, mel_names


def synthesize_from_mel(args):
    ckpt_path = args.ckpt_path
    ckpt_step = ckpt_path.split('-')[-1]
    assert os.path.isfile(args.src_mels)
    assert os.path.isfile(args.ref_mels)
    test_dir = args.test_dir
    if not os.path.isdir(test_dir):
        os.makedirs(test_dir)
    hparams = CNENHPS()
    tester = TestUtils(hparams, args.test_dir)
    # setup model
    model = BetaVAEVC(hparams)
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(ckpt_path).expect_partial()

    # set up tf function
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, hparams.Audio.num_mels], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, hparams.Audio.num_mels], dtype=tf.float32),
        tf.TensorSpec(shape=[None], dtype=tf.int32)])
    def vc(mels, mel_ext, m_lengths):
        out, _ = model.post_inference(mels, m_lengths, mel_ext)
        return out

    src_mels, src_names = read_mels(args.src_mels)
    ref_mels, ref_names = read_mels(args.ref_mels)
    for src_mel, src_name in tqdm(zip(src_mels, src_names)):
        for ref_mel, ref_name in zip(ref_mels, ref_names):
            while ref_mel.shape[0] < hparams.Dataset.chunk_size:
                ref_mel = np.concatenate([ref_mel, ref_mel], axis=0)
            ref_mel = ref_mel[:hparams.Dataset.chunk_size, :]
            assert src_mel.shape[1] == hparams.Audio.num_mels
            src_mel_batch = tf.constant(np.expand_dims(src_mel, axis=0))
            ref_mel_batch = tf.constant(np.expand_dims(ref_mel, axis=0))
            mel_len_batch = tf.constant([src_mel.shape[0]])
            ids = ['{}_to_{}'.format(src_name, ref_name)]
            prediction = vc(src_mel_batch, ref_mel_batch, mel_len_batch)
            tester.write_mels(ckpt_step, prediction.numpy(), mel_len_batch.numpy(), ids, prefix='test')
            # tester.synthesize_and_save_wavs(ckpt_step, prediction.numpy(), mel_len_batch.numpy(), ids, prefix='test')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--ckpt_path', type=str, help='path to the model ckpt')
    parser.add_argument('--test_dir', type=str, help='directory to save test results')
    parser.add_argument('--src_mels', type=str, help='source mel npy file list')
    parser.add_argument('--ref_mels', type=str, help='reference mel npy file list')
    main_args = parser.parse_args()
    synthesize_from_mel(main_args)
