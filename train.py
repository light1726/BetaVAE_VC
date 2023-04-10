import os
import sys
import random
import argparse
import datetime
import numpy as np
import tensorflow as tf

from time import time

from models import BetaVAEVC
from audio import TestUtils
from datasets import TFRecordWriter
from configs import CNENHPS, Logger


def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    return


def set_global_determinism(seed):
    set_seeds(seed=seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    return


def main():
    parser = argparse.ArgumentParser('Training parameters parser')
    parser.add_argument('--data_dir', type=str, help='dataset tfrecord directory')
    parser.add_argument('--out_dir', type=str, help='directory to save logs', default='outputs')
    args = parser.parse_args()

    hparams = CNENHPS()
    # set random seed
    set_global_determinism(hparams.Train.random_seed)

    # validate log directories
    out_dir = args.out_dir
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    model_dir = os.path.join(out_dir, 'models')
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    log_dir = os.path.join(out_dir, 'logs')
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    test_dir = os.path.join(out_dir, 'tests')
    if not os.path.isdir(test_dir):
        os.makedirs(test_dir)

    # set up test utils
    tester = TestUtils(hparams, test_dir)

    # set up logger
    sys.stdout = Logger(log_dir)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_dir = os.path.join(log_dir, current_time, 'train')
    os.makedirs(train_dir)
    val_dir = os.path.join(log_dir, current_time, 'val')
    os.makedirs(val_dir)

    # hyperparameters
    data_records = TFRecordWriter(
        save_dir=args.data_dir, chunk_size=hparams.Dataset.chunk_size)
    train_set = data_records.create_dataset(
        buffer_size=hparams.Dataset.buffer_size,
        num_parallel_reads=hparams.Dataset.num_parallel_reads,
        batch_size=hparams.Train.train_batch_size,
        num_mels=hparams.Audio.num_mels,
        shuffle_buffer=hparams.Train.shuffle_buffer,
        shuffle=hparams.Train.shuffle,
        tfrecord_files=data_records.get_tfrecords_list('train'),
        seed=hparams.Train.random_seed)
    val_set = data_records.create_dataset(
        buffer_size=hparams.Dataset.buffer_size,
        num_parallel_reads=hparams.Dataset.num_parallel_reads,
        batch_size=hparams.Train.train_batch_size,
        num_mels=hparams.Audio.num_mels,
        shuffle_buffer=hparams.Train.shuffle_buffer,
        shuffle=hparams.Train.shuffle,
        tfrecord_files=data_records.get_tfrecords_list('val'),
        seed=hparams.Train.random_seed)
    test_set = data_records.create_dataset(
        buffer_size=hparams.Dataset.buffer_size,
        num_parallel_reads=hparams.Dataset.num_parallel_reads,
        batch_size=hparams.Train.test_batch_size,
        num_mels=hparams.Audio.num_mels,
        shuffle_buffer=hparams.Train.shuffle_buffer,
        shuffle=hparams.Train.shuffle,
        tfrecord_files=data_records.get_tfrecords_list('test'),
        seed=hparams.Train.random_seed,
        drop_remainder=True)

    # 2. setup model
    model = BetaVAEVC(hparams)
    learning_rate = hparams.Train.learning_rate
    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

    # 3. define training step
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, hparams.Audio.num_mels], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, hparams.Audio.num_mels], dtype=tf.float32),
        tf.TensorSpec(shape=[None], dtype=tf.int32)])
    def train_step(mels, mel_ext, m_lengths):
        print('Tracing back at train_step')
        with tf.GradientTape() as tape:
            predictions, mel_l2, content_kl, spk_kl = model(
                inputs=mels, mel_lengths=m_lengths, inp_ext=mel_ext,
                training=True, reduce_loss=True)
            loss = (mel_l2 + hparams.Train.content_kl_weight * content_kl
                    + hparams.Train.spk_kl_weight * spk_kl)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss, mel_l2, content_kl, spk_kl

    # 4. define validate step
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, hparams.Audio.num_mels], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, hparams.Audio.num_mels], dtype=tf.float32),
        tf.TensorSpec(shape=[None], dtype=tf.int32)])
    def val_step(mels, mel_ext, m_lengths):
        print('Tracing back at val step')
        predictions, mel_l2, content_kl, spk_kl = model(
            inputs=mels, mel_lengths=m_lengths, inp_ext=mel_ext,
            training=False, reduce_loss=True)
        loss = (mel_l2 + hparams.Train.content_kl_weight * content_kl
                + hparams.Train.spk_kl_weight * spk_kl)
        return loss, mel_l2, content_kl, spk_kl

    # @tf.function
    def train_one_epoch(dataset):
        # print('tracing back at train_one_epoch')
        step = 0
        total = 0.0
        mel_l2 = 0.0
        kl = 0.0
        spk_kl = 0.
        for _, train_mels, train_m_lengths, train_mel_ext in dataset:
            step_start = time()
            _total, _mel_l2, _kl, _spk_kl = train_step(
                train_mels, train_mel_ext, train_m_lengths)
            step_end = time()
            print('Step {}: total {:.4f}, mel-l2 {:.4f}, content-kl {:.4f},'
                  ' spk-kl: {:.4f}, time {:.4f}'.format(
                step, _total.numpy(), _mel_l2.numpy(), _kl.numpy(),
                _spk_kl.numpy(), step_end - step_start))
            step += 1
            total += _total.numpy()
            mel_l2 += _mel_l2.numpy()
            kl += _kl.numpy()
            spk_kl += _spk_kl.numpy()
        return total / step, mel_l2 / step, kl / step, spk_kl / step

    # @tf.function
    def val_one_epoch(dataset):
        step = 0
        total = 0.0
        mel_l2 = 0.0
        kl = 0.0
        spk_kl = 0.
        for _, val_mels, val_m_lengths, val_mel_ext in dataset:
            _total, _mel_l2, _kl, _spk_kl = val_step(
                val_mels, val_mel_ext, val_m_lengths)
            step += 1
            total += _total.numpy()
            mel_l2 += _mel_l2.numpy()
            kl += _kl.numpy()
            spk_kl += _spk_kl.numpy()
        return total / step, mel_l2 / step, kl / step, spk_kl / step

    # 8. setup summary writer
    train_summary_writer = tf.summary.create_file_writer(train_dir)
    val_summary_writer = tf.summary.create_file_writer(val_dir)

    # 9. setup checkpoint: all workers will need checkpoint manager to load checkpoint
    checkpoint = tf.train.Checkpoint(step=tf.Variable(0, dtype=tf.int64, trainable=False),
                                     optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(checkpoint, model_dir, max_to_keep=20)
    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
        step = checkpoint.step.numpy()
    else:
        print("Initializing from scratch.")
        step = 0

    # 8. start training
    for epoch in range(step + 1, hparams.Train.epochs + 1):
        print('Training Epoch {} ...'.format(epoch))
        epoch_start = time()
        train_total, train_mel_l2, train_kl, train_spk_kl = train_one_epoch(train_set)
        epoch_dur = time() - epoch_start
        print('\nTraining Epoch {} finished in {:.3f} Secs'.format(epoch, epoch_dur))
        # save summary and evaluate
        with train_summary_writer.as_default():
            tf.summary.scalar('total-loss', train_total, step=epoch)
            tf.summary.scalar('recon-loss', train_mel_l2, step=epoch)
            tf.summary.scalar('content-kl', train_kl, step=epoch)
            tf.summary.scalar('speaker-kl', train_spk_kl, step=epoch)

        # validation
        print('Validation ...')
        val_start = time()
        val_total, val_mel_l2, val_kl, val_spk_kl = val_one_epoch(val_set)
        print('Validation finished in {:.3f} Secs'.format(time() - val_start))
        with val_summary_writer.as_default():
            tf.summary.scalar('total-loss', val_total, step=epoch)
            tf.summary.scalar('recon-loss', val_mel_l2, step=epoch)
            tf.summary.scalar('content-kl', val_kl, step=epoch)
            tf.summary.scalar('speaker-kl', val_spk_kl, step=epoch)

        print('Epoch {}: l2 {:.4f} / {:.4f}, content-kl {:.4f} / {:.4f}, spk-kl: {:.4f} / {:.4f}'.format(
            epoch, train_mel_l2, val_mel_l2, train_kl, val_kl, train_spk_kl, val_spk_kl))

        if epoch % hparams.Train.ckpt_interval == 0:
            # save checkpoint
            save_path = manager.save(checkpoint_number=epoch)
            print("Saved checkpoint for epoch {}: {}".format(epoch, save_path))

        # test
        if epoch % hparams.Train.test_interval == 0:
            print('Testing ...')
            i = 0
            ref_mels = None
            ref_spk_ids = None
            for test_ids, test_mels, test_m_lengths, test_mel_ext in test_set.take(2):
                if i == 0:
                    ref_mels = test_mel_ext
                    ref_spk_ids = test_ids
                    i += 1
                    continue
                post_mel, _ = model.post_inference(test_mels, test_m_lengths, ref_mels)
                fids = [sid.decode('utf-8') + '-ref-' + rid.decode('utf-8')
                        for sid, rid in zip(test_ids.numpy(), ref_spk_ids.numpy())]
                try:
                    tester.synthesize_and_save_wavs(
                        epoch, post_mel.numpy(), test_m_lengths.numpy(), fids, 'post')
                except:
                    print('Something wrong with the generated waveform!')
            print('test finished, check {} for the results'.format(test_dir))


if __name__ == '__main__':
    main()
