import os
import numpy as np
import tensorflow as tf

from tqdm import tqdm


class TFRecordWriter:
    def __init__(self, train_split=None, data_dir=None, save_dir=None, chunk_size=None):
        self.train_split = train_split
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.chunk_size = chunk_size
        self.train_ids_file = os.path.join(self.data_dir, 'train.txt') if data_dir is not None else None
        self.val_ids_file = os.path.join(self.data_dir, 'val.txt') if data_dir is not None else None
        self.test_ids_file = os.path.join(self.data_dir, 'test.txt') if data_dir is not None else None

    @staticmethod
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @staticmethod
    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def serialize_example(fid, mel, mel_len):
        """
        :param fid: string
        :param mel: np array, [mel_len, num_mels]
        :param mel_len: int32
        :return: byte string
        """
        feature = {
            'fid': TFRecordWriter._bytes_feature(fid.encode('utf-8')),
            'mel': TFRecordWriter._bytes_feature(tf.io.serialize_tensor(mel)),
            'mel_len': TFRecordWriter._int64_feature(mel_len)}
        # Create a Features message using tf.train.Example.
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    def _parse_fids(self, mode='train'):
        fids_f = {'train': self.train_ids_file,
                  'val': self.val_ids_file,
                  'test': self.test_ids_file}[mode]
        fids = []
        with open(fids_f, 'r', encoding='utf-8') as f:
            for line in f:
                fids.append(line.strip())
        return fids

    def _get_features(self, fid):
        mel = np.load(os.path.join(self.data_dir, 'mels', '{}.npy'.format(fid))).astype(np.float64)
        mel_len = mel.shape[0]
        return mel, mel_len

    def mel_exist(self, fid):
        mel_npy = os.path.join(self.data_dir, 'mels', '{}.npy'.format(fid))
        return os.path.isfile(mel_npy)

    def write(self, mode='train'):
        fids = self._parse_fids(mode)
        if mode == 'train':
            splited_fids = [fids[i::self.train_split] for i in range(self.train_split)]
        else:
            splited_fids = [fids]
        for i, ids in enumerate(splited_fids):
            tfrecord_path = os.path.join(self.save_dir, '{}-{}.tfrecords'.format(mode, i))
            with tf.io.TFRecordWriter(tfrecord_path) as writer:
                for fid in tqdm(ids):
                    if not self.mel_exist(fid):
                        continue
                    mel, mel_len = self._get_features(fid)
                    serialized_example = self.serialize_example(fid, mel, mel_len)
                    writer.write(serialized_example)
        return

    def write_all(self):
        self.write('train')
        self.write('val')
        self.write('test')
        return

    def pad2chunk(self, inputs):
        inp_exp = tf.tile(inputs, tf.constant([14, 1]))
        inp_exp = inp_exp[:self.chunk_size, :]
        return inp_exp

    def parse_example(self, serialized_example):
        feature_description = {
            'fid': tf.io.FixedLenFeature((), tf.string),
            'mel': tf.io.FixedLenFeature((), tf.string),
            'mel_len': tf.io.FixedLenFeature((), tf.int64)}
        example = tf.io.parse_single_example(serialized_example, feature_description)

        fid = example['fid']
        mel = tf.io.parse_tensor(example['mel'], out_type=tf.float64)
        mel_ext = self.pad2chunk(mel)
        mel_len = example['mel_len']
        return (fid,
                tf.cast(mel, tf.float32),
                tf.cast(mel_len, tf.int32),
                tf.cast(mel_ext, tf.float32))

    def create_dataset(self, buffer_size, num_parallel_reads,
                       batch_size, num_mels, shuffle_buffer, shuffle,
                       tfrecord_files, seed=1, drop_remainder=False):
        tfrecord_dataset = tf.data.TFRecordDataset(
            tfrecord_files, buffer_size=buffer_size,
            num_parallel_reads=num_parallel_reads)
        tfdataset = tfrecord_dataset.map(
            self.parse_example,
            num_parallel_calls=num_parallel_reads)
        tfdataset = tfdataset.padded_batch(
            batch_size=batch_size,
            drop_remainder=drop_remainder,
            padded_shapes=([], [None, num_mels], [], [None, num_mels]))
        tfdataset = (tfdataset.shuffle(buffer_size=shuffle_buffer, seed=seed)
                     if shuffle else tfdataset)
        tfdataset = tfdataset.prefetch(tf.data.experimental.AUTOTUNE)
        return tfdataset

    def get_tfrecords_list(self, mode='train'):
        assert self.save_dir is not None
        assert mode in ['train', 'val', 'test']
        return [os.path.join(self.save_dir, f)
                for f in os.listdir(self.save_dir) if f.startswith(mode)]
