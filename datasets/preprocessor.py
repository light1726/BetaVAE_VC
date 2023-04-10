import os
import json
import joblib
import random
import numpy as np
from audio import Audio


class MelPreprocessor:
    """
    Extract mel-spectrograms given the multiple data summary json file
    """
    def __init__(self, data_summary_paths, save_dir, hps):
        """
        :param data_summary_paths: list of data summary json files containing paths of the waveform
        :param save_dir: directory to save the feature
        :param hps: hyper-parameters
        """
        self.save_dir = save_dir
        self.data_summary_paths = data_summary_paths
        self.data_summary = self.load_dataset_info()
        self.hps = hps
        self.mel_dir = os.path.join(self.save_dir, 'mels')
        self.train_list_f = os.path.join(self.save_dir, 'train.txt')
        self.val_list_f = os.path.join(self.save_dir, 'val.txt')
        self.test_list_f = os.path.join(self.save_dir, 'test.txt')
        self.num_mels = hps.Audio.num_mels
        self.audio_processor = Audio(hps.Audio)
        self.n_jobs = hps.Dataset.preprocess_n_jobs
        self.train_set_size = None
        self.dev_set_size = None
        self.test_set_size = None

    def feature_extraction(self):
        self._validate_dir()
        print('Process text file...')
        print('Split the data set into train, dev and test set...')
        self.train_set_size, self.dev_set_size, self.test_set_size = self.write_splits()
        print('Extracting Mel-Spectrograms...')
        self.extract_mels()
        return

    def load_dataset_info(self):
        train_summary = {}
        val_summary = {}
        test_summary = {}
        for summary_f in self.data_summary_paths:
            if not os.path.isfile(summary_f):
                raise FileNotFoundError(
                    '{} not exists! Please generate it first!'.format(summary_f))
            with open(summary_f, 'r') as f:
                summary = json.load(f)
                train_summary.update(summary['train'])
                val_summary.update(summary['validation'])
                test_summary.update(summary['test'])
        dataset_summary = {'train': train_summary, 'validation': val_summary, 'test': test_summary}
        return dataset_summary

    def _validate_dir(self):
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        if not os.path.isdir(self.mel_dir):
            os.makedirs(self.mel_dir)
        return

    def write_splits(self):
        train_set = [fid for fid in self.data_summary['train'].keys()]
        val_set = [fid for fid in self.data_summary['validation'].keys()]
        test_set = [fid for fid in self.data_summary['test'].keys()]
        random.shuffle(train_set)
        random.shuffle(val_set)
        random.shuffle(test_set)
        with open(self.train_list_f, 'w') as f:
            for idx in train_set:
                f.write("{}\n".format(idx))
        with open(self.val_list_f, 'w') as f:
            for idx in val_set:
                f.write("{}\n".format(idx))
        with open(self.test_list_f, 'w') as f:
            for idx in test_set:
                f.write("{}\n".format(idx))
        return len(train_set), len(val_set), len(test_set)

    def single_mel_lf0_extraction(self, wav_f, fid):
        mel_name = os.path.join(self.mel_dir, '{}.npy'.format(fid))
        if os.path.isfile(mel_name):
            return
        else:
            wav_arr = self.audio_processor.load_wav(wav_f)
            wav_arr = self.audio_processor.trim_silence_by_trial(wav_arr, top_db=15., lower_db=25.)
            wav_arr = wav_arr / max(0.01, np.max(np.abs(wav_arr)))
            wav_arr = self.audio_processor.preemphasize(wav_arr)
            mel = self.audio_processor.melspectrogram(wav_arr)
            np.save(mel_name, mel.T)
            return

    def extract_mels(self):
        wav_list = []
        for split in self.data_summary.keys():
            for fid in self.data_summary[split].keys():
                wav_list.append((self.data_summary[split][fid], fid))
        jobs = [joblib.delayed(self.single_mel_lf0_extraction)(wav_f, fid)
                for wav_f, fid in wav_list]
        _ = joblib.Parallel(n_jobs=self.n_jobs, verbose=True)(jobs)
        return
