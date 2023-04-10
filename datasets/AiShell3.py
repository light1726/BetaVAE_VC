import os
import json
import random


class AiShell3:
    """
    Generate the json file obtained by dumping the dictionary
    {'train': {fid: wav_path, ...}, 'validation': {fid: wav_path, ...}, 'test': {fid: wav_path, ...}}
    """
    def __init__(self, data_dir, out_dir, train_spk_file=None, val_test_spk_file=None):
        self.data_dir = data_dir
        self.out_dir = out_dir
        self.train_spks = self.read_train_spks(train_spk_file)
        val_test_spks = self.read_val_test_spks(val_test_spk_file)
        random.shuffle(val_test_spks)
        self.val_spks = val_test_spks[:len(val_test_spks) // 2]
        self.test_spks = val_test_spks[len(val_test_spks) // 2:]
        self.summary_file = os.path.join(out_dir, 'aishell3-summary.json')
        self.wav_ext = '.wav'
        self.dataset_summary = {}
        self.validate_dir()

    def read_train_spks(self, f):
        train_spks = []
        if f is not None:
            with open(f, 'r', encoding='utf-8') as f:
                for line in f:
                    train_spks.append(line.strip())
        else:
            train_spks = os.listdir(os.path.join(self.data_dir, 'train/wav'))
        assert len(train_spks) > 0
        return train_spks

    def read_val_test_spks(self, f):
        vt_spks = []
        if f is not None:
            with open(f, 'r', encoding='utf-8') as f:
                for line in f:
                    vt_spks.append(line.strip())
        else:
            vt_spks = [spk for spk in os.listdir(os.path.join(self.data_dir, 'test/wav'))
                       if spk not in self.train_spks]
        assert len(vt_spks) > 0
        return vt_spks

    def validate_dir(self):
        if not os.path.isdir(self.data_dir):
            raise NotADirectoryError('{} is not a valid directory!'.format(self.data_dir))
        if not os.path.isdir(self.out_dir):
            os.makedirs(self.out_dir)
        return

    @staticmethod
    def extract_spk_fid(filename):
        """
        :param filename:
        :return: spk_name, fid
        """
        fid = filename.split('/')[-1].split('.')[0]
        spk = filename.split('/')[-2]
        return spk, fid

    def write_dataset_info(self):
        with open(self.summary_file, 'w') as f:
            json.dump(self.dataset_summary, f, sort_keys=True, indent=4)
        return

    def write_summary(self):
        train_summary = {}
        val_summary = {}
        test_summary = {}
        for root, dirs, files in os.walk(self.data_dir, followlinks=True):
            for basename in files:
                if basename.endswith(self.wav_ext):
                    wav_path = os.path.join(root, basename)
                    spk, fid = self.extract_spk_fid(wav_path)
                    if spk in self.train_spks:
                        train_summary[fid] = wav_path
                    elif spk in self.val_spks:
                        val_summary[fid] = wav_path
                    elif spk in self.test_spks:
                        test_summary[fid] = wav_path
                    else:
                        continue
        self.dataset_summary = {'train': train_summary,
                                'validation': val_summary,
                                'test': test_summary}
        self.write_dataset_info()
        return
