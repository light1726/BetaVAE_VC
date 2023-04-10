import os
import json
import random


class VCTK:
    """
    Generate the json file obtained by dumping the dictionary
    {'train': {fid: wav_path, ...}, 'validation': {fid: wav_path, ...}, 'test': {fid: wav_path, ...}}
    """
    def __init__(self, data_dir, out_dir, val_spks=None, test_spks=None):
        self.data_dir = data_dir
        self.out_dir = out_dir
        self.summary_file = os.path.join(out_dir, 'vctk-summary.json')
        self.wav_ext = 'mic2.flac'
        self.dataset_summary = {}
        assert ((val_spks is None and test_spks is None) or
                (val_spks is not None and test_spks is not None),
                "Please specify both val and test speakers!")
        self.val_spks = val_spks
        self.test_spks = test_spks
        self.validate_dir()

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
        basename = filename.split('/')[-1].split('.')[0]
        spk = basename.split('_')[0]
        uid = basename.split('_')[1]
        fid = '{}_{}'.format(spk, uid)
        return spk, fid

    def write_dataset_info(self):
        with open(self.summary_file, 'w') as f:
            json.dump(self.dataset_summary, f, sort_keys=True, indent=4)
        return

    def write_summary(self):
        dataset_summary = {}
        for root, dirs, files in os.walk(self.data_dir, followlinks=True):
            for basename in files:
                if basename.endswith(self.wav_ext):
                    filename = os.path.join(root, basename)
                    spk, fid = self.extract_spk_fid(filename)
                    wav_path = os.path.join(root, basename)
                    if spk not in dataset_summary.keys():
                        dataset_summary[spk] = {}
                        dataset_summary[spk][fid] = wav_path
                    else:
                        dataset_summary[spk][fid] = wav_path
        if self.val_spks is None and self.test_spks is None:
            all_spks = list(dataset_summary.keys())
            random.shuffle(all_spks)
            self.test_spks = all_spks[-10:]
            self.val_spks = all_spks[-20: -10]
        train_summary = {}
        val_summary = {}
        test_summary = {}
        for spk in dataset_summary.keys():
            if spk in self.val_spks:
                val_summary.update(dataset_summary[spk])
            elif spk in self.test_spks:
                test_summary.update(dataset_summary[spk])
            else:
                train_summary.update(dataset_summary[spk])

        self.dataset_summary = {'train': train_summary,
                                'validation': val_summary,
                                'test': test_summary}
        self.write_dataset_info()
        return
