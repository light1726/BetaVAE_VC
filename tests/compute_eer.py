import os
import json
import random
import argparse
import scipy.stats
import numpy as np
from tqdm import tqdm
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d


def create_trials(data_dir, mode='spk', anchor_f=None):
    """
    :param data_dir: under which there are spk_uid.npy
    :param mode: suffix of the feature numpy filename
    :return: trials: [(enrolled_vector, test_vector), ...]
             labels: [0, 1, ...]
             spk2vecs: {spk: {fid: [arr, ...]}}
    """
    assert mode in ['spk', 'content']

    def get_spk_fid(basename):
        if basename.startswith('SSB'):
            spk = basename[:7]
            fid = basename.split('-')[0]
        else:
            spk = basename.split('_')[0]
            fid = basename.split('-')[0]
        return spk, fid

    spk2vecs = {}
    for f in os.listdir(data_dir):
        if f.endswith('{}.npy'.format(mode)):
            spk, fid = get_spk_fid(f)
            path = os.path.join(data_dir, f)
            feat = np.load(path)
            if len(feat.shape) == 2:
                feat = np.mean(feat, axis=0)
            if spk in spk2vecs.keys():
                spk2vecs[spk][fid] = feat
            else:
                spk2vecs[spk] = {fid: feat}
    trials = []
    labels = []
    if anchor_f is not None:
        anchor_dict = get_anchors(anchor_f)
    else:
        anchor_dict = {}
        for spk in spk2vecs.keys():
            if len(spk2vecs[spk].keys()) <= 100:
                continue
            anchors = random.sample(list(spk2vecs[spk].keys()), 4)
            anchor_dict[spk] = anchors
        lang = 'CN' if spk.startswith('SSB') else 'EN'
        with open('SV_anchors-{}.json'.format(lang), 'w', encoding='utf-8') as f:
            json.dump(anchor_dict, f)
    for spk in spk2vecs.keys():
        if len(spk2vecs[spk].keys()) <= 100:
            continue
        anchors = anchor_dict[spk]
        aps = list(spk2vecs[spk].keys())
        for anchor in anchors:
            if anchor not in aps:
                print(anchor)
            else:
                aps.remove(anchor)
        ans = []
        for s in spk2vecs.keys():
            if s != spk:
                ans += list(spk2vecs[s].keys())
        anchor_key = '{}_anchor'.format(spk)
        spk2vecs[spk][anchor_key] = np.mean(
            np.stack([spk2vecs[spk][k] for k in anchors], axis=0), axis=0)
        for ap in aps:
            trials.append((anchor_key, ap))
            labels.append(1)
        for an in ans:
            trials.append((anchor_key, an))
            labels.append(0)
    return trials, labels, spk2vecs


def get_anchors(anchor_f):
    with open(anchor_f, 'r') as f:
        anchor_dict = json.load(f)  # {spk: [fid1, fid2, ...]}
    return anchor_dict


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, h


def cosine_score(trials, vecs_dict):

    def get_spk(inp_str):
        if inp_str.startswith('SSB'):
            spk = inp_str[:7]
        else:
            spk = inp_str.split('_')[0]
        return spk

    scores = []
    for item in tqdm(trials):
        spk0 = get_spk(item[0])
        enroll_vector = vecs_dict[spk0][item[0]]
        spk1 = get_spk(item[1])
        test_vector = vecs_dict[spk1][item[1]]
        score = enroll_vector.dot(test_vector.T)
        denom = np.linalg.norm(enroll_vector) * np.linalg.norm(test_vector)
        score = score / denom
        scores.append(score)
    return scores


def compute_eer(labels, scores):
    """sklearn style compute eer
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    threshold = interp1d(fpr, thresholds)(eer)
    return eer, threshold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir', type=str,
        help='directory of all test speaker embedding vectors', required=True)
    parser.add_argument(
        '--mode', type=str, choices=['spk', 'content'], required=True)
    parser.add_argument('--n_runs', type=int, default=1)
    parser.add_argument('--anchor', type=str, help='Anchor file')
    args = parser.parse_args()
    assert os.path.isdir(args.data_dir)
    eers = []
    labels = []
    for _ in range(args.n_runs):
        trials, labels, spk2vecs = create_trials(args.data_dir, args.mode, anchor_f=args.anchor)
        scores = cosine_score(trials, spk2vecs)
        eer, _ = compute_eer(labels, scores)
        eers.append(eer)
    eer, h = mean_confidence_interval(eers)
    n_trials = len(labels)
    n_pos = sum(labels)
    n_neg = n_trials - n_pos
    print('There are {} trials for {} representation, {} of them are positive trials '
          'and {} are negative ones.'.format(n_trials, args.mode, n_pos, n_neg))
    print('The EER is {:.4f} ± {:.4f}'.format(eer, h))
    return


if __name__ == '__main__':
    main()
