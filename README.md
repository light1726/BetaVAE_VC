### BetaVAE_VC
This repo contains code for paper "Disentangled Speech Representation Learning for One-Shot Cross-Lingual Voice Conversion Using ß-VAE" in SLT 2022.
####  [Samples](https://beta-vaevc.github.io/) | [Paper](https://www1.se.cuhk.edu.hk/~hccl/publications/pub/2023%20SLT2022-Beta_VAE_based_one_shot_cross_lingual_VC.pdf) | [Pretrained Models](https://drive.google.com/drive/folders/1FerYnoB60B3aQgt-lAO9g8D3u0RoJe9T?usp=sharing)


#### 0. Setup Conda Environment
```bash
conda env create -f environment.yml
conda activate betavae-vc-env
```

#### 1. Data preprocessing
* Download corpus
1. English: [VCTK](https://datashare.ed.ac.uk/handle/10283/3443)
2. Mandarin: [AISHELL3](https://www.openslr.org/93/)
* Modify the paths specified in ```configs/haparams.py```: ```corpus_dir``` for both ```VCTK``` and ```AiShell3```, ```dataset_dir``` for extracted features and TFRecord files.
* Prepare the dataset for training:
```bash
python preprocess.py
```

#### 2. Training
```bash
CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=true python train.py --out_dir ./outputs --data_dir /path/to/save/features/tfrecords
```

#### 3. Inference
```bash
# inference from mels
# test-mels.txt contains list of paths for mel-spectrograms with *.npy format, one path per line
CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=true python inference-from-mel.py --ckpt_path ./outputs/models/ckpt-500 --test_dir outputs/tests --src_mels test-mels.txt --ref_mels test-mels.txt

# inference from wavs
# test-wavs.txt contains list of paths for speech with *.wav format, one path per line
CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=true python inference-from-wav.py --ckpt_path ./outputs/models/ckpt-500 --test_dir outputs/tests --src_wavs test-wavs.txt --ref_wavs test-wavs.txt
```

#### 4. Latent extraction
```bash
CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=true python feature_extraction.py --data_dir /path/to/save/features/tfrecords --save_dir ./outputs/features --ckpt_path ./outputs/models/ckpt-300
```

#### 5. EER computation based on the extracted latents
```bash
# compute EER using content embeddings
python tests/compute_eer.py --data_dir ./outputs/features/EN --mode content
# compute EER using speaker embeddings
python tests/compute_eer.py --data_dir ./outputs/features/EN --mode spk
```

### Cite this work
```text
@inproceedings{slt2022_hui_disentanle,
  author    = {Hui Lu and
               Disong Wang and
               Xixin Wu and
               Zhiyong Wu and
               Xunying Liu and
               Helen Meng},
  title     = {Disentangled Speech Representation Learning for One-Shot Cross-Lingual
               Voice Conversion Using Beta-VAE},
  booktitle = {{IEEE} Spoken Language Technology Workshop, {SLT} 2022, Doha, Qatar,
               January 9-12, 2023},
  pages     = {814--821},
  publisher = {{IEEE}},
  year      = {2022},
  doi       = {10.1109/SLT54892.2023.10022787},
}
```