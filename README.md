# The SpeechBrain Toolkit [![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/SpeechBrain1/)

<p align="center">
  <img src="http://www.darnault-parcollet.fr/Parcollet/hiddennoshare/speechbrain.github.io/img/logo_noname_rounded_big.png" alt="drawing" width="250"/>
</p>

SpeechBrain is an **open-source** and **all-in-one** conversational AI toolkit based on PyTorch.

The goal is to create a **single**, **flexible**, and **user-friendly** toolkit that can be used to easily develop **state-of-the-art speech technologies**, including systems for **speech recognition**, **speaker recognition**, **speech enhancement**, **speech separation**, **language identification**, **multi-microphone signal processing**, and many others.

*SpeechBrain is currently in beta*.

**News:** the call for new sponsors (2022) is open. [Take a look here if you are interested!](https://drive.google.com/file/d/1Njn_T2qLJCLPmF2LJ_X7yxxobqK3-CPW/view?usp=sharing)

**News: we are looking for a full-time research engineer.** [Take a look here if you are interested!](https://drive.google.com/file/d/1m9aIvSg3KTrOVVY9fvJ_DGdIsskzkqXf/view)


| **[Discourse](https://speechbrain.discourse.group)** | **[Tutorials](https://speechbrain.github.io/tutorial_basics.html)** | **[Website](https://speechbrain.github.io/)** | **[Documentation](https://speechbrain.readthedocs.io/en/latest/index.html)** | **[Contributing](https://speechbrain.readthedocs.io/en/latest/contributing.html)** | **[HuggingFace](https://huggingface.co/speechbrain)** |

# Key features

SpeechBrain provides various useful tools to speed up and facilitate research on speech and language technologies:
- Various pretrained models nicely integrated with <img src="https://huggingface.co/front/assets/huggingface_logo.svg" alt="drawing" width="40"/> <sub>(HuggingFace)</sub> in our official [organization account](https://huggingface.co/speechbrain). These models are coupled with easy-inference interfaces that facilitate their use.  To help everyone replicate our results, we also provide all the experimental results and folders (including logs, training curves, etc.) in a shared Google Drive folder.
- The `Brain` class, a fully-customizable tool for managing training and evaluation loops over data. The annoying details of training loops are handled for you while retaining complete flexibility to override any part of the process when needed.
- A YAML-based hyperparameter file that specifies all the hyperparameters, from individual numbers (e.g., learning rate) to complete objects (e.g., custom models). This elegant solution dramatically simplifies the training script.
- Multi-GPU training and inference with PyTorch Data-Parallel or Distributed Data-Parallel.
- Mixed-precision for faster training.
- A transparent and entirely customizable data input and output pipeline. SpeechBrain follows the PyTorch data loading style and enables users to customize the I/O pipelines (e.g., adding on-the-fly downsampling, BPE tokenization, sorting, threshold ...).
- On-the-fly dynamic batching
- Efficient reading of large datasets from a shared  Network File System (NFS) via [WebDataset](https://github.com/webdataset/webdataset).
- Interface with [HuggingFace](https://huggingface.co/speechbrain) for popular models such as wav2vec2  and Hubert.
- Interface with [Orion](https://github.com/Epistimio/orion) for hyperparameter tuning.


### Speech recognition

SpeechBrain supports state-of-the-art methods for end-to-end speech recognition:
- Support of wav2vec 2.0 pretrained model with finetuning.
- State-of-the-art performance or comparable with other existing toolkits in several ASR benchmarks.
- Easily customizable neural language models, including RNNLM and TransformerLM. We also share several pre-trained models that you can easily use (more to come!). We support the Hugging Face `dataset` to facilitate the training over a large text dataset.
- Hybrid CTC/Attention end-to-end ASR:
    - Many available encoders: CRDNN (VGG + {LSTM,GRU,LiGRU} + DNN), ResNet, SincNet, vanilla transformers, contextnet-based transformers or conformers. Thanks to the flexibility of SpeechBrain, any fully customized encoder could be connected to the CTC/attention decoder and trained in a few hours of work. The decoder is fully customizable: LSTM, GRU, LiGRU, transformer, or your neural network!
    - Optimised and fast beam search on both CPUs and GPUs.
- Transducer end-to-end ASR with both a custom Numba loss and the torchaudio one. Any encoder or decoder can be plugged into the transducer ranging from VGG+RNN+DNN to conformers.
- Pre-trained ASR models for transcribing an audio file or extracting features for a downstream task.

### Feature extraction and augmentation

SpeechBrain provides efficient (GPU-friendly) speech augmentation and feature extraction pipelines:
- On-the-fly and fully-differentiable acoustic feature extraction: filter banks can be learned. This strategy simplifies the training pipeline (you don't have to dump features on disk).
- On-the-fly feature normalization (global, sentence, batch, or speaker level).
- On-the-fly environmental corruptions based on noise, reverberation, and babble for robust model training.
- On-the-fly frequency and time domain SpecAugment with speed augmentation.

### Speaker recognition, identification and diarization
SpeechBrain provides different models for speaker recognition, identification, and diarization on different datasets:
- State-of-the-art performance on speaker recognition and diarization based on ECAPA-TDNN models.
- Original Xvectors implementation (inspired by Kaldi) with PLDA.
- Spectral clustering for speaker diarization (combined with speakers embeddings).
- Libraries to extract speaker embeddings with a pre-trained model on your data.

###  Language Identification
SpeechBrain provides different models for language identification.
In particular, our best model is based on an ECAPA-TDNN trained with the [voxlingua107 dataset](http://bark.phon.ioc.ee/voxlingua107/).


### Speech Translation
- Recipes for transformer and conformer-based end-to-end speech translation.
- Possibility to choose between normal training (Attention), multi-objectives (CTC+Attention), and multitasks (ST + ASR).

### Speech enhancement and separation
- Recipes for spectral masking, spectral mapping, and time-domain speech enhancement.
- Multiple sophisticated enhancement losses, including differentiable STOI loss, MetricGAN, and mimic loss.
- State-of-the-art performance on speech separation with Conv-TasNet, DualPath RNN, and SepFormer.

### Multi-microphone processing
Combining multiple microphones is a powerful approach to achieve robustness in adverse acoustic environments:
- Delay-and-sum, MVDR, and GeV beamforming.
- Speaker localization.

### Performance
The recipes released with speechbrain implement speech processing systems with competitive or state-of-the-art performance. In the following, we report the best performance achieved on some popular benchmarks:

| Dataset        | Task           | System  | Performance  |
| ------------- |:-------------:| -----:|-----:|
| LibriSpeech      | Speech Recognition | wav2vec2 | WER=1.90% (test-clean) |
| LibriSpeech      | Speech Recognition | CNN + Transformer | WER=2.46% (test-clean) |
| TIMIT      | Speech Recognition | CRDNN + distillation | PER=13.1% (test) |
| TIMIT      | Speech Recognition | wav2vec2 + CTC/Att. | PER=8.04% (test) |
| CommonVoice (English) | Speech Recognition | wav2vec2 + CTC | WER=15.69% (test) |
| CommonVoice (French) | Speech Recognition | wav2vec2 + CTC | WER=9.96% (test) |
| CommonVoice (Italian) | Speech Recognition | wav2vec2 + seq2seq | WER=9.86% (test) |
| CommonVoice (Kinyarwanda) | Speech Recognition | wav2vec2 + seq2seq | WER=18.91% (test) |
| AISHELL (Mandarin) | Speech Recognition | wav2vec2 + seq2seq | CER=5.58% (test) |
| Fisher-callhome (spanish) | Speech translation | conformer (ST + ASR) | BLEU=48.04 (test) |
| VoxCeleb2      | Speaker Verification | ECAPA-TDNN | EER=0.69% (vox1-test) |
| AMI      | Speaker Diarization | ECAPA-TDNN | DER=3.01% (eval)|
| VoiceBank      | Speech Enhancement | MetricGAN+| PESQ=3.08 (test)|
| WSJ2MIX      | Speech Separation | SepFormer| SDRi=22.6 dB (test)|
| WSJ3MIX      | Speech Separation | SepFormer| SDRi=20.0 dB (test)|
| WHAM!     | Speech Separation | SepFormer| SDRi= 16.4 dB (test)|
| WHAMR!     | Speech Separation | SepFormer| SDRi= 14.0 dB (test)|
| Libri2Mix     | Speech Separation | SepFormer| SDRi= 20.6 dB (test-clean)|
| Libri3Mix     | Speech Separation | SepFormer| SDRi= 18.7 dB (test-clean)|
| LibryParty | Voice Activity Detection | CRDNN | F-score=0.9477 (test) |
| IEMOCAP | Emotion Recognition | wav2vec | Accuracy=79.8% (test) |
| CommonLanguage | Language Recognition | ECAPA-TDNN | Accuracy=84.9% (test) |
| Timers and Such | Spoken Language Understanding | CRDNN | Intent Accuracy=89.2% (test) |
| SLURP | Spoken Language Understanding | CRDNN | Intent Accuracy=87.54% (test) |
| VoxLingua 107 | Identification | ECAPA-TDNN | Sentence Accuracy=93.3% (test) |

For more details, take a look into the corresponding implementation in recipes/dataset/.

### Pretrained Models

Beyond providing recipes for training the models from scratch, SpeechBrain shares several pre-trained models (coupled with easy-inference functions) on [HuggingFace](https://huggingface.co/speechbrain). In the following, we report some of them:

| Task        | Dataset | Model |
| ------------- |:-------------:| -----:|
| Speech Recognition | LibriSpeech | [CNN + Transformer](https://huggingface.co/speechbrain/asr-transformer-transformerlm-librispeech) |
| Speech Recognition | LibriSpeech | [CRDNN](https://huggingface.co/speechbrain/asr-crdnn-transformerlm-librispeech) |
| Speech Recognition | CommonVoice(English) | [wav2vec + CTC](https://huggingface.co/speechbrain/asr-wav2vec2-commonvoice-en) |
| Speech Recognition | CommonVoice(French) | [wav2vec + CTC](https://huggingface.co/speechbrain/asr-crdnn-commonvoice-fr) |
| Speech Recognition | CommonVoice(Italian) | [wav2vec + CTC](https://huggingface.co/speechbrain/asr-wav2vec2-commonvoice-it) |
| Speech Recognition | CommonVoice(Kinyarwanda) | [wav2vec + CTC](https://huggingface.co/speechbrain/asr-wav2vec2-commonvoice-rw) |
| Speech Recognition | AISHELL(Mandarin) | [wav2vec + CTC](https://huggingface.co/speechbrain/asr-wav2vec2-transformer-aishell) |
| Speaker Recognition | Voxceleb | [ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) |
| Speech Separation | WHAMR! | [SepFormer](https://huggingface.co/speechbrain/sepformer-whamr) |
| Speech Enhancement | Voicebank | [MetricGAN+](https://huggingface.co/speechbrain/metricgan-plus-voicebank) |
| Spoken Language Understanding | Timers and Such | [CRDNN](https://huggingface.co/speechbrain/slu-timers-and-such-direct-librispeech-asr) |
| Language Identification | CommonLanguage | [ECAPA-TDNN](https://huggingface.co/speechbrain/lang-id-commonlanguage_ecapa) |


### Documentation & Tutorials
SpeechBrain is designed to speed up the research and development of speech technologies. Hence, our code is backed-up with different levels of documentation:
- **Educational-level:** we provide various Google Colab (i.e., interactive) tutorials describing all the building blocks of SpeechBrain ranging from the core of the toolkit to a specific model designed for a particular task. The tutorials are designed not only to help people familiarize themselves with SpeechBrain but, more in general, to help them familiarize themselves with speech and language technologies.
- **Functional-level:** all classes in SpeechBrain contains a detailed docstring. It describes the input and output formats, the different arguments, the usage of the function, the potentially associated bibliography, and a function example used for test integration during pull requests.
- **Low-level:** The code also uses a lot of in-line comments to describe nontrivial parts of the code.

### Under development
We are currently implementing speech synthesis pipelines and real-time speech processing pipelines. An interface with the Finite State Transducers (FST) implemented by the [Kaldi 2 team](https://github.com/k2-fsa/k2) is under development.

# Quick installation
SpeechBrain is constantly evolving. New features, tutorials, and documentation will appear over time.
SpeechBrain can be installed via PyPI. Moreover,  a local installation can be used by those users that what to run experiments and modify/customize the toolkit. SpeechBrain supports both CPU and GPU computations. For most all the recipes, however, a GPU is necessary during training. Please note that CUDA must be properly installed to use GPUs.


## Install via PyPI

Once you have created your Python environment (Python 3.8+) you can simply type:

```
pip install speechbrain
```

Then you can access SpeechBrain with:

```
import speechbrain as sb
```

## Install with GitHub

Once you have created your Python environment (Python 3.8+) you can simply type:

```
git clone https://github.com/speechbrain/speechbrain.git
cd speechbrain
pip install -r requirements.txt
pip install --editable .
```

Then you can access SpeechBrain with:

```
import speechbrain as sb
```

Any modification made to the `speechbrain` package will be automatically interpreted as we installed it with the `--editable` flag.

## Test Installation
Please, run the following script to make sure your installation is working:
```
pytest tests
pytest --doctest-modules speechbrain
```

# Running an experiment
In SpeechBrain, you can run experiments in this way:

```
> cd recipes/<dataset>/<task>/
> python experiment.py params.yaml
```

The results will be saved in the `output_folder` specified in the yaml file. The folder is created by calling `sb.core.create_experiment_directory()` in `experiment.py`. Both detailed logs and experiment outputs are saved there. Furthermore, less verbose logs are output to stdout.

# SpeechBrain Roadmap

As a community-based and open source project, SpeechBrain needs the help of its community to grow in the right direction. Opening the roadmap to our users enables the toolkit to benefit from new ideas, new research axes, or even new technologies. The roadmap, available on our [Discourse](https://speechbrain.discourse.group/t/speechbrain-a-community-roadmap/179) lists all the changes and updates that need to be done in the current version of SpeechBrain. Users are more than welcome to propose new items via new Discourse topics!

# Learning SpeechBrain

We provide users with different resources to learn how to use SpeechBrain:
- General information can be found on the [website](https://speechbrain.github.io).
- We offer many tutorials, you can start from the [basic ones](https://speechbrain.github.io/tutorial_basics.html) about SpeechBrain basic functionalities and building blocks. We provide also more advanced tutorials (e.g SpeechBrain advanced, signal processing ...). You can browse them via the Tutorials drop-down menu on [SpeechBrain website](https://speechbrain.github.io) in the upper right.
- Details on the SpeechBrain API, how to contribute, and the code are given in the [documentation](https://speechbrain.readthedocs.io/en/latest/index.html).

# License
SpeechBrain is released under the Apache License, version 2.0. The Apache license is a popular BSD-like license. SpeechBrain can be redistributed for free, even for commercial purposes, although you can not take off the license headers (and under some circumstances, you may have to distribute a license document). Apache is not a viral license like the GPL, which forces you to release your modifications to the source code. Note that this project has no connection to the Apache Foundation, other than that we use the same license terms.

# Social Media
We constantly update the community using Twitter. [Feel free to follow us](https://twitter.com/speechbrain1)

# Citing SpeechBrain
Please, cite SpeechBrain if you use it for your research or business.

```bibtex
@misc{speechbrain,
  title={{SpeechBrain}: A General-Purpose Speech Toolkit},
  author={Mirco Ravanelli and Titouan Parcollet and Peter Plantinga and Aku Rouhe and Samuele Cornell and Loren Lugosch and Cem Subakan and Nauman Dawalatabad and Abdelwahab Heba and Jianyuan Zhong and Ju-Chieh Chou and Sung-Lin Yeh and Szu-Wei Fu and Chien-Feng Liao and Elena Rastorgueva and François Grondin and William Aris and Hwidong Na and Yan Gao and Renato De Mori and Yoshua Bengio},
  year={2021},
  eprint={2106.04624},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  note={arXiv:2106.04624}
}
```

