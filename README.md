
# **Voice Anti-Spoofing Countermeasure on ASVspoof2019 LA**
Implementation of the Light CNN-9 architecture with Max-Feature-Map activation for detecting bona-fide vs. spoofed speech. Supports end-to-end training, evaluation, and inference, with configurable data, model, and logging settings.

<br/>

## Table of Contents
- [**Voice Anti-Spoofing Countermeasure on ASVspoof2019 LA**](#voice-anti-spoofing-countermeasure-on-asvspoof2019-la)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Getting Started](#getting-started)
    - [Requirements](#requirements)
    - [Installation](#installation)
    - [Data Preparation](#data-preparation)
    - [Usage](#usage)
    - [Evaluation \& Grading](#evaluation--grading)
  - [Project Structure \& Model Architecture](#project-structure--model-architecture)
  - [Credits](#credits)
  - [License](#license)



<br/>

## Features

* **Light CNN-9** with competitive MFM activation for robust feature gating.
* **Hydra** for configurable experiments.
* **SpecAugment** and balanced sampling support.
* Automatic **EER**, **ROC**, **DET** computation.
* Optional **Comet ML** integration for experiment tracking.
* Pretrained checkpoint (`best_lightcnn.pt`) included.

<br/>

## Getting Started

### Requirements

* Python 3.8+
* [PyTorch](https://pytorch.org/) 1.10+
* [torchaudio](https://pytorch.org/audio/)
* [Hydra](https://hydra.cc/) 1.1+
* [Comet ML](https://www.comet.com/) (optional)
* NumPy, SciPy, scikit-learn, matplotlib


### Installation

```bash
git clone [https://github.com/tkvlnko/DL\_antispoof.git](https://github.com/tkvlnko/DL_antispoof.git)
cd DL\_antispoof
```
Install requirements
```bash
pip install -r requirements.txt
```
Opionally
```bash
comet.api_key=$COMET_API_KEY
````


### Data Preparation

Download the **ASVspoof2019 LA** dataset (Logical Access partition) from [kaggle](https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset) 
and organize it as:

```
/path/to/ASVspoof2019
├── protocols
│   ├── ASVspoof2019.LA.cm.train.trl.txt
│   ├── ASVspoof2019.LA.cm.dev.trl.txt
│   └── ASVspoof2019.LA.cm.eval.trl.txt
├── ASVspoof2019\_LA\_train
│   └── flac
│       ├── \*.flac
├── ASVspoof2019\_LA\_dev
│   └── flac
│       ├── \*.flac
└── ASVspoof2019\_LA\_eval
````

---

### Usage

Train a new LightCNN model on the train/dev splits:

```bash
python train.py 
````



<br/>

### Evaluation & Grading

Logs and checkpoints will be written to `hydra.run.dir` and into terminal by default [EER on dev snd eval]. Run:

```bash
python predict_eval.py 
```
```bash
python grading.py 
```
to obtian model results and EER




<br/>

## Project Structure & Model Architecture

```
.
├── train.py             # training loops & metrics
├── requirements.txt     
├── best_lightcnn.pt     # pretrained LightCNN checkpoint
└── src/                 
    ├── datasets/        # ASVspoof Dataset & collate
    ├── logger/          # cometML
    ├── model/           # LightCNN & MFM activation
    ├── metrics/         # EER / ROC computation
    └── configs/         # Hydra config files (dataset, model, dataloader)
```


The countermeasure is based on the **Light CNN-9** design from Wen *et al.* (2016), adapted to anti-spoofing:

1. **Input & Preprocessing**

   * **Input shape**: `B×1×F×T`

     * `B` = batch size,
     * `1` = single input channel (log-power spectrogram),
     * `F` ≈ 257 frequency bins (`n_fft/2+1`),
     * `T` = number of time frames (variable).
   * **Front-end**: STFT → power spectrum → `log(1 + x)` → (optional) SpecAugment.

2. **Max-Feature-Map (MFM) Activation**

   * Each MFM layer contains a convolution (or linear) with **2×** the desired output channels, then splits its output into two halves and takes an element-wise maximum.
   * This competitive gating prunes weaker feature responses and yields exactly `C` channels from an initial `2C`, improving sparsity and generalization.

3. **Convolutional Backbone**
   We stack **five convolutional blocks**, alternating 1×1 and 3×3 kernels, interleaved with MFM and pooling:

   | Block | Layers                                                                                             | Output Shape        |
   | :---: | :------------------------------------------------------------------------------------------------- | :------------------ |
   | **1** | `Conv(1→96 @5×5, pad=2)` → MFM(96→48) → `MaxPool2d(2×2)`                                           | `48×(F/2)×(T/2)`    |
   | **2** | `1×1 Conv(48→96)` → MFM(96→48) → `Conv(48→192 @3×3, pad=1)` → MFM(192→96) → `MaxPool2d(2×2)`       | `96×(F/4)×(T/4)`    |
   | **3** | `1×1 Conv(96→192)` → MFM(192→96) → `Conv(96→384 @3×3, pad=1)` → MFM(384→192) → `MaxPool2d(2×2)`    | `192×(F/8)×(T/8)`   |
   | **4** | `1×1 Conv(192→384)` → MFM(384→192) → `Conv(192→256 @3×3, pad=1)` → MFM(256→128)                    | `128×(F/8)×(T/8)`   |
   | **5** | `1×1 Conv(128→256)` → MFM(256→128) → `Conv(128→256 @3×3, pad=1)` → MFM(256→128) → `MaxPool2d(2×2)` | `128×(F/16)×(T/16)` |



4. **Global Feature Aggregation**

   * **AdaptiveAvgPool2d((1,1))** collapses the `(F/16)×(T/16)` feature map to a single vector of length 128 per example.
   * This design makes the network **time-invariant**: it can handle any input length T.

5. **Fully-Connected Classifier**

   * **Flatten** to `[B×128]`.
   * **MFM FC**: Linear(128→512) → MFM → 256 features.
   * **Dropout** (0.2 by default) for regularization.
   * **Final Linear**: 256 → 2 logits (bona-fide vs. spoof).

6. **Loss & Inference**

   *  **CrossEntropyLoss** is used on the two logits during training.
   * At inference, we apply a softmax and take the “bona-fide” class probability as the confidence score.
   * **EER** is computed by sweeping thresholds over these scores to find the operating point where **FAR = FRR**.



<br/>

## Credits

This repository is based on a modified fork of [pytorch-template](https://github.com/Blinorot/pytorch_project_template) .

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
