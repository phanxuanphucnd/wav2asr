## Table of contents

1. [Introduction](#introduction)
2. [How to use `arizona_asr`](#how_to_use)
    - [Installation](#installation)
    - [Data structure](#data_structure)
    - [Example usage](#usage)


## <a name='introduction'></a> Introduction

A library version wrapped wav2vec 2.0 framework for Automatic Speech Recognition task, which attempts to build a accurate speech recognition models 
with small amount of transcribed data. (eg. 1 hour).

Transfer learning is still the main technique:

- Transfer from self-supervised models (pretrain on unlabeled data)
- Transfer from multingual models (pretrain on multingual data)

<img src="./docs/imgs/wav2vec_asr.png" alt="drawing" width="600"/>


## <a name='how_to_use'></a> How to use `arizona_asr`

### Installation <a name='installation'></a>

- Install dependencies ``fairseq`` and ``wav2letter``:

```js

>>> pip install libs/fairseq-1.0.0a0+9b5b09b-cp36-cp36m-linux_x86_64.whl

>>> sudo apt-get update && sudo apt-get -y install apt-utils libpq-dev libsndfile-dev

>>> sudo apt install libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev

>>> sudo apt-get install libsndfile1-dev libopenblas-dev libfftw3-dev libgflags-dev libgoogle-glog-dev

```

- Install ``kenlm`` and ``wav2letter`` decoder bindings:

```js

>>> git clone https://github.com/kpu/kenlm.git

>>> cd kenlm

>>> mkdir -p build && cd build

>>> cmake ..

>>> make -j 4

>>> cd ../..

>>> git clone -b v0.2 https://github.com/facebookresearch/wav2letter.git

>>> cd wav2letter/bindings/python

>>> export KENLM_ROOT_DIR={path_to_libs_kenlm} && pip install -e .

>>> cd ../../..


Note: path_to_libs_kenlm: is the path to the folder kenlm cloned above.

```


- Install ``arizona_asr`` library:

```js

>>> python setup.py bdist_wheel

>>> pip install dist/arizona_asr-0.0.1-py3-none-any.whl 

```

### Data structure <a name='data_structure'></a>

```

data
├── label_data
│   │   transcript.txt
│   └── audios
│       │   file111.wav
│       │   file112.wav
│       │   ...
│   
├── unlabel_data
│   │   file021.wav
│   │   file022.wav
│   │   ...
│
├── dict.txt
└── text_corpus.txt

```

**- Transcript file:**

- One trainng sample per line with format:

```
audio_absolute_path \tab transcript
```

- Examples of a transcript file:

```

/path/to/1.wav AND IT WAS A MATTER OF COURSE THAT IN THE MIDDLE AGES WHEN THE CRAFTSMEN
/path/to/2.wav AND WAS IN FACT THE KIND OF LETTER USED IN THE MANY SPLENDID MISSALS PSALTERS PRODUCED BY PRINTING IN THE FIFTEENTH CENTURY
/path/to/3.wav JOHN OF SPIRES AND HIS BROTHER VINDELIN FOLLOWED BY NICHOLAS JENSON BEGAN TO PRINT IN THAT CITY
/path/to/4.wav BEING THIN TOUGH AND OPAQUE

```

NOTE:
- One sample per line
- Upper case
- All numbers should be transformed into verbal form
- All special characters (eg. punctuation) should be removed. The final text should contain words only
- Words in a sentence must be separated by whitespace character
- Labeled audio file:
    - Format: wav, PCM 16 bit, single channel, Sampling_rate: 16000.
    - Silence should be removed from the audio.
    - Also, each audio should contain only one person speaking.

### Examples usage <a name='usage'></a>

- Pretraining from unlabel-data:

```js

arizona_asr pretraining --audio_path path/to/audio_directory --init_model wav2vec_small

```

Where:

- audio_path: Path to unlabel audio directory
- init_model: The library provide somes optionals ['wav2vec_small_en', 'wav2vec_small_vi'] or maybe pass the path to to pretrained model

Logs and checkpoints will be stored at outputs directory.\
Log_file path: outputs/date_time/exp_id/hydra_train.log. You should check the loss value to decide when to stop the training process.\
Best_checkpoint path: outputs/date_time/exp_id/checkpoints/checkpoint_best.pt.

