# -*- coding: utf-8 -*-

import os
import ast
import sys
import math
import uuid
import wave
import torch
import random
import struct
import soundfile
import numpy as np
import editdistance

from shutil import copy2
from typing import Union, List
from fairseq.data.data_utils import post_process
from fairseq.logging.meters import StopwatchMeter, TimeMeter
from fairseq import checkpoint_utils, options, progress_bar, tasks, utils


def add_asr_eval_argument(parser, lm_type, lm_model, lm_weight, word_score, lexicon, beam_size):
    
    parser.add_argument("--kspmodel", default=None, help="sentence piece model")
    parser.add_argument(
        "--wfstlm", default=None, help="wfstlm on dictonary output units"
    )
    parser.add_argument(
        "--rnnt_decoding_type",
        default="greedy",
        help="wfstlm on dictonary output units",
    )
    try:
        parser.add_argument(
            "--lm-weight",
            "--lm_weight",
            type=float,
            default=lm_weight,
            help="weight for lm while interpolating with neural score",
        )
    except:
        pass
    parser.add_argument(
        "--rnnt_len_penalty", default=-0.5, help="rnnt length penalty on word level"
    )
    
    #     parser.add_argument(
    #         "--w2l-decoder",
    #         choices=["viterbi", "kenlm", "fairseqlm"],
    #         help="use a w2l decoder",
    #     )

    parser.add_argument("--w2l-decoder",default=lm_type,
                        help="use a w2l decoder",)
    parser.add_argument("--lexicon", help="lexicon for w2l decoder", default=lexicon)
    parser.add_argument("--unit-lm", action="store_true", help="if using a unit lm")
    parser.add_argument("--kenlm-model", "--lm-model", help="lm model for w2l decoder", default=lm_model)
    parser.add_argument("--beam-threshold", type=float, default=beam_size)
    parser.add_argument("--beam-size-token", type=float, default=100)
    parser.add_argument("--word-score", type=float, default=word_score)
    parser.add_argument("--unk-weight", type=float, default=-math.inf)
    parser.add_argument("--sil-weight", type=float, default=0.0)
    parser.add_argument(
        "--dump-emissions",
        type=str,
        default=None,
        help="if present, dumps emissions into this file and exits",
    )
    parser.add_argument(
        "--dump-features",
        type=str,
        default=None,
        help="if present, dumps features into this file and exits",
    )
    parser.add_argument(
        "--load-emissions",
        type=str,
        default=None,
        help="if present, loads emissions from this file",
    )

    return parser

def generate_random_wav(wav_path, sr=16000):
    noise_output = wave.open(wav_path, 'w')
    noise_output.setparams((1, 2, sr, 0, 'NONE', 'not compressed'))

    for i in range(0, sr*3):
        value = random.randint(-32767, 32767)
        packed_value = struct.pack('h', value)
        noise_output.writeframes(packed_value)

    noise_output.close()

sys.argv.append('/mnt/disks2/data')

class Wav2AsrLearner:
    def __init__(
        self,
        pretrain_model: str=None,
        finetune_model: str=None,
        dictionary: str=None,
        lm_type: str=None,
        lm_lexicon: str=None,
        lm_model: str=None,
        lm_weight: float=1.51,
        word_score: float=2.57,
        beam_size: int=100,
        temp_path: str='temp'
    ) -> None:
        """ Initialize a Learner class

        :param pretrain_model: Path to the pretrained Wav2vec model
        :param finetune_model: Path to the fine-tuned model
        :param dictionary: Path to the dictionary
        :param lm_type: Language model type
        :param lm_lexicon: Path to the lexicon
        :param lm_model: Path to the language model
        :param lm_weight: How much language model affect the result, the higher the more important
        :param word_score: Weight score for group of letter forming a word
        :param beam_size: Number of path for decoding, the higher the better but slower
        :param temp_path: Directory for storing temporary files during processing
        """

        parser = options.get_generation_parser()
        parser = add_asr_eval_argument(
            parser=parser,
            lm_type=lm_type,
            lm_model=lm_model,
            lm_weight=lm_weight,
            word_score=word_score,
            lexicon=lm_lexicon,
            beam_size=beam_size
        )
        args = options.parse_args_and_arch(parser)
        args.task = 'audio_pretraining'
        args.path = finetune_model
        args.nbest = 1
        args.criterion = 'ctc'
        args.labels = 'ltr'
        args.post_process = 'letter'
        args.max_tokens = 4000000
        args.w2vec_dict = dictionary
        self.args = args
        self.models = None
        self.saved_cfg = None
        self.generator = None
        self.state = None
        self.temp_path = os.path.abspath(temp_path)
        self.pretrain_model = os.path.abspath(pretrain_model)
        self.beam_size = beam_size
        
        if not os.path.exists(self.temp_path):
            os.makedirs(self.temp_path)

        # Transcribe a test sample
        sample_audio_path = os.path.abspath(self.temp_path, 'noise.wav')
        generate_random_wav(sample_audio_path, 16000)
        self.transcribe([sample_audio_path])
        os.remove(sample_audio_path)
        print("Loading completed !")

    def transcribe(self, input: Union[List, str]):
        if isinstance(input, str):
            input = [input]

        return None
