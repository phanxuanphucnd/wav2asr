# -*- coding: utf-8 -*-

import os
import sys
import click
import torch
import soundfile
import multiprocessing

from tqdm import tqdm
from sklearn.utils import shuffle

from arizona.utils.gen_dict import gen_dict
from arizona.utils.print_utils import print_name
from arizona.utils.misc_utils import download_url

INIT_MODEL_MAPPING = {
    'wav2vec-base-en': 'https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt',
    'wav2vec-large-en': 'https://dl.fbaipublicfiles.com/fairseq/wav2vec/libri960_big.pt'
}

@click.group()
def asr():
    print_name()
    pass


@asr.command()
@click.option('--audio_path', required=True,
              type=str, default=None,
              help='Path to the unlabeled audio data.')
@click.option('--init_model', required=True,
              type=str, default='wav2vec-base-en',
              help='The name of pretrained model or path to the pretrain wav2vec model.')
@click.option('--batch_size', required=False,
              type=int, default=1200000,
              help='Batch size, try to decrease this number if any CUDA memory problems occur.')
def pretraining(audio_path: str, init_model: str, batch_size: int):
    
    temp_dir = os.path.abspath('./temp')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    MANIFEST_PATH = './manifest.py'
    
    cmd = 'python3 ' + MANIFEST_PATH + ' ' + audio_path + ' --dest ' + temp_dir + ' --ext wav --valid-percent 0.05'
    os.system(cmd)

    # Pretrain the model
    NUM_GPU = torch.cuda.device_count()
    NUM_CPU = multiprocessing.cpu_count()

    if NUM_GPU == 0:
        print(f"Pytorch cannot find any GPUs !!")
        sys.exit(0)

    cmd = ["fairseq-hydra-train"]
    cmd.append("task.data=" + str(temp_dir))
    cmd.append("distributed_training.distributed_world_size=" + str(NUM_GPU))
    cmd.append("+optimization.update_freq='[" + str(int(64 / NUM_GPU)) + "]'")

    if init_model in INIT_MODEL_MAPPING:
        url = INIT_MODEL_MAPPING.get(init_model.lower())
        file_name = init_model + '.pt'
        dest = './.denver/'
        download_url(url=url, dest=dest,name=file_name)
        init_model_path = os.path.abspath(dest + file_name)
    else:
        init_model_path = os.path.abspath(init_model)

    if init_model != None:
        cmd.append("checkpoint.restore_file=" + init_model_path)
        cmd.append("checkpoint.reset_optimizer=True")
        cmd.append("checkpoint.reset_lr_scheduler=True")
        cmd.append("checkpoint.reset_dataloader=True")
        cmd.append("checkpoint.reset_meters=True")
    else:
        print(f"Warning: `init_model` is None!")

    #cmd.append("optimization.max_update=2000000")
    cmd.append("dataset.num_workers=" + str(NUM_CPU))
    cmd.append("dataset.max_tokens=" + str(batch_size))
    cmd.append("--config-dir arizona/asr/configs/pretraining")
    cmd.append("--config-name wav2vec2-base-en")
    cmd = ' '.join(cmd)
    print(f"\nExecute: {cmd}")

    os.system(cmd)


@asr.command()
@click.option('--transcript_file', required=True,
              default=None, type=str,
              help='Path to the description file.')
@click.option('--pretrain_model', required=True,
              default=None, type=str,
              help='The name of pretrained model or path to the pretrained Wav2vec model.')
@click.option('--batch_size', required=False,
              default=2800000, type=int,
              help='Batch size, try decrease this number if any CUDA memory problems occurs.')
@click.option('--pct', required=False,
              default=0.05, type=float,
              help='Percentage of data use for validation.')
@click.option('--seed', required=False,
              default=123, type=int,
              help="The number of random seed state.")
@click.option('--restore_file', required=False,
              default=None, type=str,
              help='Resume training from fine-tuned checkpoint.')
def finetuning(
    transcript_file: str,
    pretrain_model: str,
    batch_size: int,
    pct: float,
    seed: int,
    restore_file
):
    
    pretrain_model = os.path.abspath(pretrain_model)
    save_dir = os.path.abspath('./.manifest')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Generate dictionary from transcript_file
    gen_dict(transcript_file=transcript_file, save_dir=save_dir)
    # Pretrain the model
    NUM_GPU = torch.cuda.device_count()
    NUM_CPU = multiprocessing.cpu_count()

    if NUM_GPU == 0:
        print(f"Pytorch cannot find any GPUs !!")
        sys.exit(0)

    # Create manifest files
    train_words = os.path.join(save_dir, 'train.wrd')
    valid_words = os.path.join(save_dir, 'valid.wrd')
    train_letters = os.path.join(save_dir, 'train.ltr')
    valid_letters = os.path.join(save_dir, 'valid.ltr')
    train_map = os.path.join(save_dir, 'train.tsv')
    valid_map = os.path.join(save_dir, 'valid.tsv')

    with open(transcript_file) as f:
        data = f.read().splitlines()

    words = [d.split('\t')[1].upper() for d in data]
    letters = [d.replace(' ', '|') for d in words]
    letters = [' '.join(list(d)) + ' |' for d in letters]

    paths = [d.split('\t')[0] for d in data]
    total_duration = 0

    for i in tqdm(range(0, len(paths))):
        audio_info = soundfile.info(paths[i])
        frames = audio_info.frames
        total_duration += audio_info.duration
        paths[i] = paths[i] + '\t' + str(frames)

    SPLIT_NUM = int(len(words)) * (1 - pct)
    words, letters, paths = shuffle(words, letters, paths, random_state=seed)

    train_w, valid_w = words[:SPLIT_NUM], words[SPLIT_NUM:]
    train_l, valid_l = letters[:SPLIT_NUM], letters[SPLIT_NUM:]
    train_p, valid_p = paths[:SPLIT_NUM], paths[SPLIT_NUM:]

    with open(train_words,'w') as f:
        f.write('\n'.join(train_w))
        
    with open(valid_words,'w') as f:
        f.write('\n'.join(valid_w))
    
    with open(train_letters,'w') as f:
        f.write('\n'.join(train_l))
        
    with open(valid_letters,'w') as f:
        f.write('\n'.join(valid_l))
        
    with open(train_map,'w') as f:
        f.write('\n')
        f.write('\n'.join(train_p))
    
    with open(valid_map,'w') as f:
        f.write('\n')
        f.write('\n'.join(valid_p))

    if total_duration <= 5:
        config_name = "base_1h"
    elif total_duration <= 50:
        config_name = "base_10h"
    elif total_duration <= 500:
        config_name = "base_100h"
    else:
        config_name = "base_960h"

    cmd = ["fairseq-hydra-train"]
    cmd.append("task.data=" + str(save_dir))
    cmd.append("distributed_training.distributed_world_size=" + str(NUM_GPU))
    cmd.append("+optimization.update_freq='[" + str(int(24/NUM_GPU)) + "]'")
    cmd.append("model.w2v_path=" + pretrain_model)
    cmd.append("dataset.num_workers=" + str(NUM_CPU))
    cmd.append("dataset.max_tokens=" + str(batch_size))

    if restore_file is not None:
        cmd.append("checkpoint.restore_file=" + restore_file)
        #cmd.append("checkpoint.reset_optimizer=True")
        #cmd.append("checkpoint.reset_lr_scheduler=True")
        #cmd.append("checkpoint.reset_dataloader=True")
        #cmd.append("checkpoint.reset_meters=True")
    
    #cmd.append("optimization.max_update=100000")
    #cmd.append("dataset.validate_after_updates=0")
    #cmd.append("model.freeze_finetune_updates=0")
    cmd.append("--config-dir arizona/asr/configs/finetuning")
    cmd.append("--config-name " + config_name)
    cmd = ' '.join(cmd)
    print(f"\nExecute: {cmd}")

    os.system(cmd)

@asr.command()
@click.option('--transcript_file', required=True,
              default=None, type=str,
              help='Path to the description file.')
def train_lm(transcript_file: str):

    raise NotImplementedError


# Command: pretraining
asr.add_command(pretraining)

# Command: finetuning
asr.add_command(finetuning)

# Command: train_lm
asr.add_command(train_lm)

if __name__ == '__main__':
    asr()