# -*- coding: utf-8 -*-

import os
import sys
import click
import torch
import soundfile
import multiprocessing

from arizona_asr.utils.gen_dict import gen_dict
from arizona_asr.utils.print_utils import print_name

@click.group()
def entry_point():
    print_name()
    pass

@click.command()
@click.option('--audio_path', required=True,
              type=str, default=None,
              help='Path to the unlabeled audio data.')
@click.option('--init_model', required=True,
              type=str, default=None,
              help="The name of pretrained model or path to the pretrain wav2vec model.")
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

    if init_model != None:
        cmd.append("checkpoint.restore_file=" + os.path.abspath(init_model))
        cmd.append("checkpoint.reset_optimizer=True")
        cmd.append("checkpoint.reset_lr_scheduler=True")
        cmd.append("checkpoint.reset_dataloader=True")
        cmd.append("checkpoint.reset_meters=True")

    #cmd.append("optimization.max_update=2000000")
    cmd.append("dataset.num_workers=" + str(NUM_CPU))
    cmd.append("dataset.max_tokens=" + str(batch_size))
    cmd.append("--config-dir config/pretraining")
    cmd.append("--config-name wav2vec2_base_librispeech")
    cmd = ' '.join(cmd)

    print(f"Execute: {cmd}")
    os.system(cmd)


@click.command()
@click.option('--transcript_file', required=True,
              default=None, type=str,
              help='Path to the description file.')
@click.option('--pretrain_model', required=True,
              default=None, type=str,
              help='The name of pretrained model or path to the pretrained Wav2vec model.')
@click.option('--bach_size', required=False,
              default=2800000, type=int,
              help='Batch size, try decrease this number if any CUDA memory problems occurs.')
@click.option('--pct', required=False,
              default=0.05, type=float,
              help='Percentage of data use for validation.')
@click.option('--restore_file', required=False,
              default=None, type=str,
              help='Resume training from fine-tuned checkpoint.')
def fine_tuning(transcript_file: str, pretrain_model: str, batch_size: int, pct: float, restore_file):
    
    pretrain_model = os.path.abspath(pretrain_model)
    save_dir = os.path.abspath('./.denver/manifest')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Generate dictionary from transcript_file
    gen_dict(transcript_path=transcript_file, save_dir=save_dir)
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
    valid_map = os.path.join(save_dir, )


# Command: pretraining
entry_point.add_command(pretraining)


if __name__ == '__main__':
    entry_point()