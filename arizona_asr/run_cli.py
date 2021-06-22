# -*- coding: utf-8 -*-

import os
import sys
import click
import torch
import multiprocessing

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



# Command: pretraining
entry_point.add_command(pretraining)


if __name__ == '__main__':
    entry_point()