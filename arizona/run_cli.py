# -*- coding: utf-8 -*-

import click

from arizona.asr.cli import asr

@click.group()
def entry_point():
    pass


entry_point.add_command(asr)


if __name__ == '__main__':
    entry_point()