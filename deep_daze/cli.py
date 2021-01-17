import fire
from deep_daze import Imagine
from pathlib import Path

def train(text, num_layers = 16, deeper = False):
    if deeper:
        num_layers = 32

    imagine = Imagine(
        text,
        num_layers = num_layers
    )

    if imagine.filename.exists():
        answer = input('Imagined image already exists, do you want to overwrite?').lower()
        if answer not in ('yes', 'y'):
            exit()

    imagine()

def main():
    fire.Fire(train)
