import fire
from deep_daze import Imagine
from pathlib import Path

def train(
    text,
    lr = 1e-5,
    num_layers = 16,
    gradient_accumulate_every = 4,
    save_every = 100,
    image_width = 512,
    deeper = False
):
    if deeper:
        num_layers = 32

    imagine = Imagine(
        text,
        lr = lr,
        num_layers = num_layers,
        gradient_accumulate_every = gradient_accumulate_every,
        save_every = save_every
    )

    if imagine.filename.exists():
        answer = input('Imagined image already exists, do you want to overwrite? (y/n) ').lower()
        if answer not in ('yes', 'y'):
            exit()

    imagine()

def main():
    fire.Fire(train)
