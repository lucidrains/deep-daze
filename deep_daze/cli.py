import sys
import fire
from deep_daze import Imagine
from pathlib import Path

def train(
    text,
    lr = 1e-5,
    num_layers = 16,
    batch_size = 4,
    gradient_accumulate_every = 4,
    epochs = 20,
    iterations = 1050,
    save_every = 100,
    image_width = 512,
    deeper = False,
    overwrite = False,
    save_progress = False,
    seed = None,
    open_folder = True
):
    print('Starting up...')

    if deeper:
        num_layers = 32

    imagine = Imagine(
        text,
        lr = lr,
        num_layers = num_layers,
        batch_size = batch_size,
        gradient_accumulate_every = gradient_accumulate_every,
        epochs = epochs,
        iterations = iterations,
        save_every = save_every,
        save_progress = save_progress,
        seed = seed,
        open_folder = open_folder
    )

    if not overwrite and imagine.filename.exists():
        answer = input('Imagined image already exists, do you want to overwrite? (y/n) ').lower()
        if answer not in ('yes', 'y'):
            sys.exit()

    imagine()

def main():
    fire.Fire(train)
