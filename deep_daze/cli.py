import sys

import fire

from deep_daze import Imagine


def train(
        text,
        learning_rate=1e-5,
        num_layers=16,
        batch_size=4,
        gradient_accumulate_every=4,
        epochs=20,
        iterations=1050,
        save_every=100,
        image_width=512,
        deeper=False,
        overwrite=False,
        save_progress=False,
        seed=None,
        open_folder=True,
        save_date_time=False,
        start_image_path=None,
        start_image_train_iters=50,
        theta_initial=None,
        theta_hidden=None,
        start_image_lr=3e-4

):
    """
    :param text: (required) A phrase less than 77 characters which you would like to visualize.
    :param learning_rate: The learning rate of the neural net.
    :param num_layers: The number of hidden layers to use in the Siren neural net.
    :param batch_size: The number of generated images to pass into Siren before calculating loss. Decreasing this can lower memory and accuracy.
    :param gradient_accumulate_every: Calculate a weighted loss of n samples for each iteration. Increasing this can help increase accuracy with lower batch sizes.
    :param epochs: The number of epochs to run.
    :param iterations: The number of times to calculate and backpropagate loss in a given epoch.
    :param save_progress: Whether or not to save images generated before training Siren is complete.
    :param save_every: Generate an image every time iterations is a multiple of this number.
    :param open_folder:  Whether or not to open a folder showing your generated images.
    :param overwrite: Whether or not to overwrite existing generated images of the same name.
    :param deeper: Uses a Siren neural net with 32 hidden layers.
    :param image_width: The desired resolution of the image.
    :param seed: A seed to be used for deterministic runs.
    :param save_date_time: Save files with a timestamp prepended e.g. `%y%m%d-%H%M%S-my_phrase_here.png`
    :param start_image_path: Path to the image you would like to prime the generator with initially
    :param start_image_train_iters: Number of iterations for priming, defaults to 50
    :param theta_initial: Hyperparameter describing the frequency of the color space. Only applies to the first layer of the network.
    :param theta_hidden: Hyperparameter describing the frequency of the color space. Only applies to the hidden layers of the network.
    :param start_image_lr: Learning rate for the start image training.
    """
    # Don't instantiate imagine if the user just wants help.
    if any("--help" in arg for arg in sys.argv):
        print("Type `imagine --help` for usage info.")
        sys.exit()

    num_layers = 32 if deeper else num_layers

    imagine = Imagine(
        text,
        lr=learning_rate,
        num_layers=num_layers,
        batch_size=batch_size,
        gradient_accumulate_every=gradient_accumulate_every,
        epochs=epochs,
        iterations=iterations,
        image_width=image_width,
        save_every=save_every,
        save_progress=save_progress,
        seed=seed,
        open_folder=open_folder,
        save_date_time=save_date_time,
        start_image_path=start_image_path,
        start_image_train_iters=start_image_train_iters,
        theta_initial=theta_initial,
        theta_hidden=theta_hidden,
        start_image_lr=start_image_lr,
    )

    print('Starting up...')
    if not overwrite and imagine.filename.exists():
        answer = input('Imagined image already exists, do you want to overwrite? (y/n) ').lower()
        if answer not in ('yes', 'y'):
            sys.exit()

    imagine()


def main():
    fire.Fire(train)
