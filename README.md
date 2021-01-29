## Deep Daze

<img src="./samples/mist-over-green-hills.png" width="200px"></img>

*mist over green hills*

<img src="./samples/shattered-plates.png" width="200px"></img>

*shattered plates on the grass*

<img src="./samples/cosmic-love.png" width="200px"></img>

*cosmic love and attention*

<img src="./samples/time-traveler.png" width="200px"></img>

*a time traveler in the crowd*

<img src="./samples/life-plague.png" width="200px"></img>

*life during the plague*

<img src="./samples/peace-sunlit-forest.png" width="200px"></img>

*meditative peace in a sunlit forest*

## What is this?

Simple command line tool for text to image generation using OpenAI's <a href="https://github.com/openai/CLIP">CLIP</a> and <a href="https://arxiv.org/abs/2006.09661">Siren</a>. Credit goes to <a href="https://twitter.com/advadnoun">Ryan Murdock</a> for the discovery of this technique (and for coming up with the great name)!

Original notebook [![Open In Colab][colab-badge]][colab-notebook]

New simplified notebook [![Open In Colab][colab-badge]][colab-notebook-2]

[colab-notebook]: <https://colab.research.google.com/drive/1FoHdqoqKntliaQKnMoNs3yn5EALqWtvP>
[colab-notebook-2]: <https://colab.research.google.com/drive/1_YOHdORb0Fg1Q7vWZ_KlrtFe9Ur3pmVj?usp=sharing>
[colab-badge]: <https://colab.research.google.com/assets/colab-badge.svg>

This will require that you have an Nvidia GPU

## Install

```bash
$ pip install deep-daze
```

## Examples

```bash
$ imagine "a house in the forest"
```

That's it.

If you have enough memory, you can get better quality by adding a `--deeper` flag

```bash
$ imagine "shattered plates on the ground" --deeper
```

### Advanced

In true deep learning fashion, more layers will yield better results. Default is at `16`, but can be increased to `32` depending on your resources.

```bash
$ imagine "stranger in strange lands" --num-layers 32
```

## Usage
```bash
NAME
    imagine

SYNOPSIS
    imagine TEXT <flags>

POSITIONAL ARGUMENTS
    TEXT
        (required) A phrase less than 77 characters which you would like to visualize.

FLAGS
    --learning_rate=LEARNING_RATE
        Default: 1e-05
        The learning rate of the neural net.
    --num_layers=NUM_LAYERS
        Default: 16
        The number of hidden layers to use in the Siren neural net.
    --batch_size=BATCH_SIZE
        Default: 4
        The number of generated images to pass into Siren before calculating loss. Decreasing this can lower memory and accuracy.
    --gradient_accumulate_every=GRADIENT_ACCUMULATE_EVERY
        Default: 4
        Calculate a weighted loss of n samples for each iteration. Increasing this can help increase accuracy with lower batch sizes.
    --epochs=EPOCHS
        Default: 20
        The number of epochs to run.
    --iterations=ITERATIONS
        Default: 1050
        The number of times to calculate and backpropagate loss in a given epoch.
    --save_every=SAVE_EVERY
        Default: 100
        Generate an image every time iterations is a multiple of this number.
    --image_width=IMAGE_WIDTH
        Default: 512
        The desired resolution of the image.
    --deeper=DEEPER
        Default: False
        Uses a Siren neural net with 32 hidden layers.
    --overwrite=OVERWRITE
        Default: False
        Whether or not to overwrite existing generated images of the same name.
    --save_progress=SAVE_PROGRESS
        Default: False
        Whether or not to save images generated before training Siren is complete.
    --seed=SEED
        Type: Optional[]
        Default: None
        A seed to be used for deterministic runs.
    --open_folder=OPEN_FOLDER
        Default: True
        Whether or not to open a folder showing your generated images.
    --save_date_time=SAVE_DATE_TIME
        Default: False
        Save files with a timestamp prepended e.g. `%y%m%d-%H%M%S-my_phrase_here`
```

If you would like to invoke it in code.

```python
from deep_daze import Imagine

imagine = Imagine(
    text = 'cosmic love and attention',
    num_layers = 24
)

imagine()
```

## Where is this going?

This is just a teaser. We will be able to generate images, sound, anything at will, with natural language. The holodeck is about to become real in our lifetimes.

Please join replication efforts for DALL-E for <a href="https://github.com/lucidrains/dalle-pytorch">Pytorch</a> or <a href="https://github.com/EleutherAI/DALLE-mtf">Mesh Tensorflow</a> if you are interested in furthering this technology.

## Alternatives

<a href="https://github.com/lucidrains/big-sleep">Big Sleep</a> - CLIP and the generator from Big GAN

## Citations

```bibtex
@misc{unpublished2021clip,
    title  = {CLIP: Connecting Text and Images},
    author = {Alec Radford, Ilya Sutskever, Jong Wook Kim, Gretchen Krueger, Sandhini Agarwal},
    year   = {2021}
}
```

```bibtex
@misc{sitzmann2020implicit,
    title   = {Implicit Neural Representations with Periodic Activation Functions},
    author  = {Vincent Sitzmann and Julien N. P. Martel and Alexander W. Bergman and David B. Lindell and Gordon Wetzstein},
    year    = {2020},
    eprint  = {2006.09661},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

[colab-notebook]: <https://colab.research.google.com/drive/1FoHdqoqKntliaQKnMoNs3yn5EALqWtvP>
