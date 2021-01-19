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

## Usage

```bash
$ imagine "a house in the forest"
```

That's it.

If you have enough memory, you can get better quality by adding a `--deeper` flag

```bash
$ imagine "shattered plates on the ground" --deeper
```

## Advanced

In true deep learning fashion, more layers will yield better results. Default is at `16`, but can be increased to `32` depending on your resources.

```bash
$ imagine "stranger in strange lands" --num-layers 32
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
