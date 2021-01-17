## Deep Daze

Deep Daze - Simple command line tool for text to image generation using OpenAI's <a href="https://github.com/openai/CLIP">CLIP</a> and Siren. Credit goes to <a href="https://twitter.com/advadnoun">Ryan Murdock</a> for the discovery of this technique (and for coming up with the great name)!

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

## Advanced

In true deep learning fashion, more layers will yield better results. Default is at `8`, but can be increased to `16` depending on your resources.

```bash
$ imagine "stranger in strange lands" --num-layers 16
```

If you would like to invoke it in code.

```python
from deep_daze import Imagine

imagine = Imagine(
    text = 'cosmic love and attention',
    num_layers = 10
)

imagine()
```

## Where is this going?

This is just a teaser. We will be able to generate images, sound, anything at will, with natural language. The holodeck is about to become real in our lifetimes.

Please join replication efforts for DALL-E for <a href="https://github.com/lucidrains/dalle-pytorch">Pytorch</a> or <a href="https://github.com/EleutherAI/DALLE-mtf">Mesh Tensorflow</a> if you are interested in furthering this technology.

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
    title={Implicit Neural Representations with Periodic Activation Functions},
    author={Vincent Sitzmann and Julien N. P. Martel and Alexander W. Bergman and David B. Lindell and Gordon Wetzstein},
    year={2020},
    eprint={2006.09661},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
