## Deep Daze
| | |
|---|---|
| <img src="./samples/Mist_over_green_hills.jpg" width="256px"></img> | <img src="./samples/Shattered_plates_on_the_grass.jpg" width="256px"></img> 
| *mist over green hills* | *shattered plates on the grass* |
| <img src="./samples/Cosmic_love_and_attention.jpg" width="256px"></img> | <img src="./samples/A_time_traveler_in_the_crowd.jpg" width="256px"></img> |
| *cosmic love and attention* | *a time traveler in the crowd* |
| <img src="./samples/Life_during_the_plague.jpg" width="256px"></img>| <img src="./samples/Meditative_peace_in_a_sunlit_forest.jpg" width="256px"></img> |
| *life during the plague* | *meditative peace in a sunlit forest* |
| <img src="./samples/A_man_painting_a_completely_red_image.png" width="256px"></img> | <img src="./samples/A_psychedelic_experience_on_LSD.png" width="256px"></img> |
| *a man painting a completely red image* | *a psychedelic experience on LSD* |

## What is this?

Simple command line tool for text to image generation using OpenAI's <a href="https://github.com/openai/CLIP">CLIP</a> and <a href="https://arxiv.org/abs/2006.09661">Siren</a>. Credit goes to <a href="https://twitter.com/advadnoun">Ryan Murdock</a> for the discovery of this technique (and for coming up with the great name)!

Original notebook [![Open In Colab][colab-badge]][colab-notebook]

New simplified notebook [![Open In Colab][colab-badge]][colab-notebook-2]

[colab-notebook]: <https://colab.research.google.com/drive/1FoHdqoqKntliaQKnMoNs3yn5EALqWtvP>
[colab-notebook-2]: <https://colab.research.google.com/drive/1_YOHdORb0Fg1Q7vWZ_KlrtFe9Ur3pmVj?usp=sharing>
[colab-badge]: <https://colab.research.google.com/assets/colab-badge.svg>

This will require that you have an Nvidia GPU or AMD GPU
- Recommended: 16GB VRAM
- Minimum Requirements: 4GB VRAM (Using VERY LOW settings, see usage instructions below) 

## Install

### Linux

Install using `pip`. It's always recommended to use a virtualenv if you are exploring several projects which use Torch, Tensorflow, etc.

```bash
$ python3 -m venv env
$ source env/bin/activate
$ pip install deep-daze
```  

### Windows Install

Install Python 3.6+ and install using `pip`. Like with Linux you shouldn't install this globally. If you have more than one Python installation confirm which is set in your path.

```bash
> python --version
Python 3.9.2
```

Or if you have Python 2 and 3 installed.

```bash
> python3 --version
Python 3.9.2
```

Create your virtualenv and install the package.

```bash
> python -m venv env
> .\env\Scripts\activate
> pip install deep-daze
```

### Verifying your Installation

Check that your Torch package was installed with CUDA enabled by running this one liner.

```bash
python -c "import torch; x = (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None); print(x)"
```

If the only output is `None` then you need to confirm that the CUDA drivers are installed and accessible. This will apply to Windows and Linux systems.

Windows with a RTX 2070

```bash
> python -c "import torch; x = (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None); print(x)"
NVIDIA GeForce RTX 2070
```

An AWS Deep Learning instance with a Tesla T4.

```bash
$ python -c "import torch; x = (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None); print(x)"
Tesla T4
```

A Windows laptop **without** the CUDA drivers installed.

```bash
> python -c "import torch; x = (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None); print(x)"
None
```

## Examples

```bash
$ imagine "a house in the forest"
```
For Windows:

```bash
> .\env\Scripts\activate
> imagine "a house in the forest"
```

That's it. If you have enough memory, you can get better quality by adding a `--deeper` flag

```bash
$ imagine "shattered plates on the ground" --deeper
```

By default this will output an image every 100 iterations, and 20 epochs with 1050 iterations results in 210 images from where you are running it. You can pass the `--output_folder` parameter and it will recursively create the directories and place the generated files in there instead of where you ran the command.

For Linux

```bash
$ imagine "a house in the forest" --output_folder="~/deep/thoughts"
```

For Windows

```bash
$ imagine "a house in the forest" --output_folder="C:\users\name\Documents\deepthoughts"
```

### Advanced

In true deep learning fashion, more layers will yield better results. Default is at `16`, but can be increased to `32` depending on your resources.

```bash
$ imagine "stranger in strange lands" --num-layers 32
```

## Usage

### CLI
```bash
NAME
    imagine

SYNOPSIS
    imagine <flags>

FLAGS
    --text=TEXT
        Type: Optional[]
        Default: None
        (required) A phrase less than 77 characters which you would 
        like to visualize.
    --img=IMG
        Type: Optional[]
        Default: None
        The path to a jpg or png image which you would like to imagine. 
        Can be combined with text.
    --learning_rate=LEARNING_RATE
        Default: 1e-05
        The learning rate of the neural net.
    --num_layers=NUM_LAYERS
        Default: 16
        The number of hidden layers to use in the Siren neural net.
    --hidden_size=HIDDEN_SIZE
        Default: 256
        The hidden layer size of the Siren net.
    --batch_size=BATCH_SIZE
        Default: 4
        The number of generated images to pass into Siren before calculating loss. 
        Decreasing this can lower memory and accuracy.
    --gradient_accumulate_every=GRADIENT_ACCUMULATE_EVERY
        Default: 4
        Calculate a weighted loss of n samples for each iteration. 
        Increasing this can help increase accuracy with lower batch sizes.
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
        Default: True
        Whether or not to save images generated before training Siren is complete.
    --seed=SEED
        Type: Optional[]
        Default: None
        A seed to be used for deterministic runs.
    --output_folder=OUTPUT_FOLDER
        Type: Optional[]
        Default: None
        The path to the folder to output the generated images, otherwise it will 
        generate them in the directory you are running it from.
    --open_folder=OPEN_FOLDER
        Default: True
        Whether or not to open a folder showing your generated images.
    --save_date_time=SAVE_DATE_TIME
        Default: False
        Save files with a timestamp prepended 
        e.g. `%y%m%d-%H%M%S-my_phrase_here.png`
    --start_image_path=START_IMAGE_PATH
        Type: Optional[]
        Default: None
        Path to the image you would like to prime the generator with initially
    --start_image_train_iters=START_IMAGE_TRAIN_ITERS
        Default: 50
        Number of iterations for priming, defaults to 50
    --theta_initial=THETA_INITIAL
        Type: Optional[]
        Default: None
        Hyperparameter describing the frequency of the color space. 
        Only applies to the first layer of the network.
    --theta_hidden=THETA_HIDDEN
        Type: Optional[]
        Default: None
        Hyperparameter describing the frequency of the color space. 
        Only applies to the hidden layers of the network.
    --start_image_lr=START_IMAGE_LR
        Default: 0.0003
        Learning rate for the start image training.
    --lower_bound_cutout=LOWER_BOUND_CUTOUT
        Default: 0.1
        The lower bound for the cutouts used in generation.
    --upper_bound_cutout=UPPER_BOUND_CUTOUT
        Default: 1.0
        The upper bound for the cutouts used in generation.
    --saturate_bound=SATURATE_BOUND
        Default: False
        If True, the LOWER_BOUND_CUTOUT is linearly increased to 0.75 
        during training.
    --create_story=CREATE_STORY
        Default: False
        Creates a story by optimizing each epoch on a new sliding-window 
        of the input words. If this is enabled, much longer texts than 77 
        chars can be used. Requires save_progress to visualize the 
        transitions of the story.
    --story_start_words=STORY_START_WORDS
        Default: 5
        Only used if create_story is True. How many words to optimize 
        on for the first epoch.
    --story_words_per_epoch=STORY_WORDS_PER_EPOCH
        Default: 5
        Only used if create_story is True. How many words to add to the 
        optimization goal per epoch after the first one.
    --story_separator=STORY_SEPARATOR
        Type: Optional[]
        Default: None
        Only used if create_story is True. Defines a separator like '.' that 
        splits the text into groups for each epoch. Separator needs to be in 
        the text otherwise it will be ignored!
    --averaging_weight=AVERAGING_WEIGHT
        Default: 0.3
        How much to weigh the averaged features of the random cutouts over the 
        individual random cutouts. Increasing this value leads to more details 
        being represented at the cost of some global coherence and a 
        parcellation into smaller scenes.
    --gauss_sampling=GAUSS_SAMPLING
        Default: False
        Whether to use sampling from a Gaussian distribution instead of a 
        uniform distribution.
    --gauss_mean=GAUSS_MEAN
        Default: 0.6
        The mean of the Gaussian sampling distribution.
    --gauss_std=GAUSS_STD
        Default: 0.2
        The standard deviation of the Gaussian sampling distribution.
    --do_cutout=DO_CUTOUT
        Default: True
    --center_bias=CENTER_BIAS
        Default: False
        Whether to use a Gaussian distribution centered around the center of 
        the image to sample the locations of random cutouts instead of a 
        uniform distribution. Leads to the main generated objects to be more 
        focused in the center.
    --center_focus=CENTER_FOCUS
        Default: 2
        How much to focus on the center if using center_bias. 
        
        std = sampling_range / center_focus. 
        
        High values lead to a very correct representation in the center but 
        washed out colors and details towards the edges,
    --jit=JIT
        Default: True
        Whether to use the jit-compiled CLIP model. The jit model is faster, 
        but only compatible with torch version 1.7.1.
    --save_gif=SAVE_GIF
        Default: False
        Only used if save_progress is True. Saves a GIF animation of the 
        generation procedure using the saved frames.
    --save_video=SAVE_VIDEO
        Default: False
        Only used if save_progress is True. Saves a MP4 animation of the 
        generation procedure using the saved frames.
    --model_name=MODEL_NAME
        Default: 'ViT-B/32'
        The model name to use. Options are RN50, RN101, RN50x4, and ViT-B/32.
    --optimizer=OPTIMIZER
        Default: 'AdamP'
        The optimizer to use. options are Adam, AdamP, and DiffGrad.
```

### Priming

Technique first devised and shared by <a href="https://twitter.com/quasimondo">Mario Klingemann</a>, it allows you to prime the generator network with a starting image, before being steered towards the text.

Simply specify the path to the image you wish to use, and optionally the number of initial training steps.

```bash
$ imagine 'a clear night sky filled with stars' --start_image_path ./cloudy-night-sky.jpg
```

| <img src="./samples/prime-orig.jpg" width="256px"></img> | <img src="./samples/prime-trained.png" width="256px"></img> |
|---|---|
| Primed starting image | Then trained with the prompt `A pizza with green pepper.`

### Optimize for the interpretation of an image

We can also feed in an image as an optimization goal, instead of only priming the generator network. Deepdaze will then render its own interpretation of that image:
```bash
$ imagine --img samples/Autumn_1875_Frederic_Edwin_Church.jpg
```

| <img src="./samples/Autumn_1875_Frederic_Edwin_Church_original.jpg" width="256px"></img> | <img src="./samples/Autumn_1875_Frederic_Edwin_Church.jpg" width="256px"></img> |
|---|---|
| Original image | The network's interpretation |


| <img src="./samples/hot-dog.jpg" width="256px"></img> | <img src="./samples/hot-dog_imagined.png" width="256px"></img> |
|---|---|
| Original image | The network's interpretation |

#### Optimize for text and image combined

```bash
$ imagine "A psychedelic experience." --img samples/hot-dog.jpg
```
The network's interpretation:  
<img src="./samples/psychedelic_hot_dog.png" width="256px"></img>

### New: Create a story
The regular mode for texts only allows 77 tokens. If you want to visualize a full story/paragraph/song/poem, set `create_story` to `True`. Given the poem below:

> Stopping by Woods On a Snowy Evening” by Robert Frost - 
"Whose woods these are I think I know. His house is in the village though; He will not see me stopping here To watch his woods fill up with snow. My little horse must think it queer To stop without a farmhouse near Between the woods and frozen lake The darkest evening of the year. He gives his harness bells a shake To ask if there is some mistake. The only other sound’s the sweep Of easy wind and downy flake. The woods are lovely, dark and deep, But I have promises to keep, And miles to go before I sleep, And miles to go before I sleep.".

We get:

https://user-images.githubusercontent.com/19983153/109539633-d671ef80-7ac1-11eb-8d8c-380332d7c868.mp4


### Python
#### Invoke `deep_daze.Imagine` in Python
```python
from deep_daze import Imagine

imagine = Imagine(
    text = 'cosmic love and attention',
    num_layers = 24,
)
imagine()
```

#### Save progress every fourth iteration
Save images in the format insert_text_here.00001.png, insert_text_here.00002.png, ...up to `(total_iterations % save_every)`
```python
imagine = Imagine(
    text=text,
    save_every=4,
    save_progress=True
)
```

#### Prepend current timestamp on each image.
Creates files with both the timestamp and the sequence number.

e.g. 210129-043928_328751_insert_text_here.00001.png, 210129-043928_512351_insert_text_here.00002.png, ...
```python
imagine = Imagine(
    text=text,
    save_every=4,
    save_progress=True,
    save_date_time=True,
)
```

#### High GPU memory usage
If you have at least 16 GiB of vram available, you should be able to run these settings with some wiggle room.
```python
imagine = Imagine(
    text=text,
    num_layers=42,
    batch_size=64,
    gradient_accumulate_every=1,
)
```

#### Average GPU memory usage
```python
imagine = Imagine(
    text=text,
    num_layers=24,
    batch_size=16,
    gradient_accumulate_every=2
)
```

#### Very low GPU memory usage (less than 4 GiB)
If you are desperate to run this on a card with less than 8 GiB vram, you can lower the image_width.
```python
imagine = Imagine(
    text=text,
    image_width=256,
    num_layers=16,
    batch_size=1,
    gradient_accumulate_every=16 # Increase gradient_accumulate_every to correct for loss in low batch sizes
)
```

### VRAM and speed benchmarks:
These experiments were conducted with a 2060 Super RTX and a 3700X Ryzen 5. We first mention the parameters (bs = batch size), then the memory usage and in some cases the training iterations per second:

For an image resolution of 512: 
* bs 1,  num_layers 22: 7.96 GB
* bs 2,  num_layers 20: 7.5 GB
* bs 16, num_layers 16: 6.5 GB

For an image resolution of 256:
* bs 8, num_layers 48: 5.3 GB
* bs 16, num_layers 48: 5.46 GB - 2.0 it/s
* bs 32, num_layers 48: 5.92 GB - 1.67 it/s
* bs 8, num_layers 44: 5 GB - 2.39 it/s
* bs 32, num_layers 44, grad_acc 1: 5.62 GB - 4.83 it/s
* bs 96, num_layers 44, grad_acc 1: 7.51 GB - 2.77 it/s
* bs 32, num_layers 66, grad_acc 1: 7.09 GB - 3.7 it/s
    
@NotNANtoN recommends a batch size of 32 with 44 layers and training 1-8 epochs.


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
