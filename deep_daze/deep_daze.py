import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.cuda.amp import GradScaler, autocast

from pathlib import Path
from tqdm import trange
import torchvision

from deep_daze.clip import load, tokenize, normalize_image
from siren_pytorch import SirenNet

from collections import namedtuple
from einops import rearrange

assert torch.cuda.is_available(), 'CUDA must be available in order to use Deep Daze'

# constants

RegConfig = namedtuple('RegConfig', ['num', 'ratio', 'downsized_image_size'])

DEFAULT_REG_CONFIG = [
    RegConfig(num = 3, ratio = (0.5, 0.95), downsized_image_size = None),
    RegConfig(num = 1, ratio = (0.5, 0.95), downsized_image_size = 64),
]

# helpers

def exists(val):
    return val is not None

def interpolate(image, size):
    return F.interpolate(image, (size, size), mode = 'bilinear', align_corners = False)

def rand_cutout(image, ratio = (0.5, 0.95)):
    lo, hi, width = *ratio, image.shape[-1]
    size = torch.randint(int(lo * width), int(hi * width), ())
    offsetx = torch.randint(0, width - size, ())
    offsety = torch.randint(0, width - size, ())
    cutout = image[:, :, offsetx:offsetx + size, offsety:offsety + size]
    return cutout

# load clip

perceptor, preprocess = load()

# load siren

class SirenWrapper(nn.Module):
    def __init__(self, net, image_width, image_height):
        super().__init__()
        self.net = net
        self.image_width = image_width
        self.image_height = image_height

        tensors = [torch.linspace(-1, 1, steps = image_width), torch.linspace(-1, 1, steps = image_height)]
        mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
        mgrid = rearrange(mgrid, 'h w c -> (h w) c')
        self.register_buffer('grid', mgrid)

    def forward(self):
        coords = self.grid.clone().detach().requires_grad_()
        out = self.net(coords)
        out = rearrange(out, '(h w) c -> () c h w', h = self.image_height, w = self.image_width)
        out = (out.tanh() + 1) * 0.5
        return out

class DeepDaze(nn.Module):
    def __init__(
        self,
        num_layers = 8,
        image_width = 512,
        loss_coef = 100,
        reg_config = DEFAULT_REG_CONFIG
    ):
        super().__init__()
        self.loss_coef = loss_coef
        self.image_width = image_width

        self.model = SirenWrapper(
            SirenNet(
                dim_in = 2,
                dim_hidden = 256,
                num_layers = num_layers,
                dim_out = 3,
                use_bias = True
            ),
            image_width = image_width,
            image_height = image_width
        )

        self.reg_config = reg_config

    def forward(self, text, return_loss = True):
        width = self.image_width
        out = self.model()

        if not return_loss:
            return out

        cutout_specs = self.reg_config

        pieces = []

        for (num_images, (lo, hi), downsize) in cutout_specs:
            for _ in range(num_images):
                cutout = rand_cutout(out, ratio = (lo, hi))
                if exists(downsize):
                    cutout = interpolate(cutout, downsize)
                resized_cutout = interpolate(cutout, 224)
                pieces.append(normalize_image(resized_cutout))

        image = torch.cat(pieces)

        with autocast(enabled = False):
          image_embed = perceptor.encode_image(image)
          text_embed = perceptor.encode_text(text)

        loss = -self.loss_coef * torch.cosine_similarity(text_embed, image_embed, dim = -1).mean()
        return loss

class Imagine(nn.Module):
    def __init__(
        self,
        text,
        *,
        lr = 1e-5,
        gradient_accumulate_every = 4,
        save_every = 100,
        image_width = 512,
        num_layers = 16,
        reg_config = DEFAULT_REG_CONFIG
    ):
        super().__init__()

        model = DeepDaze(
            image_width = image_width,
            num_layers = num_layers,
            reg_config = reg_config
        ).cuda()

        self.model = model

        self.scaler = GradScaler()
        self.optimizer = Adam(model.parameters(), lr)
        self.gradient_accumulate_every = gradient_accumulate_every

        self.text = text
        self.filename = Path(f'./{self.text}.png')

        self.encoded_text = tokenize(text).cuda()
        self.save_every = save_every

    def train_step(self, epoch, i):

        for _ in range(self.gradient_accumulate_every):
            with autocast():
                loss = self.model(self.encoded_text)
            self.scaler.scale(loss / self.gradient_accumulate_every).backward()

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        if i % self.save_every == 0:
            with torch.no_grad():
                al = normalize_image(self.model(self.encoded_text, return_loss = False).cpu())
                torchvision.utils.save_image(al, str(self.filename))
                print(f'image updated at "./{str(self.filename)}"')

    def forward(self):
        print(f'Imagining "{self.text}" from the depths of my weights...')

        for epoch in trange(20, desc = 'epochs'):
            for i in trange(1000, desc = 'iteration'):
                self.train_step(epoch, i)
