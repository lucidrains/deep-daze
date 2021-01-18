import torch
import torch.nn.functional as F
from random import sample
from torch import nn
from torch.optim import Adam
from torch.cuda.amp import GradScaler, autocast

from pathlib import Path
from tqdm import trange
import torchvision

from deep_daze.clip import load, tokenize, normalize_image
from siren_pytorch import SirenNet, SirenWrapper

from collections import namedtuple
from einops import rearrange

assert torch.cuda.is_available(), 'CUDA must be available in order to use Deep Daze'

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

def norm_siren_output(img):
    return (img.tanh() + 1) * 0.5

class DeepDaze(nn.Module):
    def __init__(
        self,
        num_layers = 8,
        image_width = 512,
        loss_coef = 100,
        batch_size = 4
    ):
        super().__init__()
        self.loss_coef = loss_coef
        self.image_width = image_width

        self.batch_size = batch_size
        self.sizing_schedule_counter = 0

        siren = SirenNet(
            dim_in = 2,
            dim_hidden = 256,
            num_layers = num_layers,
            dim_out = 3,
            use_bias = True
        )

        self.model = SirenWrapper(
            siren,
            image_width = image_width,
            image_height = image_width
        )

    def forward(self, text, return_loss = True):
        out = self.model()
        out = norm_siren_output(out)

        if not return_loss:
            return out

        pieces = []
        width = out.shape[-1]

        for size in sample(self.sample_sizes(), self.batch_size):
            offsetx = torch.randint(0, width - size, ())
            offsety = torch.randint(0, width - size, ())
            apper = out[:, :, offsetx:offsetx + size, offsety:offsety + size]
            apper = torch.nn.functional.interpolate(apper, (224, 224), mode = 'bilinear', align_corners = False)
            pieces.append(normalize_image(apper))

        image = torch.cat(pieces)

        with autocast(enabled = False):
            image_embed = perceptor.encode_image(image)
            text_embed = perceptor.encode_text(text)

        loss = -self.loss_coef * torch.cosine_similarity(text_embed, image_embed, dim = -1).mean()
        return loss

    def sample_sizes(self):
        self.sizing_schedule_counter+=1
        counter = self.sizing_schedule_counter
        pieces_per_group = 4
        # 6 piece schedule increasing in context as model saturates
        if counter<500:
            partition = [4,5,3,2,1,1]
        elif counter<1000:
            partition = [2,5,4,2,2,1]
        elif counter<1500:
            partition = [1,4,5,3,2,1]
        elif counter<2000:
            partition = [1,3,4,4,2,2]
        elif counter<2500:
            partition = [1,2,2,4,4,3]
        elif counter<3000:
            partition = [1,1,2,3,4,5]
        else:
            partition = [1,1,1,2,4,7]

        dbase = .38
        step = .1
        width = self.image_width

        sizes = []
        for part_index in range(len(partition)):
            groups = partition[part_index]
            for _ in range(groups*pieces_per_group):
                sizes.append(torch.randint(
                    int((dbase+step*part_index+.01)*width),
                    int((dbase+step*(1+part_index))*width), ()))

        sizes.sort()
        return sizes


class Imagine(nn.Module):
    def __init__(
        self,
        text,
        *,
        lr = 1e-5,
        batch_size = 4,
        gradient_accumulate_every = 4,
        save_every = 100,
        image_width = 512,
        num_layers = 16
    ):
        super().__init__()

        model = DeepDaze(
            batch_size = batch_size,
            image_width = image_width,
            num_layers = num_layers
        ).cuda()

        self.model = model

        self.scaler = GradScaler()
        self.optimizer = Adam(model.parameters(), lr)
        self.gradient_accumulate_every = gradient_accumulate_every

        self.text = text

        textpath = self.text.replace(' ','_')
        self.filename = Path(f'./{textpath}.png')

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
            for i in trange(1050, desc='iteration'):
                self.train_step(epoch, i)
