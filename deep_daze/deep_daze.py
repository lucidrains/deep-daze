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

from einops import rearrange

assert torch.cuda.is_available(), 'CUDA must be available in order to use Deep Daze'

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
        loss_coef = 100
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
                use_bias = False
            ),
            image_width = image_width,
            image_height = image_width
        ).cuda()

    def forward(self, text, return_loss = True):
        width = self.image_width
        out = self.model()

        if not return_loss:
            return out

        cutn = 64
        pieces = []
        for ch in range(cutn):
            size = torch.randint(int(.5 * width), int(.98 * width), ())
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

class Imagine(nn.Module):
    def __init__(
        self,
        text,
        *,
        lr = 1e-5,
        save_every = 150,
        image_width = 512,
        num_layers = 8
    ):
        super().__init__()
        model = DeepDaze(
            image_width = image_width,
            num_layers = num_layers
        )

        self.model = model
        self.optimizer = Adam(model.parameters(), lr)
        self.scaler = GradScaler()

        self.text = text
        self.filename = Path(f'./{self.text}.png')

        self.encoded_text = tokenize(text).cuda()
        self.save_every = save_every

    def train_step(self, epoch, i):
        self.optimizer.zero_grad()

        with autocast():
            loss = self.model(self.encoded_text)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if i % self.save_every == 0:
            with torch.no_grad():
                al = normalize_image(self.model(self.encoded_text, return_loss = False).cpu())
                torchvision.utils.save_image(al, str(self.filename))
                print(f'image saved to {str(self.filename)}')

    def forward(self):
        print(f'Imagining "{self.text}" from the depths of my weights...')

        for epoch in trange(10000, desc = 'epochs'):
            for i in trange(1000, desc = 'iteration'):
                self.train_step(epoch, i)
