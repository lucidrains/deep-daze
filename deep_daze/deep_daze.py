import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.cuda.amp import GradScaler, autocast

import torchvision
from torchvision.utils import save_image

import os
import sys
import signal
import subprocess
from collections import namedtuple
from pathlib import Path
from tqdm import trange

from deep_daze.clip import load, tokenize
from siren_pytorch import SirenNet, SirenWrapper

from einops import rearrange

assert torch.cuda.is_available(), 'CUDA must be available in order to use Deep Daze'

# graceful keyboard interrupt

terminate = False                            

def signal_handling(signum,frame):           
    global terminate                         
    terminate = True                         

signal.signal(signal.SIGINT,signal_handling) 

# helpers

def exists(val):
    return val is not None

def interpolate(image, size):
    return F.interpolate(image, (size, size), mode = 'bilinear', align_corners = False)

def rand_cutout(image, size):
    width = image.shape[-1]
    offsetx = torch.randint(0, width - size, ())
    offsety = torch.randint(0, width - size, ())
    cutout = image[:, :, offsetx:offsetx + size, offsety:offsety + size]
    return cutout

def open_folder(path):
    if os.path.isfile(path):
        path = os.path.dirname(path)

    if not os.path.isdir(path):
        return

    cmd_list = None
    if sys.platform == 'darwin':
        cmd_list = ['open', '--', path]
    elif sys.platform == 'linux2' or sys.platform == 'linux':
        cmd_list = ['xdg-open', path]
    elif sys.platform in ['win32', 'win64']:
        cmd_list = ['explorer', path.replace('/','\\')]
    if cmd_list == None:
        return

    try:
        subprocess.check_call(cmd_list)
    except subprocess.CalledProcessError:
        pass
    except OSError:
        pass

# load clip

perceptor, normalize_image = load()

# load siren

def norm_siren_output(img):
    return ((img + 1) * 0.5).clamp(0, 1)

class DeepDaze(nn.Module):
    def __init__(
        self,
        total_batches,
        batch_size,
        num_layers = 8,
        image_width = 512,
        loss_coef = 100,
    ):
        super().__init__()
        self.loss_coef = loss_coef
        self.image_width = image_width

        self.batch_size = batch_size
        self.total_batches = total_batches
        self.num_batches_processed = 0

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

        self.generate_size_schedule()

    def forward(self, text, return_loss = True):
        out = self.model()
        out = norm_siren_output(out)

        if not return_loss:
            return out

        pieces = []
        width = out.shape[-1]
        size_slice = slice(self.num_batches_processed, self.num_batches_processed + self.batch_size)

        for size in self.scheduled_sizes[size_slice]:
            apper = rand_cutout(out, size)
            apper = interpolate(apper, 224)
            pieces.append(normalize_image(apper))

        image = torch.cat(pieces)

        with autocast(enabled = False):
            image_embed = perceptor.encode_image(image)
            text_embed = perceptor.encode_text(text)

        self.num_batches_processed += self.batch_size

        loss = -self.loss_coef * torch.cosine_similarity(text_embed, image_embed, dim = -1).mean()
        return loss

    def generate_size_schedule(self):
        batches = 0
        counter = 0
        self.scheduled_sizes = []

        while batches < self.total_batches:
            counter += 1
            sizes = self.sample_sizes(counter)
            batches += len(sizes)
            self.scheduled_sizes.extend(sizes)

    def sample_sizes(self, counter):
        pieces_per_group = 4

        # 6 piece schedule increasing in context as model saturates
        if counter < 500:
            partition = [4,5,3,2,1,1]
        elif counter < 1000:
            partition = [2,5,4,2,2,1]
        elif counter < 1500:
            partition = [1,4,5,3,2,1]
        elif counter < 2000:
            partition = [1,3,4,4,2,2]
        elif counter < 2500:
            partition = [1,2,2,4,4,3]
        elif counter < 3000:
            partition = [1,1,2,3,4,5]
        else:
            partition = [1,1,1,2,4,7]

        dbase = .38
        step = .1
        width = self.image_width

        sizes = []
        for part_index in range(len(partition)):
            groups = partition[part_index]
            for _ in range(groups * pieces_per_group):
                sizes.append(torch.randint(
                    int((dbase + step * part_index + .01) * width),
                    int((dbase + step * (1 + part_index)) * width), ()))

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
        num_layers = 16,
        epochs = 20,
        iterations = 1050,
        save_progress = False,
        seed = None,
        open_folder = True
    ):
        super().__init__()

        if exists(seed):
            print(f'setting seed: {seed}')
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            random.seed(seed)
            torch.backends.cudnn.deterministic=True

        self.epochs = epochs
        self.iterations = iterations
        total_batches = epochs * iterations * batch_size * gradient_accumulate_every

        model = DeepDaze(
            total_batches = total_batches,
            batch_size = batch_size,
            image_width = image_width,
            num_layers = num_layers
        ).cuda()

        self.model = model

        self.scaler = GradScaler()
        self.optimizer = Adam(model.parameters(), lr)
        self.gradient_accumulate_every = gradient_accumulate_every
        self.save_every = save_every

        self.text = text
        textpath = self.text.replace(' ','_')

        self.textpath = textpath
        self.filename = Path(f'./{textpath}.png')
        self.save_progress = save_progress

        self.encoded_text = tokenize(text).cuda()

        self.open_folder = open_folder

    def train_step(self, epoch, i):
        total_loss = 0

        for _ in range(self.gradient_accumulate_every):
            with autocast():
                loss = self.model(self.encoded_text)
            loss = loss / self.gradient_accumulate_every
            total_loss += loss
            self.scaler.scale(loss).backward()

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        if i % self.save_every == 0:
            with torch.no_grad():
                img = normalize_image(self.model(self.encoded_text, return_loss = False).cpu())
                img.clamp_(0., 1.)
                save_image(img, str(self.filename))
                print(f'image updated at "./{str(self.filename)}"')

                if self.save_progress:
                    current_total_iterations = epoch * self.iterations + i
                    num = current_total_iterations // self.save_every
                    save_image(img, Path(f'./{self.textpath}.{num}.png'))

        return total_loss

    def forward(self):
        print(f'Imagining "{self.text}" from the depths of my weights...')

        if self.open_folder:
            open_folder('./')
            self.open_folder = False

        for epoch in trange(self.epochs, desc = 'epochs'):
            pbar = trange(self.iterations, desc='iteration')
            for i in pbar:
                loss = self.train_step(epoch, i)
                pbar.set_description(f'loss: {loss.item():.2f}')

                if terminate:
                    print('interrupted by keyboard, gracefully exiting')
                    return
