import os
import signal
import subprocess
import sys
import random
from datetime import datetime
from pathlib import Path
from shutil import copy

import torch
import torch.nn.functional as F
from siren_pytorch import SirenNet, SirenWrapper
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch_optimizer import DiffGrad, AdamP

from PIL import Image
import torchvision.transforms as T
from torchvision.utils import save_image

from tqdm import trange, tqdm

from deep_daze.clip import load, tokenize

assert torch.cuda.is_available(), 'CUDA must be available in order to use Deep Daze'

# graceful keyboard interrupt

terminate = False


def signal_handling(signum, frame):
    global terminate
    terminate = True


signal.signal(signal.SIGINT, signal_handling)

perceptor, normalize_image = load()


# Helpers

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d

def interpolate(image, size):
    return F.interpolate(image, (size, size), mode='bilinear', align_corners=False)


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
        cmd_list = ['explorer', path.replace('/', '\\')]
    if cmd_list == None:
        return

    try:
        subprocess.check_call(cmd_list)
    except subprocess.CalledProcessError:
        pass
    except OSError:
        pass


def norm_siren_output(img):
    return ((img + 1) * 0.5).clamp(0, 1)


class DeepDaze(nn.Module):
    def __init__(
            self,
            total_batches,
            batch_size,
            num_layers=8,
            image_width=512,
            loss_coef=100,
            theta_initial=None,
            theta_hidden=None
    ):
        super().__init__()
        # load clip

        self.loss_coef = loss_coef
        self.image_width = image_width

        self.batch_size = batch_size
        self.total_batches = total_batches
        self.num_batches_processed = 0

        w0 = default(theta_hidden, 30.)
        w0_initial = default(theta_initial, 30.)

        siren = SirenNet(
            dim_in=2,
            dim_hidden=256,
            num_layers=num_layers,
            dim_out=3,
            use_bias=True,
            w0=w0,
            w0_initial=w0_initial
        )

        self.model = SirenWrapper(
            siren,
            image_width=image_width,
            image_height=image_width
        )

        self.generate_size_schedule()

    def forward(self, text, return_loss=True):
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

        with autocast(enabled=False):
            image_embed = perceptor.encode_image(image)
            text_embed = perceptor.encode_text(text)

        self.num_batches_processed += self.batch_size

        loss = -self.loss_coef * torch.cosine_similarity(text_embed, image_embed, dim=-1).mean()
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
            partition = [4, 5, 3, 2, 1, 1]
        elif counter < 1000:
            partition = [2, 5, 4, 2, 2, 1]
        elif counter < 1500:
            partition = [1, 4, 5, 3, 2, 1]
        elif counter < 2000:
            partition = [1, 3, 4, 4, 2, 2]
        elif counter < 2500:
            partition = [1, 2, 2, 4, 4, 3]
        elif counter < 3000:
            partition = [1, 1, 2, 3, 4, 5]
        else:
            partition = [1, 1, 1, 2, 4, 7]

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
            lr=1e-5,
            batch_size=4,
            gradient_accumulate_every=4,
            save_every=100,
            image_width=512,
            num_layers=16,
            epochs=20,
            iterations=1050,
            save_progress=False,
            seed=None,
            open_folder=True,
            save_date_time=False,
            start_image_path=None,
            start_image_train_iters=10,
            start_image_lr=3e-4,
            theta_initial=None,
            theta_hidden=None,
    ):

        super().__init__()

        if exists(seed):
            tqdm.write(f'setting seed: {seed}')
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            random.seed(seed)
            torch.backends.cudnn.deterministic = True

        self.epochs = epochs
        self.iterations = iterations
        total_batches = epochs * iterations * batch_size * gradient_accumulate_every

        model = DeepDaze(
            total_batches=total_batches,
            batch_size=batch_size,
            image_width=image_width,
            num_layers=num_layers,
            theta_initial=theta_initial,
            theta_hidden=theta_hidden
        ).cuda()

        self.model = model
        self.scaler = GradScaler()
        self.optimizer = AdamP(model.parameters(), lr)
        self.gradient_accumulate_every = gradient_accumulate_every
        self.save_every = save_every
        self.save_date_time = save_date_time
        self.open_folder = open_folder
        self.save_progress = save_progress
        self.text = text
        self.textpath = text.replace(" ", "_")
        self.filename = self.image_output_path()
        self.encoded_text = tokenize(text).cuda()

        self.start_image = None
        self.start_image_train_iters = start_image_train_iters
        self.start_image_lr = start_image_lr
        if exists(start_image_path):
            file = Path(start_image_path)
            assert file.exists(), f'file does not exist at given starting image path {self.start_image_path}'
            image = Image.open(str(file))

            transform = T.Compose([
                T.Resize(image_width),
                T.CenterCrop((image_width, image_width)),
                T.ToTensor(),
                T.Normalize(0.5, 0.5)
            ])

            image_tensor = transform(image)[None, ...].cuda()
            self.start_image = image_tensor

    def image_output_path(self, sequence_number=None):
        """
        Returns underscore separated Path.
        A current timestamp is prepended if `self.save_date_time` is set.
        Sequence number left padded with 6 zeroes is appended if `save_every` is set.
        :rtype: Path
        """
        output_path = self.textpath
        if sequence_number:
            sequence_number_left_padded = str(sequence_number).zfill(6)
            output_path = f"{output_path}.{sequence_number_left_padded}"
        if self.save_date_time:
            current_time = datetime.now().strftime("%y%m%d-%H%M%S_%f")
            output_path = f"{current_time}_{output_path}"
        return Path(f"{output_path}.png")

    def generate_and_save_image(self, sequence_number=None):
        """
        :param sequence_number:
        :param custom_filename: A custom filename to use when saving - e.g. "testing.png"
        """
        with torch.no_grad():
            img = normalize_image(self.model(self.encoded_text, return_loss=False).cpu())
            img.clamp_(0., 1.)
            self.filename = self.image_output_path(sequence_number=sequence_number)
            save_image(img, self.filename)
            save_image(img, f"{self.textpath}.png")

            tqdm.write(f'image updated at "./{str(self.filename)}"')

    def train_step(self, epoch, iteration) -> int:
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

        if (iteration % self.save_every == 0) and self.save_progress:
            current_total_iterations = epoch * self.iterations + iteration
            sequence_number = current_total_iterations // self.save_every
            self.generate_and_save_image(sequence_number=sequence_number)

        return total_loss

    def forward(self):
        if exists(self.start_image):
            tqdm.write('Preparing with initial image...')
            optim = DiffGrad(self.model.parameters(), lr = self.start_image_lr)
            pbar = trange(self.start_image_train_iters, desc='iteration')
            for _ in pbar:
                loss = self.model.model(self.start_image)
                loss.backward()
                pbar.set_description(f'loss: {loss.item():.2f}')

                optim.step()
                optim.zero_grad()

                if terminate:
                    print('interrupted by keyboard, gracefully exiting')
                    return exit()

            del self.start_image
            del optim

        tqdm.write(f'Imagining "{self.text}" from the depths of my weights...')

        self.model(self.encoded_text) # do one warmup step due to potential issue with CLIP and CUDA

        if self.open_folder:
            open_folder('./')
            self.open_folder = False

        for epoch in trange(self.epochs, desc='epochs'):
            pbar = trange(self.iterations, desc='iteration')
            for i in pbar:
                loss = self.train_step(epoch, i)
                pbar.set_description(f'loss: {loss.item():.2f}')

                if terminate:
                    print('interrupted by keyboard, gracefully exiting')
                    return
