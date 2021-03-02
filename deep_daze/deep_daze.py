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
    return ((img + 1) * 0.5).clamp(0.0, 1.0)


def create_clip_img_transform(image_width):
    clip_mean = [0.48145466, 0.4578275, 0.40821073]
    clip_std = [0.26862954, 0.26130258, 0.27577711]
    transform = T.Compose([
                    #T.ToPILImage(),
                    T.Resize(image_width),
                    T.CenterCrop((image_width, image_width)),
                    T.ToTensor(),
                    T.Normalize(mean=clip_mean, std=clip_std)
            ])
    return transform


def create_text_path(text=None, img=None, encoding=None):
    if text is not None:
        input_name = text.replace(" ", "_")[:perceptor.context_length]
    elif img is not None:
        if isinstance(img, str):
            input_name = "".join(img.replace(" ", "_").split(".")[:-1])
        else:
            input_name = "PIL_img"
    else:
        input_name = "your_encoding"
    return input_name


class DeepDaze(nn.Module):
    def __init__(
            self,
            total_batches,
            batch_size,
            num_layers=8,
            image_width=512,
            loss_coef=100,
            theta_initial=None,
            theta_hidden=None,
            lower_bound_cutout=0.1, # should be smaller than 0.8
            upper_bound_cutout=1.0,
            saturate_bound=False,
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

        self.saturate_bound = saturate_bound
        self.saturate_limit = 0.75 # cutouts above this value lead to destabilization
        self.lower_bound_cutout = lower_bound_cutout
        self.upper_bound_cutout = upper_bound_cutout


    def forward(self, text_embed, return_loss=True, dry_run=False):
        out = self.model()
        out = norm_siren_output(out)

        if not return_loss:
            return out

        # sample cutout sizes between lower and upper bound
        width = out.shape[-1]
        lower_bound = self.lower_bound_cutout
        if self.saturate_bound:
            progress_fraction = self.num_batches_processed / self.total_batches
            lower_bound += (self.saturate_limit - self.lower_bound_cutout) * progress_fraction
            
        lower = lower_bound * width
        upper = self.upper_bound_cutout * width
        sizes = torch.randint(int(lower), int(upper), (self.batch_size,))

        # create normalized random cutouts
        image_pieces = torch.cat([normalize_image(interpolate(rand_cutout(out, size), 224)) for size in sizes])
        # calc image embedding
        with autocast(enabled=False):
            image_embed = perceptor.encode_image(image_pieces)
        # count batches
        if not dry_run:
            self.num_batches_processed += self.batch_size
        # calc loss
        loss = -self.loss_coef * torch.cosine_similarity(text_embed, image_embed, dim=-1).mean()
        return out, loss


class Imagine(nn.Module):
    def __init__(
            self,
            *,
            text=None,
            img=None,
            clip_encoding=None,
            lr=1e-5,
            batch_size=4,
            gradient_accumulate_every=4,
            save_every=100,
            image_width=512,
            num_layers=16,
            epochs=20,
            iterations=1050,
            save_progress=True,
            seed=None,
            open_folder=True,
            save_date_time=False,
            start_image_path=None,
            start_image_train_iters=10,
            start_image_lr=3e-4,
            theta_initial=None,
            theta_hidden=None,
            lower_bound_cutout=0.1, # should be smaller than 0.8
            upper_bound_cutout=1.0,
            saturate_bound=False,
            create_story=False,
            story_start_words=5,
            story_words_per_epoch=5,
    ):

        super().__init__()

        if exists(seed):
            tqdm.write(f'setting seed: {seed}')
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            random.seed(seed)
            torch.backends.cudnn.deterministic = True
            
        # fields for story creation:
        self.create_story = create_story
        self.words = None
        self.all_words = text.split(" ") if text is not None else None
        self.num_start_words = story_start_words
        self.words_per_epoch = story_words_per_epoch
        if create_story:
            assert text is not None,  "We need text input to create a story..."
            # overwrite epochs to match story length
            num_words = len(self.all_words)
            self.epochs = 1 + (num_words - self.num_start_words) / self.words_per_epoch
            # add one epoch if not divisible
            self.epochs = int(self.epochs) if int(self.epochs) == self.epochs else int(self.epochs) + 1
            print("Running for ", self.epochs, "epochs")
        else: 
            self.epochs = epochs
        
        self.iterations = iterations
        self.image_width = image_width
        total_batches = self.epochs * self.iterations * batch_size * gradient_accumulate_every
        model = DeepDaze(
            total_batches=total_batches,
            batch_size=batch_size,
            image_width=image_width,
            num_layers=num_layers,
            theta_initial=theta_initial,
            theta_hidden=theta_hidden,
            lower_bound_cutout=lower_bound_cutout,
            upper_bound_cutout=upper_bound_cutout,
            saturate_bound=saturate_bound,
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
        self.image = img
        self.textpath = create_text_path(text=text, img=img, encoding=clip_encoding)
        self.filename = self.image_output_path()
        
        # create coding to optimize for
        self.clip_img_transform = create_clip_img_transform(perceptor.input_resolution.item())
        self.clip_encoding = self.create_clip_encoding(text=text, img=img, encoding=clip_encoding)

        self.start_image = None
        self.start_image_train_iters = start_image_train_iters
        self.start_image_lr = start_image_lr
        if exists(start_image_path):
            file = Path(start_image_path)
            assert file.exists(), f'file does not exist at given starting image path {self.start_image_path}'
            image = Image.open(str(file))

            image_tensor = self.clip_img_transform(image)[None, ...].cuda()
            self.start_image = image_tensor
            
    def create_clip_encoding(self, text=None, img=None, encoding=None):
        self.text = text
        self.img = img
        if encoding is not None:
            encoding = encoding.cuda()
        elif self.create_story:
            encoding = self.update_story_encoding(epoch=0, iteration=1)
        elif text is not None and img is not None:
            encoding = (self.create_text_encoding(text) + self.create_img_encoding(img)) / 2
        elif text is not None:
            encoding = self.create_text_encoding(text)
        elif img is not None:
            encoding = self.create_img_encoding(img)
        return encoding

    def create_text_encoding(self, text):
        tokenized_text = tokenize(text).cuda()
        with torch.no_grad():
            text_encoding = perceptor.encode_text(tokenized_text).detach()
        return text_encoding
    
    def create_img_encoding(self, img):
        if isinstance(img, str):
            img = Image.open(img)
        normed_img = self.clip_img_transform(img).unsqueeze(0).cuda()
        with torch.no_grad():
            img_encoding = perceptor.encode_image(normed_img).detach()
        return img_encoding
    
    def set_clip_encoding(self, text=None, img=None, encoding=None):
        encoding = self.create_clip_encoding(text=text, img=img, encoding=encoding)
        self.clip_encoding = encoding.cuda()
        
    def update_story_encoding(self, epoch, iteration):
        if self.words is None:
            self.words = " ".join(self.all_words[:self.num_start_words])
            self.all_words = self.all_words[self.num_start_words:]
        else:
            # add words_per_epoch new words
            count = 0
            while count < self.words_per_epoch and len(self.all_words) > 0:
                new_word = self.all_words[0]
                self.words = " ".join(self.words.split(" ") + [new_word])
                self.all_words = self.all_words[1:]
                count += 1
                # TODO: possibly do not increase count for stop-words and break if a "." is encountered.
            # remove words until it fits in context length
            while len(self.words) > perceptor.context_length:
                # remove first word
                self.words = " ".join(self.words.split(" ")[1:])
        # get new encoding
        print("Now thinking of: ", '"', self.words, '"')
        sequence_number = self.get_img_sequence_number(epoch, iteration)
        # save new words to disc
        with open("story_transitions.txt", "a") as f:
            f.write(f"{epoch}, {sequence_number}, {self.words}\n")
        
        encoding = self.create_text_encoding(self.words)
        return encoding

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
        return Path(f"{output_path}.jpg")

    def train_step(self, epoch, iteration):
        total_loss = 0

        for _ in range(self.gradient_accumulate_every):
            with autocast():
                out, loss = self.model(self.clip_encoding)
            loss = loss / self.gradient_accumulate_every
            total_loss += loss
            self.scaler.scale(loss).backward()    
        out = out.cpu().float().clamp(0., 1.)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        if (iteration % self.save_every == 0) and self.save_progress:
            self.save_image(epoch, iteration, img=out)

        return out, total_loss
    
    def get_img_sequence_number(self, epoch, iteration):
        current_total_iterations = epoch * self.iterations + iteration
        sequence_number = current_total_iterations // self.save_every
        return sequence_number

    @torch.no_grad()
    def save_image(self, epoch, iteration, img=None):
        sequence_number = self.get_img_sequence_number(epoch, iteration)

        if img is None:
            img = self.model(self.clip_encoding, return_loss=False).cpu().float().clamp(0., 1.)
        self.filename = self.image_output_path(sequence_number=sequence_number)
        
        pil_img = T.ToPILImage()(img.squeeze())
        pil_img.save(self.filename, quality=95, subsampling=0)
        pil_img.save(f"{self.textpath}.jpg", quality=95, subsampling=0)
        #save_image(img, self.filename)
        #save_image(img, f"{self.textpath}.png")

        tqdm.write(f'image updated at "./{str(self.filename)}"')

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

        tqdm.write(f'Imagining "{self.textpath}" from the depths of my weights...')

        with torch.no_grad():
            self.model(self.clip_encoding, dry_run=True) # do one warmup step due to potential issue with CLIP and CUDA

        if self.open_folder:
            open_folder('./')
            self.open_folder = False

        for epoch in trange(self.epochs, desc='epochs'):
            pbar = trange(self.iterations, desc='iteration')
            for i in pbar:
                _, loss = self.train_step(epoch, i)
                pbar.set_description(f'loss: {loss.item():.2f}')

                if terminate:
                    print('interrupted by keyboard, gracefully exiting')
                    return
            # Update clip_encoding per epoch if we are creating a story
            if self.create_story:
                self.clip_encoding = self.update_story_encoding(epoch, i)

        self.save_image(epoch, i) # one final save at end
