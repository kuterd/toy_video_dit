import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from einops import rearrange, repeat
from torch import nn, Tensor, optim
import torch

from IPython.display import display, HTML

import io
import base64

def corrupt(x, amount):
    noise = torch.rand_like(x)
    amount = amount.view(-1, 1, 1, 1, 1)
    # Noise interpolation.
    return x * (1 - amount) + noise * amount, noise

def patchify(sample, patch_w, patch_h):
    prefix = "b" if len(sample.shape) == 5 else "" 
    return rearrange(sample, f"{prefix} t c (h ph) (w pw) -> {prefix} (t h w) (ph pw c)", pw = patch_w, ph = patch_h)

def unpatchify(tokens, patch_w, patch_h, width, height, channels):
    prefix = "b" if len(tokens.shape) == 3 else "" 
    
    h_tokens = height // patch_h
    w_tokens = width // patch_w
    
    return rearrange(tokens, f"{prefix} (t h w) (ph pw c) -> {prefix} t c (h ph) (w pw)", c = channels, h = h_tokens, w = w_tokens, pw = patch_w, ph = patch_h)

def video_to_gif(video_frames, duration=100):
    pil_frames = [Image.fromarray(frame) for frame in video_frames]
    gif_io = io.BytesIO()
    pil_frames[0].save(gif_io, format='GIF', append_images=pil_frames[1:], save_all=True, duration=duration, loop=0)
    gif_io.seek(0)
    return gif_io

def create_grid_gif(videos, grid_shape=(2, 4), duration=100):
    b, t, c, w, h = videos.shape
    grid_w = w * grid_shape[1]
    grid_h = h * grid_shape[0]

    combined_frames = []

    for frame_idx in range(t):
        grid_image = Image.new('RGB', (grid_w, grid_h))

        for i in range(min(b, grid_shape[0] * grid_shape[1])):
            row, col = divmod(i, grid_shape[1])
            video_frames = videos[i]

            video_frame = video_frames[frame_idx]
            gif_frame = Image.fromarray(video_frame[0])
            grid_image.paste(gif_frame, (col * w, row * h))

        combined_frames.append(grid_image)

    gif_io = io.BytesIO()
    combined_frames[0].save(gif_io, format='GIF', append_images=combined_frames[1:], save_all=True, duration=duration, loop=0)
    gif_io.seek(0)
    
    return gif_io

def display_video_grid(videos, grid_shape):
    grid_gif_io = create_grid_gif(videos, grid_shape=grid_shape)
    grid_gif_base64 = base64.b64encode(grid_gif_io.getvalue()).decode('utf-8')
    
    html = f'<img src="data:image/gif;base64,{grid_gif_base64}" />'
    
    display(HTML(html))

def generate_ids(batch_size, frames, patch_w, patch_h, width, height, device):
    x_ids = torch.zeros(frames, width // patch_w, height // patch_h, 4, device=device)
    x_ids[...,0] = x_ids[...,0] + torch.arange(0, width // patch_w, device=device)[None, :, None]
    x_ids[...,1] = x_ids[...,1] + torch.arange(0, height // patch_h, device=device)[None, None, :]
    x_ids[...,2] = x_ids[...,2] + torch.arange(0, frames, device=device)[:, None, None]
    return repeat(x_ids, "t h w c -> b (t h w) c", b=batch_size)

import numpy as np

class Schedule:
    def __init__(self, sigmas: torch.FloatTensor):
        self.sigmas = sigmas
    def __getitem__(self, i) -> torch.FloatTensor:
        return self.sigmas[i]
    def __len__(self) -> int:
        return len(self.sigmas)
    def sample_batch(self, x0: torch.FloatTensor) -> torch.FloatTensor:
        return self[torch.randint(len(self), (x0.shape[0],))].to(x0)
    def sample_sigmas(self, steps: int) -> torch.FloatTensor:
        indices = list((len(self) * (1 - np.arange(0, steps)/steps))
                       .round().astype(np.int64) - 1)
        return self[indices + [0]]

class ScheduleLinear(Schedule):
    def __init__(self, N: int, sigma_min: float=0.001, sigma_max: float=1):
        super().__init__(torch.linspace(sigma_min, sigma_max, N))
