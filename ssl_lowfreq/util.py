import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal
import random

def highpass_filter(data, freq, dt, pad=None):
    """Perform high-pass filter to a data
    Args:
        freq (float): The cut-off frequency (in Hz).
        data (array_like): Data to be filtered of shape (..., nt, :), where nt = number of time samples.
        dt (float): Time sampling rate of the data (in s).
        pad (array_like, optional): Padding to the data (before and after, with zeros) before filter. 
                                    Defaults to None (no padding). 
        See :func:'~torch.nn.functional.pad'.
    
    Returns:
        array_like: High-pass filtered data.
    """

    if pad is not None:
        data = F.pad(data, (0, 0, pad, pad), mode='replicate')

    sos = signal.butter(8,  freq / (0.5 * (1 / dt)), 'hp', output='sos')
    data = torch.tensor(signal.sosfiltfilt(sos, data.cpu().numpy(), axis=-2).copy(), dtype=torch.float32)
    if pad is not None:
        data = data[:, :, pad:-pad, :]

    return data.cuda()

def add_noise(data, noise_level):
    """Add noise to a data.
    Args:
        noise level
    
    Returns:
        array_like: Noisy data.
    """

    noise = 0.01 * noise_level * torch.std(data) * torch.randn_like(data, dtype=torch.float32)
    noisy = data + noise

    return noisy