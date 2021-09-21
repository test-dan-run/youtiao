import random
import torch
from typing import Optional


def calculate_rms(samples):
    """
    Calculates the root mean square.
    Based on https://github.com/iver56/audiomentations/blob/master/audiomentations/core/utils.py
    """
    return torch.sqrt(torch.mean(torch.square(samples)))

class ReverseOverlay(torch.nn.Module):
    def __init__(
        self, 
        p: Optional[float] = 0.5,
        sample_rate: Optional[int] = 16000, 
        min_snr_in_db: Optional[int] = 3, 
        max_snr_in_db: Optional[int] = 30) -> None:
        
        super(ReverseOverlay, self).__init__()

        self.sample_rate = sample_rate
        self.min_snr_in_db = min_snr_in_db
        self.max_snr_in_db = max_snr_in_db
        self.p = p

        self.transform_parameters = {}

    def randomize_params(self) -> None:

        self.transform_parameters['snr_in_db'] = random.uniform(self.min_snr_in_db, self.max_snr_in_db)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:

        if random.random() >= self.p:
            return waveform

        self.randomize_params()

        if waveform.dim() < 2:
            waveform = waveform.unsqueeze(0)
            
        # add noise to target sample based on randomized snr 
        noise_rms = calculate_rms(waveform) / (
            10 ** (self.transform_parameters["snr_in_db"] / 20)
        )

        reversed_waveform = waveform.flip(dims=(1,))
        waveform = waveform + reversed_waveform * noise_rms

        return waveform