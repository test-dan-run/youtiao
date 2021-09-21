import torch
import random
from math import ceil

from torch_audiomentations.utils.fft import rfft, irfft
from torch_audiomentations.utils.io import Audio

def calculate_rms(samples):
    """
    Calculates the root mean square.
    Based on https://github.com/iver56/audiomentations/blob/master/audiomentations/core/utils.py
    """
    return torch.sqrt(torch.mean(torch.square(samples)))

def _gen_noise(f_decay, num_samples, sample_rate, device):
    """
    Generate colored noise with f_decay decay using torch.fft
    """
    noise = torch.normal(
        torch.tensor(0.0, device=device),
        torch.tensor(1.0, device=device),
        (sample_rate,),
        device=device,
    )

    f_decay = f_decay.to(device=device)

    spec = rfft(noise)
    mask = 1 / (
        torch.linspace(1, (sample_rate / 2) ** 0.5, spec.shape[0], device=device)
        ** f_decay
    )
    spec *= mask
    noise = Audio.rms_normalize(irfft(spec).unsqueeze(0)).squeeze()
    noise = torch.cat([noise] * int(ceil(num_samples / sample_rate)))
    return noise[:num_samples]

class AddColoredNoise(torch.nn.Module):
    """
    Add colored noises to the input audio.
    """

    def __init__(
        self,
        min_snr_in_db: float = 3.0,
        max_snr_in_db: float = 30.0,
        min_f_decay: float = -2.0,
        max_f_decay: float = 2.0,
        p: float = 0.5,
        sample_rate: int = 16000,
    ):
        """
        :param min_snr_in_db: minimum SNR in dB.
        :param max_snr_in_db: maximium SNR in dB.
        :param min_f_decay:
            defines the minimum frequency power decay (1/f**f_decay).
            Typical values are "white noise" (f_decay=0), "pink noise" (f_decay=1),
            "brown noise" (f_decay=2), "blue noise (f_decay=-1)" and "violet noise"
            (f_decay=-2)
        :param max_f_decay:
            defines the maximum power decay (1/f**f_decay) for non-white noises.
        :param p:
        :param sample_rate:
        """

        super(AddColoredNoise, self).__init__()

        self.min_snr_in_db = min_snr_in_db
        self.max_snr_in_db = max_snr_in_db
        if self.min_snr_in_db > self.max_snr_in_db:
            raise ValueError("min_snr_in_db must not be greater than max_snr_in_db")

        self.min_f_decay = min_f_decay
        self.max_f_decay = max_f_decay
        if self.min_f_decay > self.max_f_decay:
            raise ValueError("min_f_decay must not be greater than max_f_decay")

        self.p = p
        self.sample_rate = sample_rate
        self.transform_parameters = {}

    def randomize_parameters(self):

        # (batch_size, ) SNRs
        for param, mini, maxi in [
            ("snr_in_db", self.min_snr_in_db, self.max_snr_in_db),
            ("f_decay", self.min_f_decay, self.max_f_decay),
        ]:
            dist = torch.distributions.Uniform(
                low=torch.tensor(
                    mini, dtype=torch.float32
                ),
                high=torch.tensor(
                    maxi, dtype=torch.float32
                ),
                validate_args=True,
            )
            self.transform_parameters[param] = dist.sample()

    def forward(self, waveform: torch.Tensor):

        if random.random() <= self.p:

            _, num_samples = waveform.shape
            self.randomize_parameters()

            # (num_samples,)
            noise = _gen_noise(
                self.transform_parameters["f_decay"],
                num_samples,
                self.sample_rate,
                waveform.device)

            noise_rms = calculate_rms(waveform) / (
                10 ** (self.transform_parameters["snr_in_db"].to(waveform.device) / 20)
            )

            waveform = waveform + noise_rms * noise

        return waveform