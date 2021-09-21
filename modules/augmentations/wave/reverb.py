from torchaudio import sox_effects
import random

import torch

class Reverb(torch.nn.Module):
    def __init__(
        self,
        min_reverb: int = 0,
        max_reverb: int = 100,
        min_damp_factor: int = 0,
        max_damp_factor: int = 100,
        min_room_size: int = 0,
        max_room_size: int = 100,
        p: float = 0.5,
        sample_rate: int = 16000
    ):
        """
        :param p: The probability of applying this transform
        """

        super(Reverb, self).__init__()
        self.sample_rate = sample_rate
        self.min_reverb = min_reverb
        self.max_reverb = max_reverb
        self.min_damp_factor = min_damp_factor
        self.max_damp_factor = max_damp_factor
        self.min_room_size = min_room_size
        self.max_room_size = max_room_size
        self.p = p

        self.transform_parameters = {}

    def randomize_params(self) -> None:

        self.transform_parameters['reverb'] = random.uniform(self.min_reverb, self.max_reverb)
        self.transform_parameters['damp_factor'] = random.uniform(self.min_damp_factor, self.max_damp_factor)
        self.transform_parameters['room_size'] = random.uniform(self.min_room_size, self.max_room_size)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if random.random() <= self.p:
            self.randomize_params()

            effect = [
                ['reverb', str(self.transform_parameters['reverb']), str(self.transform_parameters['damp_factor']), str(self.transform_parameters['room_size'])],
                ['channels', '1']]

            waveform, _ = sox_effects.apply_effects_tensor(waveform, self.sample_rate, effect)

        return waveform