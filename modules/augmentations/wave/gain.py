# torch implementation of https://github.com/iver56/audiomentations/blob/master/audiomentations/augmentations/transforms.py
import torch
import random

class Gain(torch.nn.Module):
    """
    Multiply the audio by a random amplitude factor to reduce or increase the volume. This
    technique can help a model become somewhat invariant to the overall gain of the input audio.
    Warning: This transform can return samples outside the [-1, 1] range, which may lead to
    clipping or wrap distortion, depending on what you do with the audio in a later stage.
    See also https://en.wikipedia.org/wiki/Clipping_(audio)#Digital_clipping
    """
    
    def __init__(
        self,
        min_gain_db: int = -12,
        max_gain_db: int = 12,
        p: float = 0.5,
    ):
        super(Gain, self).__init__()

        assert min_gain_db <= max_gain_db
        self.min_gain_db = min_gain_db
        self.max_gain_db = max_gain_db
        self.p = p
        self.transform_parameters = {}

    def randomize_parameters(self):
        
        random_gain = random.uniform(self.min_gain_db, self.max_gain_db)
        self.transform_parameters['amplitude_ratio'] = 10 ** (random_gain / 20)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if random.random() <= self.p:
            self.randomize_parameters()
            waveform = waveform * self.transform_parameters['amplitude_ratio']
        
        return waveform
