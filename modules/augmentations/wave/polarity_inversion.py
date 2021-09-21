# torch implementation of https://github.com/iver56/audiomentations/blob/master/audiomentations/augmentations/transforms.py
import torch
import random

class PolarityInversion(torch.nn.Module):
    """
    Flip the audio samples upside-down, reversing their polarity. In other words, multiply the
    waveform by -1, so negative values become positive, and vice versa. The result will sound
    the same compared to the original when played back in isolation. However, when mixed with
    other audio sources, the result may be different. This waveform inversion technique
    is sometimes used for audio cancellation or obtaining the difference between two waveforms.
    However, in the context of audio data augmentation, this transform can be useful when
    training phase-aware machine learning models.
    """

    def __init__(
        self,
        p: float = 0.5
    ):
        """
        :param p: The probability of applying this transform
        """

        super(PolarityInversion, self).__init__()
        self.p = p

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if random.random() <= self.p:
            waveform = -waveform
        return waveform