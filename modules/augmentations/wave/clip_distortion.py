# torch implementation of https://github.com/iver56/audiomentations/blob/master/audiomentations/augmentations/transforms.py
import torch
import random

class ClipDistortion(torch.nn.Module):
    """
    Distort signal by clipping a random percentage of points
    The percentage of points that will be clipped is drawn from a uniform distribution between
    the two input parameters min_percentile_threshold and max_percentile_threshold. If for instance
    30% is drawn, the samples are clipped if they're below the 15th or above the 85th percentile.
    """

    def __init__(
        self,
        min_percent_threshold: int = 0,
        max_percent_threshold: int = 40,
        p: float = 0.5,
    ):
        """
        :param min_percentile_threshold: int, A lower bound on the total percent of samples that
            will be clipped
        :param max_percentile_threshold: int, A upper bound on the total percent of samples that
            will be clipped
        :param p: The probability of applying this transform
        """
        super(ClipDistortion, self).__init__()

        assert min_percent_threshold <= max_percent_threshold
        assert 0 <= min_percent_threshold <= 100
        assert 0 <= max_percent_threshold <= 100

        self.min_percent_threshold = min_percent_threshold
        self.max_percent_threshold = max_percent_threshold
        self.p = p
        self.transform_parameters = {}

    def randomize_parameters(self):

        self.transform_parameters['threshold_to_clip'] = random.randint(
            a = self.min_percent_threshold, 
            b = self.max_percent_threshold)
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if random.random() <= self.p:
            self.randomize_parameters()
            
            lower_percentile = (self.transform_parameters['threshold_to_clip'] / 2) / 100
            rnge = torch.Tensor([lower_percentile, 1.0 - lower_percentile])
            lower_threshold, upper_threshold = torch.quantile(waveform, rnge, dim=-1)
            waveform = torch.clip(waveform, lower_threshold.item(), upper_threshold.item())
        
        return waveform


    
        