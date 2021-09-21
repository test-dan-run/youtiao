from torchaudio.transforms import FrequencyMasking
import torch
from typing import Optional

class CustomFrequencyMasking(torch.nn.Module):

    def __init__(
            self,
            freq_mask_param: Optional[int] = 70,
            num_freq_masks: Optional[int] = 1) -> None:
    
        super(CustomFrequencyMasking, self).__init__()

        self.mask = FrequencyMasking(freq_mask_param)
        self.num_freq_masks = num_freq_masks

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        for _ in range(self.num_freq_masks):
            spectrogram = self.mask(spectrogram)
        
        return spectrogram