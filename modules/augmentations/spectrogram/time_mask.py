from torchaudio.transforms import TimeMasking
import torch
from typing import Optional

class CustomTimeMasking(torch.nn.Module):

    def __init__(
            self,
            time_mask_param: Optional[int] = 70,
            num_time_masks: Optional[int] = 1) -> None:
    
        super(CustomTimeMasking, self).__init__()

        self.mask = TimeMasking(time_mask_param)
        self.num_time_masks = num_time_masks

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        for _ in range(self.num_time_masks):
            spectrogram = self.mask(spectrogram)

        return spectrogram
