from .freq_mask import CustomFrequencyMasking as FrequencyMasking
from .time_mask import CustomTimeMasking as TimeMasking
from torchaudio.transforms import TimeStretch

from typing import List, Dict, Any

def spectrogram_augment_select(spectrogram_augment_list: List[str], augment_configs: Dict[str, Any]):
    
    transformation_list = []

    augmentations = {
        'freq_mask': FrequencyMasking,
        'time_mask': TimeMasking,
        'time_stretch': TimeStretch
    }

    for aug_name in spectrogram_augment_list:

        transformation = augmentations[aug_name](**augment_configs[aug_name])
        transformation_list.append(transformation)

    return transformation_list