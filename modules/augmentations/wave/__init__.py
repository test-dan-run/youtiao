from .background_noise import AddBackgroundNoise
from .clip_distortion import ClipDistortion
from .colored_noise import AddColoredNoise
from .gain import Gain
from .impulse_response import ApplyImpulseResponse
from .low_pass import LowPassFilter
from .polarity_inversion import PolarityInversion
from .reverse_overlay import ReverseOverlay
from .shift import Shift
from .reverb import Reverb

from typing import List, Dict, Any

def wave_augment_select(wave_augment_list: List[str], augment_configs: Dict[str, Any]):
    
    transformation_list = []

    augmentations = {
        'background_noise': AddBackgroundNoise,
        'clip_distortion': ClipDistortion,
        'impulse_response': ApplyImpulseResponse,
        'gain': Gain,
        'polarity_inversion': PolarityInversion,
        'low_pass': LowPassFilter,
        'colored_noise': AddColoredNoise,          
        'reverse_overlay': ReverseOverlay,
        'shift': Shift,
        'reverb': Reverb
    }

    for aug_name in wave_augment_list:
        transformation = augmentations[aug_name](**augment_configs[aug_name])
        transformation_list.append(transformation)

    return transformation_list