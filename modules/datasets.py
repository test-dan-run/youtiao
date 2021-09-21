#
# Created on Tue Sep 21 2021
# Authored by Daniel Leong (test-dan-run)
#
import os
import torch
import random
import torchaudio
from typing import Optional, List

class AudioDataset(torch.utils.data.Dataset):
    '''
    A Dataset class that takes in WAV directory and outputs batches of audio tensors

    Parameters
    ----------
    wav_dir: str
        Path to directory of WAV files (audio files must be mono-channeled)
    sample_rate: int
        Sample rate of audio files
    segment_length_seconds: int
        Length of the output audio tensors, in seconds
    slice_samples: bool (default=False)
        Put True if original audio files are longer that <segment_length_seconds>
        Audio tensors will be randomly sliced to obtain the target output tensor length
    augmentations: List (Optional)
        list of augmentations to be applied to form the output audio tensors
    random_state: int (Optional)

    Outputs
    ---------
    sliced_signal: torch.Tensor
        Output audio tensor to serve as model inputs
    filename: str
        Corresponding filename of the original audio tensor
    '''

    def __init__(
        self,
        wav_dir: str,
        sample_rate: int,
        segment_length_seconds: int,
        slice_samples: bool = False,
        augmentations: Optional[List] = [],
        random_state: Optional[int] = 42,
    ):
        # initialise required variables
        self.segment_length = int(sample_rate * segment_length_seconds)
        self.slice_samples = slice_samples
        self.augmentations = augmentations
        self.data =  [os.path.join(wav_dir, filename) for filename in os.listdir(wav_dir)]

        if self.slice_samples:
            random.seed(random_state)

    def __getitem__(self, idx):
        
        # load audio
        wav_path = self.data[idx]
        filename = os.path.basename(wav_path)
        input_signal, _ = torchaudio.load(wav_path, normalize=True)

        if not self.slice_samples:
            assert input_signal.size(1) == self.segment_length, f'input audio ({input_signal.size(1)}) is not of required sample length ({self.segment_length})'

        else:
            # slice audio to required segment length
            if input_signal.size(1) >= self.segment_length:
                max_signal_start = input_signal.size(1) - self.segment_length
                signal_start = random.randint(0, max_signal_start)

                sliced_signal = input_signal[:, signal_start:signal_start+self.segment_length]
            else:
                sliced_signal = torch.nn.functional.pad(input_signal, (0, self.segment_length - input_signal.size(1)), 'constant')

        # augment
        if self.augmentations:
            for augment in self.augmentations:
                sliced_signal = augment(sliced_signal)

        return sliced_signal, filename

    def __len__(self):
        return len(self.data)

class FauxPairAudioDataset(torch.utils.data.Dataset):
    '''
    A Dataset class that takes in WAV directory and outputs a pair of audio tensors
    One is a self-augmented signal, while the other is the clean input signal
    Used for noise removal/separation models

    Parameters
    ----------
    wav_dir: str
        Path to directory of WAV files (audio files must be mono-channeled)
    sample_rate: int
        Sample rate of audio files
    segment_length_seconds: int
        Length of the output audio tensors, in seconds
    slice_samples: bool (default=False)
        Put True if original audio files are longer that <segment_length_seconds>
        Audio tensors will be randomly sliced to obtain the target output tensor length
    augmentations: List (Optional)
        list of augmentations to be applied to form the output audio tensors
    random_state: int (Optional)

    Outputs
    ---------
    noisy_signal: torch.Tensor
        Audio tensor to serve as the noisy input 
    sliced_signal: torch.Tensor
        Audio tensor to serve as the clean target input
    '''

    def __init__(
        self,
        wav_dir: str,
        sample_rate: int,
        segment_length_seconds: int,
        augmentations: Optional[List] = [],
        random_state: Optional[int] = 42,
    ):
        # set random seed
        random.seed(random_state)

        # initialise required variables
        self.segment_length = int(sample_rate * segment_length_seconds)
        self.augmentations = augmentations
        self.data =  [os.path.join(wav_dir, filename) for filename in os.listdir(wav_dir)]

    def __getitem__(self, idx):
        
        # load audio
        wav_path = self.data[idx]
        input_signal, _ = torchaudio.load(wav_path, normalize=True)

        # slice audio to required segment length
        if input_signal.size(1) >= self.segment_length:
            max_signal_start = input_signal.size(1) - self.segment_length
            signal_start = random.randint(0, max_signal_start)

            sliced_signal = input_signal[:, signal_start:signal_start+self.segment_length]
        else:
            sliced_signal = torch.nn.functional.pad(input_signal, (0, self.segment_length - input_signal.size(1)), 'constant')

        # cloned signal will be augmented with noise
        noisy_signal = torch.clone(sliced_signal)

        if self.augmentations:
            for augment in self.augmentations:
                noisy_signal = augment(noisy_signal)

        return noisy_signal, sliced_signal 

    def __len__(self):
        return len(self.data)