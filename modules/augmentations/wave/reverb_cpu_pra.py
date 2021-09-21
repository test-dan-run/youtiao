import random
from typing import List

import torch
from torch_audiomentations.utils.io import Audio

import numpy as np
import pyroomacoustics as pra
from pyroomacoustics.parameters import constants

EPS = 1e-4

class Reverb_CPU(torch.nn.Module):
    def __init__(
        self,
        p: float = 0.5,
        room_ranges: List[List[float]] = [[5.0,10.0], [5.0,10.0], [3.0,4.0]],
        mic_loc_ranges: List[List[float]] = [[3.25,6.25], [3.25,6.25],[0.9, 1.8]],
        src_loc_ranges: List[List[float]] = [[5.0,7.0], [5.0,7.0], [0.9,1.8]], 
        t60_range: List[float] = [0.1, 1.0],
        sample_rate: int = 16000,
        *args,
        **kwargs,
    ):
        """
        :param local_path: Either a path to a folder with audio files or a list of paths
            to audio files.


        :param p:
        :param sample_rate:
        """
        super(Reverb_CPU, self).__init__()

        self.param_ranges = {
            'roomX': room_ranges[0],
            'roomY': room_ranges[1],
            'roomZ': room_ranges[2],

            'mic1X': mic_loc_ranges[0],
            'mic1Y': mic_loc_ranges[1],
            'mic1Z': mic_loc_ranges[2],

            'src1X': src_loc_ranges[0],
            'src1Y': src_loc_ranges[1],
            'src1Z': src_loc_ranges[2],

            't60': t60_range
        }

        self.p = p
        self.sample_rate = sample_rate
        if sample_rate is not None:
            self.audio = Audio(sample_rate=sample_rate, mono=True)

        self.transform_parameters = {}

    def randomize_parameters(self) -> None:
        
        self.transform_parameters = {k:random.uniform(v[0], v[1]) for k,v in self.param_ranges.items()}

        self.transform_parameters['mic1X'] = min(self.transform_parameters['mic1X'], self.transform_parameters['roomX']) - EPS
        self.transform_parameters['mic1Y'] = min(self.transform_parameters['mic1Y'], self.transform_parameters['roomY']) - EPS
        self.transform_parameters['src1X'] = min(self.transform_parameters['src1X'], self.transform_parameters['roomX']) - EPS
        self.transform_parameters['src1Y'] = min(self.transform_parameters['src1Y'], self.transform_parameters['roomY']) - EPS

    def forward(self, waveform: torch.Tensor, anechoic: bool = False) -> torch.Tensor:

        if random.random() <= self.p:

            self.randomize_parameters()
            d = self.transform_parameters
            p = [d['roomX'], d['roomY'], d['roomZ']]
            s1 = np.array([d['src1X'], d['src1Y'], d['src1Z']])
            mic1 = np.array([[d['mic1X']], [d['mic1Y']], [d['mic1Z']]])

            # initialise room parameters
            volume = d['roomX'] * d['roomY'] * d['roomZ']
            surface_area = 2*(d['roomX']*d['roomY'] + d['roomX']*d['roomZ'] + d['roomY']*d['roomZ'])
            absorption = 24 * volume * np.log(10.0) / (constants.get('c') * surface_area * d['t60'])

            # minimum max order to guarantee complete filter of length T60
            max_order = np.ceil(d['t60'] * constants.get('c') / min(p)).astype(int)

            # create room
            room = pra.room.ShoeBox(p, fs=self.sample_rate, t0=0., absorption=absorption, max_order=max_order, sigma2_awgn=None)
            room.add_source(s1, signal=waveform[0, :])
            room.add_microphone_array(pra.MicrophoneArray(mic1, room.fs))

            room.compute_rir()
            room.simulate()
            audio_array = room.mic_array.signals[0,:]

            waveform = torch.Tensor(audio_array[:waveform.shape[1]]).to(waveform.device)

            waveform = waveform.unsqueeze(dim=0)
            
        return waveform