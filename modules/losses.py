#
# Created on Tue Sep 21 2021
# Authored by Daniel Leong (test-dan-run)
#
import torch
from .metrics import SNR, SISNR

class SNRCost(SNR):
    '''
    The negative signal to noise ratio is calculated here. The loss is 
    always calculated over the last dimension. 
    '''
    def __init__(self) -> None:
        '''Initialise SNRCost loss module'''
        super(SNRCost, self).__init__()

    def forward(self, estimated: torch.Tensor, original: torch.Tensor) -> torch.Tensor:

        snr = self.calculate_snr(estimated, original)
        batched_loss = -snr
        loss = torch.mean(torch.flatten(batched_loss))

        return loss

class SISNRCost(SISNR):
    '''
    The negative scale-invariant signal to noise ratio is calculated here. The loss is 
    always calculated over the last dimension. 
    '''
    def __init__(self) -> None:
        '''Initialise SNRCost loss module'''
        super(SISNRCost, self).__init__()

    def forward(self, estimated: torch.Tensor, original: torch.Tensor) -> torch.Tensor:

        snr = self.calculate_snr(estimated, original)
        batched_loss = -snr
        loss = torch.mean(torch.flatten(batched_loss))

        return loss

def objective_select(obj_key):

    objectives = {
        'snr': SNRCost,
        'si_snr': SISNRCost
    }

    return objectives[obj_key]()