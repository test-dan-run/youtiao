#
# Created on Tue Sep 21 2021
# Authored by Daniel Leong (test-dan-run)
#
import torch

EPS = 1e-8

class SNR(torch.nn.Module):
    '''
    Calculate signal-to-noise ratio
    '''
    def __init__(self) -> None:
        '''Initialise SNR module'''
        super(SNR, self).__init__()

    def calculate_snr(self, estimated: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        
        # calculate SNR
        target = torch.mean(torch.square(original), dim=-1, keepdim=True)
        noise = torch.mean(torch.square(estimated - target), dim=-1, keepdim=True) + EPS

        snr = 10 * torch.log10(target/noise)

        return snr

    def forward(self, estimated: torch.Tensor, original: torch.Tensor) -> torch.Tensor:

        return self.calculate_snr(estimated, original)

class SISNR(torch.nn.Module):
    '''
    Calculate scale-invariant signal-to-noise ratio
    Reference: http://www.jonathanleroux.org/pdf/LeRoux2019ICASSP05sdr.pdf
    '''
    def __init__(self) -> None:
        '''Initialise SNR module'''
        super(SISNR, self).__init__()
    
    @staticmethod
    def pow_norm(s1: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
        return torch.sum(s1 * s2, dim=-1, keepdim=True)

    @staticmethod
    def pow_p_norm(signal: torch.Tensor) -> torch.Tensor:
        '''Compute 2 Norm'''
        return torch.pow(torch.norm(signal, p=2, dim=-1, keepdim=True), 2)

    def calculate_snr(self, estimated:torch.Tensor, original: torch.Tensor) -> torch.Tensor:

        # calculate SI-SNR
        target = self.pow_norm(estimated, original) * original / (self.pow_p_norm(original) + EPS)
        noise = estimated - target

        snr = 10 * torch.log10(self.pow_p_norm(target) / (self.pow_p_norm(noise) + EPS) + EPS)

        return snr
        
    def forward(self, estimated: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        
        return self.calculate_snr(estimated, original)