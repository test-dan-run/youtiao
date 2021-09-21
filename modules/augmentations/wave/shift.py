import torch
import typing
import random

class Shift(torch.nn.Module):
    """
    Shift the audio forwards or backwards, with or without rollover
    """

    def __init__(
        self,
        min_shift: float = -0.1,
        max_shift: float = 0.1,
        shift_unit: str = "fraction",
        rollover: bool = False,
        p: float = 0.5,
        sample_rate: typing.Optional[int] = 16000,
    ):
        """
        :param min_shift: minimum amount of shifting in time. See also shift_unit.
        :param max_shift: maximum amount of shifting in time. See also shift_unit.
        :param shift_unit: Defines the unit of the value of min_shift and max_shift.
            "fraction": Fraction of the total sound length
            "samples": Number of audio samples
            "seconds": Number of seconds
        :param rollover: When set to True, samples that roll beyond the first or last position
            are re-introduced at the last or first. When set to False, samples that roll beyond
            the first or last position are discarded. In other words, rollover=False results in
            an empty space (with zeroes).
        :param mode:
        :param p:
        :param p_mode:
        :param sample_rate:
        """
        super(Shift, self).__init__()
        self.min_shift = min_shift
        self.max_shift = max_shift
        self.shift_unit = shift_unit
        self.rollover = rollover
        self.p = p
        self.sample_rate = sample_rate
        self.transform_parameters = {}

        if self.min_shift > self.max_shift:
            raise ValueError("min_shift must not be greater than max_shift")
        if self.shift_unit not in ("fraction", "samples", "seconds"):
            raise ValueError('shift_unit must be "samples", "fraction" or "seconds"')

    def randomize_parameters(self, waveform):
        if self.shift_unit == "samples":
            min_shift_in_samples = self.min_shift
            max_shift_in_samples = self.max_shift
        elif self.shift_unit == "fraction":
            min_shift_in_samples = int(round(self.min_shift * waveform.shape[-1]))
            max_shift_in_samples = int(round(self.max_shift * waveform.shape[-1]))
        elif self.shift_unit == "seconds":
            min_shift_in_samples = int(round(self.min_shift * self.sample_rate))
            max_shift_in_samples = int(round(self.max_shift * self.sample_rate))
        else:
            raise ValueError("Invalid shift_unit")

        assert (
            torch.iinfo(torch.int32).min
            <= min_shift_in_samples
            <= torch.iinfo(torch.int32).max
        )
        assert (
            torch.iinfo(torch.int32).min
            <= max_shift_in_samples
            <= torch.iinfo(torch.int32).max
        )

        if min_shift_in_samples == max_shift_in_samples:
            self.transform_parameters['num_samples_to_shift'] = min_shift_in_samples
        else:
            self.transform_parameters['num_samples_to_shift'] = random.randint(min_shift_in_samples, max_shift_in_samples)

    def forward(
        self, waveform: torch.Tensor) -> torch.Tensor:

        if random.random() <= self.p:
            self.randomize_parameters(waveform)
            num_shift = self.transform_parameters['num_samples_to_shift']
            waveform = torch.roll(
                waveform, shifts=num_shift, dims=-1
            )

            if not self.rollover:
                if num_shift > 0:
                    waveform[..., :num_shift] = 0.0
                elif num_shift < 0:
                    waveform[..., num_shift:] = 0.0

        return waveform