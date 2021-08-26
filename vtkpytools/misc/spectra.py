import numpy as np
import re
from typing import Tuple, Optional, Sequence, Dict
from pathlib import Path


def energySpectra1D(u: np.ndarray, L: float, periodic: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    r"""Calculate energy spectra for a 1D signal

    Given :math:`u_n,\ n=0\...N-1` samples of a signal, return the energy
    associated a frequency (bin) :math:`k`. This is done using a
    real-restricted FFT.

    .. math:: \int_0^\pi

    Parameters
    ----------
    u : np.ndarray
        Array of signal(s). If a 2D array is given, then the signals are
        assumed to be continuous in the last index. ie. ``u[0, :]`` is a vector
        of a continuous signal.
    L : float
        This is the length scale used to scale the frequency values
    periodic : bool
        Whether the continuous signal samples contain the periodic endpoint.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]

    """
    if periodic: u = u[..., 0:-1] # Remove periodic data point

    n = u.shape[-1]

    utrans = np.fft.rfft(u)/n
    energy = 0.5*(utrans*np.conj(utrans))

    ks = np.fft.rfftfreq(n, L)*2*np.pi

    # double the contributions of non-zero wavemodes
    # rfft gives only half of the spectrum (for real input, it is symmetric
    # about k_0)
    energy[1:] *= 2

    return ks, energy.astype(float)
