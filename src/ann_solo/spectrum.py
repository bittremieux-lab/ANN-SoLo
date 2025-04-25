import functools
import math
from typing import Optional

import mmh3
import numba as nb
import numpy as np
from spectrum_utils.spectrum import MsmsSpectrum
from ann_solo.config import config


@nb.njit
def _check_spectrum_valid(spectrum_mz: np.ndarray, min_peaks: int,
                          min_mz_range: float) -> bool:
    """
    Check whether a spectrum is of high enough quality to be used for matching.

    Parameters
    ----------
    spectrum_mz : np.ndarray
        M/z peaks of the spectrum whose quality is checked.
    min_peaks : int
        The minimum number of peaks for a spectrum to be valid.
    min_mz_range : float
        The minimum mass range (m/z difference between the highest and lowest
        peak) for a spectrum to be valid.

    Returns
    -------
    bool
        True if the spectrum has enough peaks covering a wide enough mass
        range, False otherwise.
    """
    return (len(spectrum_mz) >= min_peaks and
            spectrum_mz[-1] - spectrum_mz[0] >= min_mz_range)


@nb.njit
def _norm_intensity(spectrum_intensity: np.ndarray) -> np.ndarray:
    """
    Normalize spectrum peak intensities.

    Parameters
    ----------
    spectrum_intensity : np.ndarray
        The spectrum peak intensities to be normalized.

    Returns
    -------
    np.ndarray
        The normalized peak intensities.
    """
    return spectrum_intensity / np.linalg.norm(spectrum_intensity)


def process_spectrum(spectrum: MsmsSpectrum, is_library: bool) -> MsmsSpectrum:
    """
    Process the peaks of the MS/MS spectrum according to the config.

    Parameters
    ----------
    spectrum : MsmsSpectrum
        The spectrum that will be processed.
    is_library : bool
        Flag specifying whether the spectrum is a query spectrum or a library
        spectrum.

    Returns
    -------
    MsmsSpectrum
        The processed spectrum. The spectrum is also changed in-place.
    """
    if spectrum.is_processed:
        return spectrum

    min_peaks = config.min_peaks
    min_mz_range = config.min_mz_range
    spectrum = spectrum.set_mz_range(config.min_mz, config.max_mz)
    if not _check_spectrum_valid(spectrum.mz, min_peaks, min_mz_range):
        spectrum.is_valid = False
        spectrum.is_processed = True
        return spectrum
    if config.resolution is not None:
        spectrum = spectrum.round(config.resolution, 'sum')
        if not _check_spectrum_valid(spectrum.mz, min_peaks, min_mz_range):
            spectrum.is_valid = False
            spectrum.is_processed = True
            return spectrum
    if config.remove_precursor:
        spectrum = spectrum.remove_precursor_peak(
            config.remove_precursor_tolerance, 'Da', 2)
        if not _check_spectrum_valid(spectrum.mz, min_peaks, min_mz_range):
            spectrum.is_valid = False
            spectrum.is_processed = True
            return spectrum
    spectrum = spectrum.filter_intensity(
        config.min_intensity, (config.max_peaks_used_library if is_library else
                               config.max_peaks_used))
    if not _check_spectrum_valid(spectrum.mz, min_peaks, min_mz_range):
        spectrum.is_valid = False
        spectrum.is_processed = True
        return spectrum
    scaling = config.scaling
    if scaling == 'sqrt':
        scaling = 'root'
    if scaling is not None:
        spectrum = spectrum.scale_intensity(
            scaling, max_rank=(config.max_peaks_used_library if is_library else
                               config.max_peaks_used))

    spectrum._inner._intensity = _norm_intensity(spectrum.intensity)

    # Set a flag to indicate that the spectrum has been processed to avoid
    # reprocessing of library spectra for multiple queries.
    spectrum.is_valid = True
    spectrum.is_processed = True

    return spectrum




@functools.lru_cache(maxsize=None)
def get_dim(min_mz, max_mz, bin_size):
    """
    Compute the number of bins over the given mass range for the given bin
    size.

    Args:
        min_mz: The minimum mass in the mass range (inclusive).
        max_mz: The maximum mass in the mass range (inclusive).
        bin_size: The bin size (in Da).

    Returns:
        A tuple containing (i) the number of bins over the given mass range for
        the given bin size, (ii) the highest multiple of bin size lower than
        the minimum mass, (iii) the lowest multiple of the bin size greater
        than the maximum mass. These two final values are the true boundaries
        of the mass range (inclusive min, exclusive max).
    """
    min_mz, max_mz = float(min_mz), float(max_mz)
    start_dim = min_mz - min_mz % bin_size
    end_dim = max_mz + bin_size - max_mz % bin_size
    return round((end_dim - start_dim) / bin_size), start_dim, end_dim


@functools.lru_cache(maxsize=None)
def hash_idx(bin_idx: int, hash_len: int) -> int:
    """
    Hash an integer index to fall between 0 and the given maximum hash index.

    Parameters
    ----------
    bin_idx : int
        The (unbounded) index to be hashed.
    hash_len : int
        The maximum index after hashing.

    Returns
    -------
    int
        The hashed index between 0 and `hash_len`.
    """
    return mmh3.hash(str(bin_idx), 42, signed=False) % hash_len


def _spectrum_to_vector(spectrum: MsmsSpectrum, min_mz: float, max_mz: float,
                       bin_size: float, hash_len: int, norm: bool = True,
                       vector: np.ndarray = None) -> np.ndarray:
    """
    Convert a `Spectrum` to a dense NumPy vector.

    Peaks are first discretized in to mass bins of width `bin_size` between
    `min_mz` and `max_mz`, after which they are hashed to random hash bins
    in the final vector.

    Parameters
    ----------
    spectrum : Spectrum
        The `Spectrum` to be converted to a vector.
    min_mz : float
        The minimum m/z to include in the vector.
    max_mz : float
        The maximum m/z to include in the vector.
    bin_size : float
        The bin size in m/z used to divide the m/z range.
    hash_len : int
        The length of the hashed vector, None if no hashing is to be done.
    norm : bool
        Normalize the vector to unit length or not.
    vector : np.ndarray, optional
        A pre-allocated vector.

    Returns
    -------
    np.ndarray
        The hashed spectrum vector with unit length.
    """
    vec_len, min_bound, _ = get_dim(min_mz, max_mz, bin_size)
    if vector is None:
        if hash_len is not None:
            vec_len = hash_len
        vector = np.zeros((vec_len,), np.float32)
        if vec_len != vector.shape[0]:
            raise ValueError('Incorrect vector dimensionality')

    for mz, intensity in zip(spectrum.mz, spectrum.intensity):
        bin_idx = math.floor((mz - min_bound) // bin_size)
        if hash_len is not None:
            bin_idx = hash_idx(bin_idx, hash_len)
        vector[bin_idx] += intensity

    if norm:
        vector /= np.linalg.norm(vector)
    return vector


def _spec_to_neutral_loss(spectrum: MsmsSpectrum) -> MsmsSpectrum:
    """
    Convert a spectrum to a neutral loss spectrum by subtracting the peak m/z
    values from the precursor m/z.

    Parameters
    ----------
    spectrum : sus.MsmsSpectrum
        The spectrum to be converted to its neutral loss spectrum.

    Returns
    -------
    sus.MsmsSpectrum
        The converted neutral loss spectrum.
    """
    # Add ghost peak at 0 m/z to anchor the m/z range after transformation.
    mz, intensity = np.copy(spectrum.mz), np.copy(spectrum.intensity)
    mz, intensity = np.insert(mz, 0, [0]), np.insert(intensity, 0, [0])
    # Create neutral loss peaks and make sure the peaks are in ascending m/z
    # order.
    # TODO: This assumes [M+H]x charged ions.
    adduct_mass = 1.007825
    neutral_mass = (
        spectrum.precursor_mz - adduct_mass
    ) * spectrum.precursor_charge
    mz, intensity = ((neutral_mass + adduct_mass) - mz)[::-1], intensity[::-1]
    return MsmsSpectrum(
        spectrum.identifier,
        spectrum.precursor_mz,
        spectrum.precursor_charge,
        np.ascontiguousarray(mz),
        np.ascontiguousarray(intensity),
        spectrum.retention_time,
    )

def spectrum_to_vector(spectrum: MsmsSpectrum, min_mz: float, max_mz: float,
                       bin_size: float, hash_len: int,
                       include_nl: bool = True) -> np.ndarray:
    """
    Convert a `Spectrum` to a dense NumPy vector.

    Peaks are first discretized in to mass bins of width `bin_size` between
    `min_mz` and `max_mz`, after which they are hashed to random hash bins
    in the final vector.

    Parameters
    ----------
    spectrum : Spectrum
        The `Spectrum` to be converted to a vector.
    min_mz : float
        The minimum m/z to include in the vector.
    max_mz : float
        The maximum m/z to include in the vector.
    bin_size : float
        The bin size in m/z used to divide the m/z range.
    hash_len : int
        The length of the hashed vector, None if no hashing is to be done.
    include_neutral_loss : bool
        Concatinate the spectrum neutral loss projection.
    Returns
    -------
    np.ndarray
        The hashed spectrum vector. Optionally concatinated hashed spectrum
        vector.
    """
    if include_nl:
        vector = np.concatenate((
            _spectrum_to_vector(
                        spectrum,
                        min_mz,
                        max_mz,
                        bin_size,
                        hash_len,
                        norm=True,
                    ),
            _spectrum_to_vector(
                        _spec_to_neutral_loss(spectrum),
                        min_mz,
                        max_mz,
                        bin_size,
                        hash_len,
                        norm=True,
                    )
        ))
    else:
        vector = _spectrum_to_vector(
                        spectrum,
                        min_mz,
                        max_mz,
                        bin_size,
                        hash_len,
                        norm=True,
                    )
    return vector



class SpectrumSpectrumMatch:

    def __init__(
            self,
            query_spectrum: MsmsSpectrum,
            library_spectrum: MsmsSpectrum = None,
            peak_matches: Optional[np.ndarray] = None,
            search_engine_score: float = math.nan,
            q: float = math.nan,
    ):
        self.query_spectrum = query_spectrum
        self.library_spectrum = library_spectrum
        self.peak_matches = peak_matches
        self.search_engine_score = search_engine_score
        self.q = q

    @property
    def sequence(self):
        return (self.library_spectrum.peptide
                if self.library_spectrum is not None else None)

    @property
    def query_identifier(self):
        return self.query_spectrum.identifier

    @property
    def query_index(self):
        return self.query_spectrum.index

    @property
    def library_identifier(self):
        return (self.library_spectrum.identifier
                if self.library_spectrum is not None else None)

    @property
    def retention_time(self):
        return self.query_spectrum.retention_time

    @property
    def charge(self):
        return self.query_spectrum.precursor_charge

    @property
    def exp_mass_to_charge(self):
        return self.query_spectrum.precursor_mz

    @property
    def calc_mass_to_charge(self):
        return (self.library_spectrum.precursor_mz
                if self.library_spectrum is not None else None)

    @property
    def is_decoy(self):
        return (self.library_spectrum.is_decoy
                if self.library_spectrum is not None else None)
