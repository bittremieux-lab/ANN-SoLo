import collections

import numba as nb
import numpy as np
import scipy.optimize
import scipy.sparse
import spectrum_utils.spectrum as sus

from ann_solo.spectrum import _spec_to_neutral_loss

SpectrumTuple = collections.namedtuple(
    "SpectrumTuple", ["precursor_mz", "precursor_charge", "mz", "intensity"]
)

#  return type
SimilarityTuple = collections.namedtuple(
    "SimilarityTuple",
    [
        "score",
        "matched_indices",
        "matched_indices_other",
    ],
)


def cosine(
    spectrum1: sus.MsmsSpectrum,
    spectrum2: sus.MsmsSpectrum,
    fragment_mz_tolerance: float,
) -> SimilarityTuple:
    """
    Compute the cosine similarity between the given spectra.

    Parameters
    ----------
    spectrum1 : sus.MsmsSpectrum
        The first spectrum.
    spectrum2 : sus.MsmsSpectrum
        The second spectrum.
    fragment_mz_tolerance : float
        The fragment m/z tolerance used to match peaks.

    Returns
    -------
    SimilarityTuple
        A tuple consisting of the cosine similarity between both spectra,
        matched intensity, maximum contribution by a signal pair, matched
        signals, and arrays of the matching peak indexes in the first and
        second spectrum.
    """
    return _cosine(spectrum1, spectrum2, fragment_mz_tolerance, False)


def modified_cosine(
    spectrum1: sus.MsmsSpectrum,
    spectrum2: sus.MsmsSpectrum,
    fragment_mz_tolerance: float,
) -> SimilarityTuple:
    """
    Compute the modified cosine similarity between the given spectra.

    Parameters
    ----------
    spectrum1 : sus.MsmsSpectrum
        The first spectrum.
    spectrum2 : sus.MsmsSpectrum
        The second spectrum.
    fragment_mz_tolerance : float
        The fragment m/z tolerance used to match peaks.

    Returns
    -------
    SimilarityTuple
        A tuple consisting of the cosine similarity between both spectra,
        matched intensity, maximum contribution by a signal pair, matched
        signals, and arrays of the matching peak indexes in the first and
        second spectrum.
    """
    return _cosine(spectrum1, spectrum2, fragment_mz_tolerance, True)


def neutral_loss(
    spectrum1: sus.MsmsSpectrum,
    spectrum2: sus.MsmsSpectrum,
    fragment_mz_tolerance: float,
) -> SimilarityTuple:
    """
    Compute the neutral loss similarity between the given spectra.

    Parameters
    ----------
    spectrum1 : sus.MsmsSpectrum
        The first spectrum.
    spectrum2 : sus.MsmsSpectrum
        The second spectrum.
    fragment_mz_tolerance : float
        The fragment m/z tolerance used to match peaks.

    Returns
    -------
    SimilarityTuple
        A tuple consisting of the cosine similarity between both spectra,
        matched intensity, maximum contribution by a signal pair, matched
        signals, and arrays of the matching peak indexes in the first and
        second spectrum.
    """
    # Convert peaks to neutral loss.
    spectrum1 = _spec_to_neutral_loss(spectrum1)
    spectrum2 = _spec_to_neutral_loss(spectrum2)
    return _cosine(spectrum1, spectrum2, fragment_mz_tolerance, False)


def _cosine(
    spectrum1: sus.MsmsSpectrum,
    spectrum2: sus.MsmsSpectrum,
    fragment_mz_tolerance: float,
    allow_shift: bool,
) -> SimilarityTuple:
    """
    Compute the cosine similarity between the given spectra.

    Parameters
    ----------
    spectrum1 : sus.MsmsSpectrum
        The first spectrum.
    spectrum2 : sus.MsmsSpectrum
        The second spectrum.
    fragment_mz_tolerance : float
        The fragment m/z tolerance used to match peaks.
    allow_shift : bool
        Boolean flag indicating whether to allow peak shifts or not.

    Returns
    -------
    SimilarityTuple
        A tuple consisting of the cosine similarity between both spectra,
        matched intensity, maximum contribution by a signal pair, matched
        signals, and arrays of the matching peak indexes in the first and
        second spectrum.
    """
    spec_tup1 = SpectrumTuple(
        spectrum1.precursor_mz,
        spectrum1.precursor_charge,
        spectrum1.mz,
        np.copy(spectrum1.intensity) / np.linalg.norm(spectrum1.intensity),
    )
    spec_tup2 = SpectrumTuple(
        spectrum2.precursor_mz,
        spectrum2.precursor_charge,
        spectrum2.mz,
        np.copy(spectrum2.intensity) / np.linalg.norm(spectrum2.intensity),
    )
    return _cosine_fast(
        spec_tup1, spec_tup2, fragment_mz_tolerance, allow_shift
    )


@nb.njit(fastmath=True, boundscheck=False)
def _cosine_fast(
    spec: SpectrumTuple,
    spec_other: SpectrumTuple,
    fragment_mz_tolerance: float,
    allow_shift: bool,
) -> SimilarityTuple:
    """
    Compute the cosine similarity between the given spectra.

    Parameters
    ----------
    spec : SpectrumTuple
        Numba-compatible tuple containing information from the first spectrum.
    spec_other : SpectrumTuple
        Numba-compatible tuple containing information from the second spectrum.
    fragment_mz_tolerance : float
        The fragment m/z tolerance used to match peaks in both spectra with
        each other.
    allow_shift : bool
        Boolean flag indicating whether to allow peak shifts or not.

    Returns
    -------
    SimilarityTuple
        A tuple consisting of the cosine similarity between both spectra,
        matched intensity, maximum contribution by a signal pair, matched
        signals, and arrays of the matching peak indexes in the first and
        second spectrum.
    """
    # Find the matching peaks between both spectra, optionally allowing for
    # shifted peaks.
    # Candidate peak indices depend on whether we allow shifts
    # (check all shifted peaks as well) or not.
    # Account for unknown precursor charge (default: 1).
    precursor_charge = max(spec.precursor_charge, 1)
    precursor_mass_diff = (
        spec.precursor_mz - spec_other.precursor_mz
    ) * precursor_charge
    # Only take peak shifts into account if the mass difference is relevant.
    num_shifts = 1
    if allow_shift and abs(precursor_mass_diff) >= fragment_mz_tolerance:
        num_shifts += precursor_charge
    other_peak_index = np.zeros(num_shifts, np.uint16)
    mass_diff = np.zeros(num_shifts, np.float32)
    for charge in range(1, num_shifts):
        mass_diff[charge] = precursor_mass_diff / charge

    # Find the matching peaks between both spectra.
    cost_matrix = np.zeros((len(spec.mz), len(spec_other.mz)), np.float32)
    for peak_index, (peak_mz, peak_intensity) in enumerate(
        zip(spec.mz, spec.intensity)
    ):
        # Advance while there is an excessive mass difference.
        for cpi in range(num_shifts):
            while other_peak_index[cpi] < len(spec_other.mz) - 1 and (
                peak_mz - fragment_mz_tolerance
                > spec_other.mz[other_peak_index[cpi]] + mass_diff[cpi]
            ):
                other_peak_index[cpi] += 1
        # Match the peaks within the fragment mass window if possible.
        for cpi in range(num_shifts):
            index = 0
            other_peak_i = other_peak_index[cpi] + index
            while (
                other_peak_i < len(spec_other.mz)
                and abs(
                    peak_mz - (spec_other.mz[other_peak_i] + mass_diff[cpi])
                )
                <= fragment_mz_tolerance
            ):
                cost_matrix[peak_index, other_peak_i] = (
                    peak_intensity * spec_other.intensity[other_peak_i]
                )
                index += 1
                other_peak_i = other_peak_index[cpi] + index

    with nb.objmode(row_ind="int64[:]", col_ind="int64[:]"):
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(
            cost_matrix, maximize=True
        )

    score = 0.0
    row_mask = np.zeros_like(row_ind, np.bool_)
    col_mask = np.zeros_like(col_ind, np.bool_)
    for (i, row), (j, col) in zip(enumerate(row_ind), enumerate(col_ind)):
        pair_score = cost_matrix[row, col]
        if pair_score > 0.0:
            score += pair_score
            row_mask[i] = col_mask[j] = True

    return SimilarityTuple(
        score,
        row_ind[row_mask],
        col_ind[col_mask],
    )
