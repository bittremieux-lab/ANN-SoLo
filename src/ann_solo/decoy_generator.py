from typing import List, Tuple, Dict

import numpy as np
from spectrum_utils import fragment_annotation, proforma
from spectrum_utils.spectrum import MsmsSpectrum

from ann_solo.config import config


def _get_similarity(target: str, decoy: str) -> float:
    """
    Computes the similarity between two strings using the edit distance
    based on a dynamic programming approach.
    Parameters
    ----------
    target: str
        The first string (reference string).
    decoy: str
        The second string (comparison string).

    Returns
    -------
    float
        A similarity score between 0 and 1, where 1 indicates identical strings
        and 0 indicates completely different strings.
    """
    m, n = len(target), len(decoy)

    # Init for dynamic programming
    previous_row = list(range(n + 1))
    current_row = [0] * (n + 1)

    for i in range(1, m + 1):
        current_row[0] = i
        for j in range(1, n + 1):
            if target[i - 1] == decoy[j - 1]:
                current_row[j] = previous_row[j - 1]
            else:
                current_row[j] = 1 + min(previous_row[j],    # Deletion
                                         current_row[j - 1], # Insertion
                                         previous_row[j - 1]) # Substitution
        previous_row, current_row = current_row, previous_row

    # Get the edit distance
    edit_distance = previous_row[n]
    # Compute similarity
    max_length = max(m, n)
    similarity = 1 - (edit_distance / max_length) if max_length > 0 else 1.0

    return similarity

def _shuffle(peptide_sequence: str, excluded_residues: List[str] =['K', 'R', 'P'],
             max_similarity: float = 0.7) -> Tuple[str, Dict[int, int]]:
    """
    Shuffles a peptide sequence randomly by keeping the number of tryptic
    termini and missed internal cleavages (K,R,P). The shuffled sequence has
    to be at least 70% dissimilar than the original sequence.

    Parameters
    ----------
    peptide_sequence: str
        The peptide sequence to shuffle.
    excluded_residues: List[str]
        The list of amino acids to maintain the tryptic property.

    Returns
    -------
    Tuple[str, Dict[int, int]]
        A tuple, where the first returned value is the shuffled sequence
        and the second is the mapping indecies of the shuffle.
    """
    # Parse peptide
    seq_original = list(peptide_sequence)
    # Create a list of the indices of the residuess that should not be shuffled
    indices_to_exclude = [i for i, elem in enumerate(seq_original[:-1]) if
                          elem in excluded_residues] + [len(seq_original) - 1]

    # Best permutation values
    best_similarity, best_shuffled, best_permutation = 1, '', []
    for i in range(10):
        # Shuffle the elements of original sequence, but exclude the
        # elements at the specified indices
        seq_shuffled = np.array(seq_original)
        random_permutation = np.random.permutation(
            [i for i in range(len(seq_shuffled)) if
             i not in indices_to_exclude])
        random_permutation = random_permutation.tolist()

        full_permutation = [
            random_permutation.pop(0) if i not in indices_to_exclude else i for
            i in range(len(seq_shuffled))]

        seq_shuffled = seq_shuffled[full_permutation]
        # Compute the similarity between seq_shuffled and seq_original using
        #  edit distance
        similarity = _get_similarity(seq_original, seq_shuffled.tolist())
        # Check if similarity is below the specified threshold
        if similarity <= max_similarity:
            return ''.join(seq_shuffled), {full_permutation[i]:i  for i in
                                           range(len(seq_original))}
        elif similarity <= best_similarity:
            best_similarity, best_shuffled, best_permutation = similarity, ''.join(
                seq_shuffled), full_permutation

    if  best_similarity > max_similarity:
        best_shuffled = None
		
    return best_shuffled, {best_permutation[i]:i  for i in
                           range(len(seq_original))}


def _decoy_seq_to_proforma(decoy_spectrum: MsmsSpectrum) -> str:
    """
    Takes the decoy spectrum and return a peptide sequence in the ProForma
    format.

    Parameters
    ----------
    decoy_spectrum: MsmsSpectrum
        Decoy spectrum.

    Returns
    -------
    str
        Modified peptide in its ProForma format.
    """
    if decoy_spectrum.proforma.modifications is None:
        return decoy_spectrum.proforma.sequence
    else:
        peptide = list(decoy_spectrum.proforma.sequence)
        modifications = {mod.position: mod.mass for mod in
                         decoy_spectrum.proforma.modifications}
        for shift, position in enumerate(sorted(modifications.keys())):
            peptide.insert(position + shift + 1, '['+str(modifications[position])+']')
        return ''.join(peptide)


def shuffle_and_reposition(spectrum: MsmsSpectrum) -> MsmsSpectrum:
    """
    Creates a decoy spectrum from a real spectrum.

    Parameters
    ----------
    spectrum: MsmsSpectrum
        Real spectrum.
    Returns
    -------
    MsmsSpectrum
        Decoy spectrum.
    """
    # annotate original spectrum
    spectrum.annotate_proforma(
        spectrum.peptide,
        config.fragment_mz_tolerance,
        config.fragment_tol_mode,
        "abpy",
        neutral_losses=True
    )

    # parse original spectrum
    parsed_sequence = proforma.parse(spectrum.proforma)
    shuffled_sequence, shuffled_sequence_mapping = _shuffle(
        parsed_sequence[0].sequence)
	# If decoy peptide is same, no decoy generation	
    if shuffled_sequence is None:
        return None
		
    # Compute theoretical fragment m/z of the original peptide
    genuine_peptide_theoretical_fragments = {
    str(ion.ion_type) + '^' + str(ion.charge): mz for ion, mz in
    fragment_annotation.get_theoretical_fragments(
        parsed_sequence[0], ion_types="abpy",
        max_charge=spectrum.precursor_charge,
        neutral_losses=fragment_annotation._neutral_loss)}

    # Constract decoy proteoform
    if parsed_sequence[0].modifications is not None:
        for modification in parsed_sequence[0].modifications:
            setattr(modification, 'position',
                    shuffled_sequence_mapping[modification.position])
    decoy_proforma = proforma.Proteoform(shuffled_sequence,
                                         parsed_sequence[0].modifications)

    # Compute theoretical fragment m/z of the shuffled peptide
    decoy_theoretical_fragments = {
    str(ion.ion_type) + '^' + str(ion.charge): mz for ion, mz in
    fragment_annotation.get_theoretical_fragments(
        decoy_proforma, ion_types="abpy",
        max_charge=spectrum.precursor_charge,
        neutral_losses=fragment_annotation._neutral_loss)}

    # Initialize decoy fragment arrays
    mz_shuffled, intensity_shuffled, annotation_shuffled = np.zeros_like(
        spectrum.mz), np.zeros_like(spectrum.intensity), np.full_like(
        spectrum.annotation, None, object)
    # Initialize new original fragment annotation array
    annotation_original = np.full_like(spectrum.annotation, None, object)
    for i in range(len(spectrum.mz)):
        peak_annotation = str(spectrum.annotation[i][0].ion_type) + '^' + str(
            spectrum.annotation[i][0].charge)
        # Assign fragment values
        intensity_shuffled[i] = spectrum.intensity[i]
        mz_shuffled[i] = spectrum.mz[i]
        annotation_shuffled[i] = None if str(spectrum.annotation[i]) == '?' \
                                else fragment_annotation.FragmentAnnotation(
                                spectrum.annotation[i][0].ion_type,
                                charge=spectrum.annotation[i][0].charge)

        # To maintain FragmentAnnotation obj for all annotations
        annotation_original[i] = None if str(spectrum.annotation[i]) == '?' \
                                else fragment_annotation.FragmentAnnotation(
                                spectrum.annotation[i][0].ion_type,
                                charge=spectrum.annotation[i][0].charge)
        # Repositon  peak according to the shuffled peptide
        if decoy_theoretical_fragments.get(peak_annotation) is not None:
            # Reposition! Take into account the original mass error
            mz_shuffled[i] = decoy_theoretical_fragments.get(peak_annotation) + \
                             (spectrum.mz[i] -
                              genuine_peptide_theoretical_fragments.get(peak_annotation))
    # Replace original annotation
    spectrum._annotation = annotation_original

    # Reorder the arrays based on the sorted mz array
    sorted_indices = np.argsort(mz_shuffled)
    mz_shuffled, intensity_shuffled, annotation_shuffled = mz_shuffled[sorted_indices], \
                                                           intensity_shuffled[sorted_indices], \
                                                           annotation_shuffled[sorted_indices]
    # Create a new `MsmsSpectrum` from the shuffled peaks
    decoy_spectrum = MsmsSpectrum("DECOY_" + spectrum.identifier,
                                  spectrum.precursor_mz,
                                  spectrum.precursor_charge, mz_shuffled,
                                  intensity_shuffled)
    decoy_spectrum.proforma = decoy_proforma
    decoy_spectrum.peptide = _decoy_seq_to_proforma(decoy_spectrum)
    decoy_spectrum._annotation = annotation_shuffled
    decoy_spectrum.is_decoy = True

    return decoy_spectrum
