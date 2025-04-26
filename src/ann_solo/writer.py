import logging
import os
import pathlib
import re
from typing import AnyStr
from typing import List
from typing import Pattern
from typing import Union

from ann_solo.config import config
from ann_solo.reader import SpectralLibraryReader
from ann_solo.spectrum import SpectrumSpectrumMatch
from . import __version__


def natural_sort_key(s: str, _nsre: Pattern[AnyStr] = re.compile('([0-9]+)'))\
        -> List[Union[str, int]]:
    """
    Key to be used for natural sorting of mixed alphanumeric strings.

    Source: https://stackoverflow.com/a/16090640

    Parameters
    ----------
    s : str
        The string to be converted to a sort key.
    _nsre : Pattern[AnyStr]
        Pattern to split the given string into alphanumeric substrings.

    Returns
    -------
    List[Union[str, int]]
        A list of separate int (numeric) and string (alphabetic) parts of the
        given string.
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]


def write_mztab(identifications: List[SpectrumSpectrumMatch], filename: str,
                lib_reader: SpectralLibraryReader) -> str:
    """
    Write the given SSMs to an mzTab file.

    Parameters
    ----------
    identifications : List[SpectrumSpectrumMatch]
        The identifications to be exported.
    filename : str
        The file name of the mzTab output file. If it does not end with a
        '.mztab' extension this will be added.
    lib_reader : SpectralLibraryReader
        The spectral library reader used during identifications.

    Returns
    -------
    str
        The file name of the mzTab output file.
    """
    # Check if the filename contains the mztab extension and add if required.
    if os.path.splitext(filename)[1].lower() != '.mztab':
        filename += '.mztab'

    logging.info('Save identifications to file %s', filename)

    # Collect the necessary metadata.
    metadata = [
        ('mzTab-version', '1.0.0'),
        ('mzTab-mode', 'Summary'),
        ('mzTab-type', 'Identification'),
        ('mzTab-ID', f'ANN-SoLo_{filename}'),
        ('title', f'ANN-SoLo identification file "{filename}"'),
        ('description', f'Identification results of file '
                        f'"{os.path.split(config.query_filename)[1]}" against '
                        f'spectral library file '
                        f'"{os.path.split(config.spectral_library_filename)[1]}"'),
        ('software[1]', f'[MS, MS:1003357, ANN-SoLo, {__version__}]'),
        ('psm_search_engine_score[1]', '[MS, MS:1001143, search engine '
                                       'specific score for PSMs,]'),
        ('psm_search_engine_score[2]', '[MS, MS:1002354, PSM-level q-value,]'),
        ('ms_run[1]-format', '[MS, MS:1001062, Mascot MGF file,]'),
        ('ms_run[1]-location', pathlib.Path(
            os.path.abspath(config.query_filename)).as_uri()),
        ('ms_run[1]-id_format', '[MS, MS:1000774, multiple peak list nativeID '
                                'format,]'),
        ('fixed_mod[1]', '[MS, MS:1002453, No fixed modifications searched,]'),
        ('variable_mod[1]', '[MS, MS:1002454, No variable modifications '
                            'searched,]'),
        ('false_discovery_rate', f'[MS, MS:1002350, PSM-level global FDR, '
                                 f'{config.fdr}]'),
    ]

    # Add relevant configuration settings.
    config_keys = [
        'resolution', 'min_mz', 'max_mz', 'remove_precursor',
        'remove_precursor_tolerance', 'min_intensity', 'min_peaks',
        'min_mz_range', 'max_peaks_used', 'max_peaks_used_library', 'scaling',
        'precursor_tolerance_mass', 'precursor_tolerance_mode',
        'precursor_tolerance_mass_open', 'precursor_tolerance_mode_open',
        'fragment_mz_tolerance', 'allow_peak_shifts', 'fdr',
        'fdr_min_group_size','mode']
    if config.mode == 'ann':
        config_keys.extend(['bin_size', 'hash_len', 'num_candidates',
                            'num_list', 'num_probe'])
    for i, key in enumerate(config_keys):
        metadata.append((f'software[1]-setting[{i}]',
                         f'{key} = {config[key]}'))

    database_version = lib_reader.get_version()

    with open(filename, 'w') as f_out:
        # Metadata section.
        for m in metadata:
            f_out.write('\t'.join(['MTD'] + list(m)) + '\n')

        # SSMs.
        f_out.write('\t'.join([
            'PSH', 'sequence', 'PSM_ID', 'accession', 'unique', 'database',
            'database_version', 'search_engine', 'search_engine_score[1]',
            'search_engine_score[2]', 'modifications', 'retention_time',
            'charge', 'exp_mass_to_charge', 'calc_mass_to_charge',
            'spectra_ref', 'pre', 'post', 'start', 'end',
            'opt_ms_run[1]_cv_MS:1003062_spectrum_index',
            'opt_ms_run[1]_cv_MS:1002217_decoy_peptide',
            'opt_ms_run[1]_num_candidates']) + '\n')
        # SSMs sorted by their query identifier.
        for ssm in sorted(identifications,
                          key=lambda s: natural_sort_key(s.query_identifier)):
            f_out.write('\t'.join([
                'PSM',
                ssm.sequence,
                str(ssm.query_identifier),
                'null', 'null',
                pathlib.Path(os.path.abspath(
                    config.spectral_library_filename)).as_uri(),
                database_version,
                '[MS, MS:1003357, ANN SoLo,]',
                str(ssm.search_engine_score),
                str(ssm.q),
                'null',
                str(ssm.retention_time),
                str(ssm.charge),
                str(ssm.exp_mass_to_charge),
                str(ssm.calc_mass_to_charge),
                f'ms_run[1]:index={ssm.query_index}',
                'null', 'null', 'null', 'null',
                str(ssm.library_identifier),
                f'{ssm.is_decoy:d}']) + '\n')

    return filename
