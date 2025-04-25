import collections
import io
import mmap
import logging
import os
import pickle
import re
from functools import lru_cache
from typing import Dict, IO, Iterator, List, Tuple, Union

import joblib
import lancedb
import numpy as np
import pandas as pd
import pyarrow as pa
import tqdm
from lxml.etree import LxmlError
from pyteomics import fasta, mass, mgf, mzml, mzxml, parser
from spectrum_utils.spectrum import MsmsSpectrum

from ann_solo import utils
from ann_solo.config import config
from ann_solo.decoy_generator import shuffle_and_reposition, _shuffle
from ann_solo.parsers import SplibParser
from ann_solo.prosit import get_predictions
from ann_solo.spectrum import process_spectrum, spectrum_to_vector

class SpectralLibraryReader:
    """
    Read spectra from a spectral library file.
    """

    _supported_extensions = ['.splib','.sptxt','.mgf', '.fasta']

    is_recreated = False

    def __init__(self, filename: str, config_hash: str = None) -> None:
        """
        Initialize the spectral library reader. Metadata for future easy access
        of the individual spectra is read from the corresponding configuration
        file.

        The configuration file contains minimally for each spectrum in the
        spectral library its precursor charge and precursor mass to quickly
        filter the spectra library. Furthermore, it also contains the settings
        used to construct this spectral library to make sure these match the
        runtime settings.

        Parameters
        ----------
        filename : str
            The file name of the spectral library.
        config_hash : str, optional
            The hash representing the current spectral library configuration.

        Raises
        ------
        FileNotFoundError
            The given spectral library file wasn't found.
        ValueError
            The configuration file wasn't found or its settings don't
            correspond to the runtime settings.
        """
        self._filename = filename
        _, self._filename_ext = os.path.splitext(os.path.basename(filename))
        self._config_hash = config_hash
        self._parser = None
        self._spectral_library_store = None
        do_create = False

        # Test if the given spectral library file is in a supported format.
        verify_extension(self._supported_extensions, self._filename)

        logging.debug('Load the spectral library configuration')

        # Verify that the configuration file
        # corresponding to this spectral library is present.
        config_filename = self._get_config_filename()
        store_filename = self._get_store_filename()

        if not os.path.isfile(config_filename) or not os.path.isdir(store_filename):
            # If not we should recreate this file
            # prior to using the spectral library.
            do_create = True
            logging.warning('Missing spectral library store or configuration '
                            'file')
        else:
            # Load the configuration file.
            config_lib_filename, self.spec_info, load_hash =\
                joblib.load(config_filename)

            # Check that the same spectral library file format is used.
            if config_lib_filename != os.path.basename(self._filename):
                do_create = True
                logging.warning('The configuration corresponds to a different '
                                'file format of this spectral library')
            # Verify that the runtime settings match the loaded settings.
            if self._config_hash != load_hash:
                do_create = True
                logging.warning('The spectral library search engine was '
                                'created using non-compatible settings')

        # (Re)create the spectral library configuration
        # if it is missing or invalid.
        if do_create:
            self._create_config()

        # Open the Spectral Library Store
        self._spectral_library_store = SpectralLibraryStore(
            self._get_store_filename())
        self._spectral_library_store.open_store()

    def _get_config_filename(self) -> str:
        """
        Gets the configuration file name for the spectral library with the
        current configuration.

        Returns
        -------
        str
            The configuration file name (.spcfg file).
        """
        if self._config_hash is not None:
            return (f'{os.path.splitext(self._filename)[0]}_'
                    f'{self._config_hash[:7]}.spcfg')
        else:
            return f'{os.path.splitext(self._filename)[0]}.spcfg'

    def _get_store_filename(self) -> str:
        """
        Gets the spectra library store file name for the spectral library
        with the current configuration.

        Returns
        -------
        str
            The spectral library file name (.hdf5 file).
        """
        if self._config_hash is not None:
            return (f'{os.path.splitext(self._filename)[0]}_'
                    f'{self._config_hash[:7]}')
        else:
            return f'{os.path.splitext(self._filename)[0]}'

    def _spectrum_to_dict(self, spectrum: MsmsSpectrum) -> dict:
        """
        Convert an MsmsSpectrum object into a dictionary representation for
        bulk insertion.
        """
        return {
            'identifier': spectrum.identifier,
            'peptide': spectrum.peptide,
            'precursor_charge': spectrum.precursor_charge,
            'precursor_mz': spectrum.precursor_mz,
            'mz': spectrum.mz,
            'intensity': spectrum.intensity,
            'annotation': spectrum.annotation,
            'is_decoy': spectrum.is_decoy,
            'projection': spectrum.projection
        }

    def _create_config(self) -> None:
        """
        Create a new configuration file for the spectral library.

        The configuration file contains for each spectrum in the spectral
        library its offset for quick random-access reading, and its precursor
        m/z for filtering using a precursor mass window. Finally, it also
        contains the settings used to construct this spectral library to make
        sure these match the runtime settings.
        """
        logging.info('Create the spectral library configuration for file %s',
                     self._filename)

        self.is_recreated = True

        # Read all the spectra in the spectral library.
        temp_info = collections.defaultdict(
            lambda: {'id': [], 'precursor_mz': []})
        # Block to store all spectra peptides if add_decoys parameter is
        # set to True
        target_peptides = set()
        if config.add_decoys and not self._filename_ext == '.fasta':
            with self as lib_reader:
                logging.info(
                    'Read all target peptides in the library first to avoid '
                    'similar generated decoys.')
                for spectrum in tqdm.tqdm(
                        lib_reader.read_library_file(),
                        desc='Library spectra read', unit='spectra'):
                    target_peptides.add(spectrum.peptide)
        # Process the input library
        with self as lib_reader:
            spectra_store = SpectralLibraryStore(self._get_store_filename())
            spectra_store.open_store("w")
            batch = []
            batch_cnt = 0 # For bulk insert
            for spectrum in tqdm.tqdm(
                    lib_reader.read_library_file(),
                    desc='Library spectra read', unit='spectra'):
                if config.add_decoys and not self._filename_ext == '.fasta':
                    # Compute decoy
                    decoy_spectrum = shuffle_and_reposition(spectrum)
                    if decoy_spectrum and decoy_spectrum.peptide not in \
                            target_peptides:
                        # Pre-process the decoy spectrum
                        decoy_spectrum.is_processed = False
                        decoy_spectrum = process_spectrum(decoy_spectrum, True)
                        if decoy_spectrum.is_valid:
                            info_charge = temp_info[
                                decoy_spectrum.precursor_charge]
                            info_charge['id'].append(decoy_spectrum.identifier)
                            info_charge['precursor_mz'].append(
                                decoy_spectrum.precursor_mz)
                            # Get the vector representation
                            decoy_spectrum.projection = spectrum_to_vector(
                                decoy_spectrum,
                                config.min_mz,
                                config.max_mz,
                                config.bin_size,
                                int(config.hash_len / 2)
                            )
                            # Append for bulk insert
                            batch.append(self._spectrum_to_dict(decoy_spectrum))
                            batch_cnt += 1

                # Pre-process the target spectrum
                spectrum.is_processed = False
                spectrum = process_spectrum(spectrum,True)
                if spectrum.is_valid:
                    info_charge = temp_info[spectrum.precursor_charge]
                    info_charge['id'].append(spectrum.identifier)
                    info_charge['precursor_mz'].append(spectrum.precursor_mz)
                    # Get the vector representation
                    spectrum.projection = spectrum_to_vector(
                        spectrum,
                        config.min_mz,
                        config.max_mz,
                        config.bin_size,
                        int(config.hash_len / 2)
                    )
                    # Append for bulk insert
                    batch.append(self._spectrum_to_dict(spectrum))
                    batch_cnt += 1
                # Persist to store every half a million spectra collected
                if batch_cnt >= 500000:
                    spectra_store.write_spectra_to_library(batch)
                    batch = []
                    batch_cnt = 0
            if batch:
                spectra_store.write_spectra_to_library(batch)
            logging.info(
                'Create scalar index on charge and identifier for fast '
                'retrieval from store.')
            spectra_store.store.create_scalar_index("precursor_charge",
                                                    index_type="BITMAP")
            spectra_store.store.create_scalar_index("identifier")
        self.spec_info = {
            'charge': {
                charge: {
                    'id': np.asarray(charge_info['id']),
                    'precursor_mz': np.asarray(charge_info['precursor_mz'],
                                               np.float32)
                } for charge, charge_info in temp_info.items()}
        }
        # Store the configuration.
        config_filename = self._get_config_filename()
        logging.debug('Save the spectral library configuration to file %s',
                      config_filename)
        joblib.dump(
            (os.path.basename(self._filename), self.spec_info,
             self._config_hash),
            config_filename, compress=9, protocol=pickle.DEFAULT_PROTOCOL)


    def open(self) -> None:
        self._parser = SplibParser(self._filename.encode())

    def close(self) -> None:
        if self._parser is not None:
            del self._parser

    def __enter__(self) -> 'SpectralLibraryReader':
        if self._filename_ext == '.splib':
            self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._filename_ext == '.splib':
            self.close()

    @lru_cache(maxsize=None)
    def read_spectrum(self, spec_id: str, process_peaks: bool = False)\
            -> MsmsSpectrum:
        """
        Read the spectrum with the specified identifier from the spectral
        library store.

        Parameters
        ----------
        spec_id : string
            The identifier of the spectrum in the spectral library file.
        process_peaks : bool, optional
            Flag whether to process the spectrum's peaks or not
            (the default is false to not process the spectrum's peaks).

        Returns
        -------
        Spectrum
            The spectrum from the spectral library store with the specified
            identifier.
        """
        spectrum = self._spectral_library_store.read_spectrum_from_library(
                                                spec_id)
        spectrum.is_processed = False
        if process_peaks:
            spectrum = process_spectrum(spectrum, True)
            spectrum._annotation = [None] * len(spectrum.intensity)


        return spectrum

    def read_spectra_projections(self, spec_ids: List[str], charge) -> List[
        np.ndarray]:
        """
        Gets a library spectra ids and returns the corresponding spectra
        projections.

        Parameters
        ----------
        spec_ids : List[str]
            A list of library spectrum ids.

        Returns
        -------
        List[np.ndarray]
            List of spectra projections corresponding to the spectra ids passed.
        """
        # Fetch queried spectra from the library
        queried_spectra = \
            self._spectral_library_store.fetch_projections_in_batch(
            spec_ids, charge)
        # Create a temporary dataframe with "id" and "projection" columns
        # Initialize "projection" with numpy arrays of zeros
        temp_dataframe = pd.DataFrame({
            "id": spec_ids
        })
        # Assuming queried_spectra is a pandas DataFrame with "identifier" and "projection" columns
        # Perform a left join to ensure missing identifiers are filled with
        # zeros and respect the order of ids
        merged_df = temp_dataframe.merge(
            queried_spectra.rename(columns={"identifier": "id"}),
            # Rename to match temp_dataframe
            on="id",
            how="left"
        )
        # Replace NaN projections with the zero-initialized values
        zero_vector = np.zeros(config.hash_len, dtype=np.float32)
        merged_df["projection"] = merged_df["projection"].where(merged_df["projection"].notna(), zero_vector)

        # Return projections as a list of numpy arrays
        return merged_df["projection"].tolist()


    def read_spectra_candidates(self, spec_ids: List[str], charge) -> List[
        MsmsSpectrum]:
        """
        Gets a library spectra ids and returns the corresponding spectra
        projections.

        Parameters
        ----------
        spec_ids : List[str]
            A list of library spectrum ids.

        Returns
        -------
        List[MsmsSpectrum]
            List of spectra projections corresponding to the spectra ids passed.
        """
        return self._spectral_library_store.read_spectra_from_library(
            spec_ids, charge)


    def read_library_file(self) -> Iterator[MsmsSpectrum]:
        """
        Read/generate all spectra from spectral library or FASTA file.

        Returns
        -------
        Iterator[Spectrum]
            An iterator of spectra in the given library file.
        """

        if self._filename_ext == '.splib':
            self._parser.seek_first_spectrum()
            try:
                while True:
                    spectrum, _ = self._parser.read_spectrum()
                    spectrum.is_processed = False
                    yield spectrum
            except StopIteration:
                return
        elif self._filename_ext == '.sptxt':
            yield from self.read_sptxt()
        elif self._filename_ext == '.mgf':
            yield from read_mgf(self._filename)
        elif self._filename_ext == '.fasta':
            yield from read_fasta(self._filename)


    def get_version(self) -> str:
        """
        Gives the spectral library version.

        Returns
        -------
        str
            A string representation of the spectral library version.
        """
        return 'null'

    def _sptxt_seq_to_proforma(self, peptide: str, modifications: List[str]) \
            -> str:
        """
        Takes a peptide and a list of modifications to return a modified
        peptide in its ProForma format.

        Parameters
        ----------
        peptide : str
            Peptide sequence in its non-modified format.
        modifications: List[str]
            A list of modifications.

        Returns
        -------
        str
            Modified peptide in its ProForma format.
        """
        peptide = parser.parse(peptide)
        for shift, modification in enumerate(modifications):
            idx, aa, modification_name = modification.split(',')
            peptide.insert(int(idx) + shift + 1, '[' + modification_name + ']')
        return ''.join(peptide)

    def _parse_sptxt_spectrum(self, identifier: int, raw_spectrum: str)\
            -> MsmsSpectrum:
        """
        Takes a raw spectrum data retrieved from an sptxt file and
        parses it to a structured object of type MsmsSpectrum.

        Parameters
        ----------
        identifier : int
            Incremented identifier of the spectrum in the library.
        raw_spectrum : string
            The spectrum in a raw format.

        Returns
        -------
        MsmsSpectrum
            An MsmsSpectrum object.
        """
        # Split raw spectrum in two chunks: metadata & spectrum
        raw_spectrum_tokens = re.split('Num\s?Peaks:\s?[0-9]+\n',
                                       raw_spectrum.strip(),
                                       flags=re.IGNORECASE)
        spectrum_metadata = raw_spectrum_tokens[0]
        spectrum = raw_spectrum_tokens[1]
        # Check if decoy
        decoy = True if re.search('decoy', spectrum_metadata,
                                  re.IGNORECASE) else False
        # Retrieve peptide & charge
        peptide_charge = spectrum_metadata.split('\n', 1)[0].split('/')
        peptide = peptide_charge[0].split(' ')[-1].strip()
        charge = int(peptide_charge[1].strip())
        # Retrieve precurssor mass
        precursor_mz = re.search('PrecursorMZ:\s?[0-9]+.[0-9]+', spectrum_metadata,
                          re.IGNORECASE)
        if precursor_mz:
            precursor_mz = re.search('[0-9]+.[0-9]+', precursor_mz.group(0))
        else:
            precursor_mz = re.search('Parent=\s?[0-9]+.[0-9]+', spectrum_metadata,
                              re.IGNORECASE)
            precursor_mz = re.search('[0-9]+.[0-9]+', precursor_mz.group(0))
        # Retrieve modifications
        modifications = re.search('Mods=.+?(?=[\s\n])',
                                 spectrum_metadata,
                                 re.IGNORECASE)
        if modifications:
            modifications = str(modifications.group(0)).split('/')[1:]
        else:
            modifications = None
        # Retrieve m/z, intensity, and annotation
        file = io.StringIO(spectrum.strip())
        mz_intensity_annotation = np.loadtxt(file, delimiter='\t',
                                             dtype=str, usecols=(0, 1, 2))


        if mz_intensity_annotation.shape[1] > 2:
            annotation = [utils._parse_fragment_annotation(annotation)
                          for mz, annotation in
                          zip(mz_intensity_annotation[:, 0].astype(float),
                              mz_intensity_annotation[:, 2].astype(str))]
        else:
            annotation = [None] * len(mz_intensity_annotation[0])
        spectrum = MsmsSpectrum(str(identifier), float(precursor_mz.group(0)),
                                charge,
                                mz_intensity_annotation[:, 0].astype(float),
                                mz_intensity_annotation[:, 1].astype(float))

        spectrum.peptide = self._sptxt_seq_to_proforma(peptide,modifications)
        spectrum.is_decoy = decoy
        spectrum._annotation = annotation

        return spectrum

    def _parse_sptxt(self) -> Iterator[Tuple[int,str]]:
        """
        Open the sptxt spectra library file and parses it
        to read all spectra.

        Returns
        -------
        Iterator[Tuple[int,str]]
            An iterator of tuples of (id, spectrum) in the given library file,
            where spectrum is in its raw text format.

        """
        with open(self._filename, 'rb') as file:
            mmapped_file = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
            for id, raw_spectrum in tqdm.tqdm(enumerate(re.finditer(
                            b'(?<![a-zA-Z])Name:\s?(?:(?!((?<![a-zA-Z])Name:\s?)).|\n)*',
                            mmapped_file.read(),
                            re.IGNORECASE),
                        1), desc='SpectraST file parse',
                        unit='spectra'):
                yield (id,'\n'.join(raw_spectrum.group(0).decode(
                    'utf-8').splitlines()))

    def read_sptxt(self) -> Iterator[MsmsSpectrum]:
        """
        Open read spectra from SpectraST spectra library.

        Returns
        -------
        Iterator[MsmsSpectrum]
            An iterator of spectra in the given library file.

        """
        # TODO: Use all logical units in the system (-1)
        for spectrum in joblib.Parallel(n_jobs=1,
                                        backend='multiprocessing')(
                joblib.delayed(
                    self._parse_sptxt_spectrum
                )(id, raw_spectrum) for id, raw_spectrum in
                self._parse_sptxt()):
            yield spectrum



class SpectralLibraryStore:
    """
        Class to efficiently store and retrieve spectra from a library file.
    """
    def __init__(self, file_path: str) -> None:
        """
        Initialize the spectral library store.

        Parameters
        ----------
        filepath : str
            The file path of the spectral library store.

        """
        self.file_path = file_path
        self.store = None

    def open_store(self, mode="r") -> None:
        """
        Create a lance dataset pointer to the spectral library store for read
        purposes.
        """
        # Open or create a LanceDB database
        db = lancedb.connect(self.file_path)
        if mode == "r":
            self.store = db.open_table("SpectralLibraryStore")
        elif mode == "w":
            schema = pa.schema([
                pa.field("identifier", pa.string()),
                pa.field("peptide", pa.string()),
                pa.field("precursor_charge", pa.int64()),
                pa.field("precursor_mz", pa.float64()),
                pa.field("mz", pa.list_(pa.float64())),
                pa.field("intensity", pa.list_(pa.float32())),
                pa.field("annotation", pa.list_(pa.string())),
                pa.field("is_decoy", pa.bool_()),
                pa.field("projection", pa.list_(pa.float32()))
            ])
            self.store = db.create_table("SpectralLibraryStore", schema=schema, mode="overwrite")

    def get_all_spectra_ids(self) -> Iterator[str]:
        """
        Retrieves all spectrum identifiers from the spectral library.

        Returns
        -------
        Iterator[str]
            An iterator yielding the unique identifiers of spectra
            stored in the library.
        """
        yield from self.store.to_arrow()["item"].to_pylist()

    def write_spectrum_to_library(self, spectrum : MsmsSpectrum) -> None:
        """
        Gets an Msmsspectrum object and stores it in the
        spectral library store for future retrieval.

        Parameters
        ----------
        spectrum : MsmsSpectrum
            The MsmsSpectrum object.

        """
        # Convert to arrow table
        spectrum_dict = {
            'identifier': [str(spectrum.identifier)],
            'peptide': [spectrum.peptide],
            'precursor_charge': [spectrum.precursor_charge],
            'precursor_mz': [spectrum.precursor_mz],
            'mz': [spectrum.mz],
            'intensity': [spectrum.intensity],
            'annotation': [[str(ann) for ann in spectrum.annotation]],
            'is_decoy': [spectrum.is_decoy],
            'projection': [spectrum.projection]

        }
        # Persist to lance
        self.store.add(spectrum_dict)

    def read_spectrum_from_library(self, spec_id : str)-> MsmsSpectrum:
        """
        Gets a library spectrum id and returns the corresponding spectrum as
        an Msmsspectrum object.

        Parameters
        ----------
        spec_id : string
            A library spectrum id.

        Returns
        -------
        MsmsSpectrum
            An MsmsSpectrum object.
        """
        spectrum_specs = self.store.search().where(f"identifier = '{str(spec_id)}'").to_pandas()

        spectrum = MsmsSpectrum(
            spectrum_specs.identifier.values[0],
            spectrum_specs.precursor_mz.values[0],
            spectrum_specs.precursor_charge.values[0],
            spectrum_specs.mz.values[0],
            spectrum_specs.intensity.values[0]
        )
        spectrum.peptide = spectrum_specs.peptide.values[0]
        spectrum._annotation = spectrum_specs.annotation.values[0]
        spectrum.is_decoy = spectrum_specs.is_decoy.values[0]

        return spectrum

    # Batch operations
    def write_spectra_to_library(self, spectra : Dict) -> None:
        """
        Stores multiple spectra in the spectral library for future retrieval.

        Parameters
        ----------
        spectra: Dict
            A dictionary containing spectrum data to be stored.
            The expected structure depends on the database format.

        """
        # Persist to lanceDB
        self.store.add(spectra)

    def read_spectra_from_library(self, spec_ids: List[str], charge: int = None) -> List[
        MsmsSpectrum]:
        """
        Fetches a batch of spectra from the library based on a list of IDs and
        an optional charge filter, and returns them as a list of MsmsSpectrum objects.

        Parameters
        ----------
        spec_ids : List[str]
            A list of library spectrum IDs.
        charge : int, optional
            An optional charge to filter the spectra. If None, all charges are included.

        Returns
        -------
        List[MsmsSpectrum]
            A list of MsmsSpectrum objects corresponding to the provided IDs.
        """

        if spec_ids is None or len(spec_ids) == 0:
            return []

        # Construct the filter for the query
        id_filter = ', '.join(f"'{id}'" for id in spec_ids)
        query = f"identifier in ({id_filter})"
        if charge is not None:
            query += f" and precursor_charge = {charge}"

        # Fetch data from the store
        spectra_data = self.store.search().limit(len(spec_ids)).select(
            columns=[
                "identifier", "precursor_mz", "precursor_charge",
                "mz", "intensity", "peptide", "annotation", "is_decoy"
            ]).where(query).to_pandas()

        # Construct and return a list of MsmsSpectrum objects
        spectra = []
        for _, row in spectra_data.iterrows():
            spectrum = MsmsSpectrum(
                identifier=row["identifier"],
                precursor_mz=row["precursor_mz"],
                precursor_charge=row["precursor_charge"],
                mz=row["mz"],
                intensity=row["intensity"]
            )
            spectrum.peptide = row["peptide"]
            spectrum._annotation = row["annotation"]
            spectrum.is_decoy = row["is_decoy"]
            spectra.append(spectrum)

        return spectra


    def fetch_projections_in_batch(self, spec_ids: List[str], charge)-> pd:
        """
        Fetches the projection vectors for a batch of library spectra based
        on their identifiers and charge state.

        Parameters
        ----------
        spec_ids: List[str]
            A list of spectrum identifiers to query.
        charge: int
            The precursor charge state to filter the spectra.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the retrieved projections.
        """
        filter = ', '.join(f"'{id}'" for id in spec_ids)
        spectra_projections = self.store.search().limit(len(spec_ids)).select(columns=[
            "identifier", "projection"]).where(f"precursor_charge = {charge}  and identifier in ({filter})").to_pandas()

        return spectra_projections


def verify_extension(supported_extensions: List[str], filename: str) -> None:
    """
    Check that the given file name has a supported extension.

    Parameters
    ----------
    supported_extensions : List[str]
        A list of supported file extensions.
    filename : str
        The file name to be checked.

    Raises
    ------
    FileNotFoundError
        If the file name does not have one of the supported extensions.
    """
    _, ext = os.path.splitext(os.path.basename(filename))
    if ext.lower() not in supported_extensions:
        logging.error('Unrecognized file format: %s', filename)
        raise FileNotFoundError(f'Unrecognized file format (supported file '
                                f'formats: {", ".join(supported_extensions)})')
    elif not os.path.isfile(filename):
        logging.error('File not found: %s', filename)
        raise FileNotFoundError(f'File {filename} does not exist')


def read_mzml(source: Union[IO, str]) -> Iterator[MsmsSpectrum]:
    """
    Get the MS/MS spectra from the given mzML file.

    Parameters
    ----------
    source : Union[IO, str]
        The mzML source (file name or open file object) from which the spectra
        are read.

    Returns
    -------
    Iterator[MsmsSpectrum]
        An iterator over the requested spectra in the given file.
    """
    with mzml.MzML(source) as f_in:
        try:
            for i, spectrum in enumerate(f_in):
                if int(spectrum.get('ms level', -1)) == 2:
                    try:
                        parsed_spectrum = _parse_spectrum_mzml(spectrum)
                        parsed_spectrum.index = i
                        parsed_spectrum.is_processed = False
                        yield parsed_spectrum
                    except ValueError as e:
                        logger.warning(f'Failed to read spectrum %s: %s',
                                       spectrum['id'], e)
        except LxmlError as e:
            logger.warning('Failed to read file %s: %s', source, e)


def _parse_spectrum_mzml(spectrum_dict: Dict) -> MsmsSpectrum:
    """
    Parse the Pyteomics spectrum dict.

    Parameters
    ----------
    spectrum_dict : Dict
        The Pyteomics spectrum dict to be parsed.

    Returns
    -------
    MsmsSpectrum
        The parsed spectrum.

    Raises
    ------
    ValueError: The spectrum can't be parsed correctly:
        - Unknown scan number.
        - Not an MS/MS spectrum.
        - Unknown precursor charge.
    """
    spectrum_id = spectrum_dict['id']

    if 'scan=' in spectrum_id:
        scan_nr = int(spectrum_id[spectrum_id.find('scan=') + len('scan='):])
    elif 'index=' in spectrum_id:
        scan_nr = int(spectrum_id[spectrum_id.find('index=') + len('index='):])
    else:
        raise ValueError(f'Failed to parse scan/index number')

    if int(spectrum_dict.get('ms level', -1)) != 2:
        raise ValueError(f'Unsupported MS level {spectrum_dict["ms level"]}')


    mz_array = spectrum_dict['m/z array']
    intensity_array = spectrum_dict['intensity array']
    retention_time = spectrum_dict['scanList']['scan'][0]['scan start time']

    precursor = spectrum_dict['precursorList']['precursor'][0]
    precursor_ion = precursor['selectedIonList']['selectedIon'][0]
    precursor_mz = precursor_ion['selected ion m/z']
    if 'charge state' in precursor_ion:
        precursor_charge = int(precursor_ion['charge state'])
    elif 'possible charge state' in precursor_ion:
        precursor_charge = int(precursor_ion['possible charge state'])
    else:
        precursor_charge = 0
    spectrum = MsmsSpectrum(str(scan_nr), precursor_mz, precursor_charge,
                            mz_array, intensity_array, None, retention_time)

    return spectrum

def read_mzxml(source: Union[IO, str]) -> Iterator[MsmsSpectrum]:
    """
    Get the MS/MS spectra from the given mzXML file.

    Parameters
    ----------
    source : Union[IO, str]
        The mzXML source (file name or open file object) from which the spectra
        are read.

    Returns
    -------
    Iterator[MsmsSpectrum]
        An iterator over the requested spectra in the given file.
    """
    with mzxml.MzXML(source) as f_in:
        try:
            for i, spectrum in enumerate(f_in):
                if int(spectrum.get('msLevel', -1)) == 2:
                    try:
                        parsed_spectrum = _parse_spectrum_mzxml(spectrum)
                        parsed_spectrum.index = i
                        parsed_spectrum.is_processed = False
                        yield parsed_spectrum
                    except ValueError as e:
                        logger.warning(f'Failed to read spectrum %s: %s',
                                       spectrum['id'], e)
        except LxmlError as e:
            logger.warning('Failed to read file %s: %s', source, e)


def _parse_spectrum_mzxml(spectrum_dict: Dict) -> MsmsSpectrum:
    """
    Parse the Pyteomics spectrum dict.

    Parameters
    ----------
    spectrum_dict : Dict
        The Pyteomics spectrum dict to be parsed.

    Returns
    -------
    MsmsSpectrum
        The parsed spectrum.

    Raises
    ------
    ValueError: The spectrum can't be parsed correctly:
        - Not an MS/MS spectrum.
        - Unknown precursor charge.
    """
    scan_nr = int(spectrum_dict['id'])

    if int(spectrum_dict.get('msLevel', -1)) != 2:
        raise ValueError(f'Unsupported MS level {spectrum_dict["msLevel"]}')

    mz_array = spectrum_dict['m/z array']
    intensity_array = spectrum_dict['intensity array']
    retention_time = spectrum_dict['retentionTime']

    precursor_mz = spectrum_dict['precursorMz'][0]['precursorMz']
    if 'precursorCharge' in spectrum_dict['precursorMz'][0]:
        precursor_charge = spectrum_dict['precursorMz'][0]['precursorCharge']
    else:
        precursor_charge = 0

    spectrum = MsmsSpectrum(str(scan_nr), precursor_mz, precursor_charge,
                            mz_array, intensity_array, None, retention_time)

    return spectrum


def _leading_substitute_pattern(match: re.Match) -> str:
    """
    Takes a match object as its argument and returns the replacement string.

    Parameters
    ----------
    match : Match
        Match object in the input string according to the pattern.

    Returns
    -------
    str
        Modified string.
    """
    if match.group(1) and match.group(2):
        return '[{}]?[{}]-{:s}'.format(match.group(1), match.group(2),
                                       match.group(3))
    elif match.group(1):
        return '[{}]-{}'.format(match.group(1), match.group(3))
    else:
        return match.group(0)


def _mgf_seq_to_proforma(peptide: str) -> str:
    """
    Takes a peptide in MassIVE-KB spectral library format to return a
    modified peptide in its ProForma format.

    Parameters
    ----------
    peptide : str
        Peptide sequence in its MassIVE-KB spectral library format.

    Returns
    -------
    str
        Modified mgf peptide in its ProForma format.
    """
    # Handle modifications with an observed experimental mass
    within_modification_pattern = r'([A-Z])([+-]?\d+\.\d+)'
    substitute_pattern = r'\1[\2]'
    formated_sequence = re.sub(within_modification_pattern,
                                   substitute_pattern, peptide)

    # Handle N-terminal or Unlocalized modifications
    leading_modification_pattern = r'([+-]?[\d.]+)([+-]?[\d.]+)?([A-Za-z]+)'

    # Handle leading modifications
    formated_sequence = re.sub(leading_modification_pattern,
                                   _leading_substitute_pattern,
                                   formated_sequence)

    return formated_sequence

def read_mgf(filename: str) -> Iterator[MsmsSpectrum]:
    """
    Read all spectra from the given mgf file.

    Parameters
    ----------
    filename: str
        The mgf file name from which to read the spectra.

    Returns
    -------
    Iterator[Spectrum]
        An iterator of spectra in the given mgf file.
    """
    # Get all spectra.
    with mgf.MGF(filename) as file:
        for i, mgf_spectrum in enumerate(file, 1):
            # Create spectrum.
            identifier = mgf_spectrum['params'][
                'title' if 'title' in mgf_spectrum['params'] else 'scan']

            precursor_mz = float(mgf_spectrum['params']['pepmass'][0])
            retention_time = float(mgf_spectrum['params']['rtinseconds']) if\
                'rtinseconds' in mgf_spectrum['params'] else None
            if 'charge' in mgf_spectrum['params']:
                precursor_charge = int(mgf_spectrum['params']['charge'][0])
            else:
                precursor_charge = 0

            spectrum = MsmsSpectrum(identifier, precursor_mz, precursor_charge,
                                    mgf_spectrum['m/z array'],
                                    mgf_spectrum['intensity array'],
                                    retention_time=retention_time)
            spectrum.index = i
            spectrum.is_processed = False
            spectrum.is_decoy = True if 'decoy' in mgf_spectrum['params'] \
                else False

            if 'seq' in mgf_spectrum['params']:
                spectrum.peptide = _mgf_seq_to_proforma(mgf_spectrum[
                                                            'params']['seq'])
                spectrum._annotation = [None] * len(mgf_spectrum['m/z array'])

            yield spectrum


def read_query_file(filename: str) -> Iterator[MsmsSpectrum]:
    """
    Read all spectra from the given mgf, mzml, or mzxml file.

    Parameters
    ----------
    filename: str
        The peak file name from which to read the spectra.

    Returns
    -------
    Iterator[Spectrum]
        An iterator of spectra in the given mgf file.
    """
    verify_extension(['.mgf', '.mzml', '.mzxml'],
                     filename)

    _, ext = os.path.splitext(os.path.basename(filename))

    if ext == '.mgf':
        return read_mgf(filename)
    elif ext == '.mzml':
        return read_mzml(filename)
    elif ext == '.mzxml':
        return read_mzxml(filename)


def read_fasta(filename: str) -> Iterator[MsmsSpectrum]:
    """
    Read protein sequences from a FASTA file, and process them to predict
    both target and decoy peptide spectra using Prosit.

    Parameters
    ----------
    filename: str
        The FASTA file name from which to read the protein sequences.

    Returns
    -------
    Iterator[MsmsSpectrum]
        An iterator of processed spectra.
    """
    proteins = [
        protein.sequence for protein in fasta.read(filename)
    ]

    ## Get all peptides based on desired protease
    _peptides = set().union(
        *[
            parser.cleave(protein, config.protease, config.missed_cleavages)
            for protein in proteins
        ]
    )
    logging.info(
        'Number of target peptides identified in  the sequence database is = '
        '%d',
        len(_peptides))
    ## Discard peptides with unknown residues
    valid_residues = set("KLYAGIEVCMWDPNFRSHQT")
    _peptides = [
        peptide for peptide in _peptides if all(p in valid_residues for p in peptide)
    ]
    logging.info(
        'Number of target peptides kept after discarding peptides with '
        'unknown '
        'residues is = %d',
        len(_peptides))
    ## Initialize lists to pass to Prosit
    _peptides_size = len(_peptides)
    peptides = []
    precursor_charges = []
    collision_energies = []
    for collision_energy in config.collision_energies:
        for precursor_charge in range(config.min_precursor_charge,
                                      config.max_precursor_charge + 1):
            peptides.extend(_peptides)
            collision_energies.extend([collision_energy] * _peptides_size)
            precursor_charges.extend([precursor_charge] * _peptides_size)

    ## Generate predictions for target peptides
    for batch_id, target_peptides_batch in enumerate(get_predictions(
                            peptides, precursor_charges, collision_energies)):
        offset = batch_id * config.prosit_batch_size
        for idx, intensities in enumerate(target_peptides_batch['intensities']):
            spectrum = MsmsSpectrum(str(offset + idx),
                                    mass.fast_mass(sequence=target_peptides_batch['peptide_sequences'][idx],
                                                   ion_type="M",
                                                   charge=target_peptides_batch['precursor_charges'][idx]),
                                    target_peptides_batch['precursor_charges'][idx],
                                    target_peptides_batch['mz'][idx],
                                    intensities)

            spectrum.peptide = target_peptides_batch['peptide_sequences'][idx]
            spectrum._annotation = [utils._parse_fragment_annotation(
                annotation.decode('utf-8')) for annotation in
                target_peptides_batch['annotation'][idx]]
            spectrum.is_decoy = False
            yield spectrum

    ## Generate predictions for decoy peptides
    decoy_peptides, decoy_precursor_charges, decoy_collision_energies = [], [], []
    target_peptides = set(peptides) # Set for O(1) of check of existing target peptide
    for peptide, charge, col_energie in zip(peptides, precursor_charges, collision_energies):
        decoy_peptide = _shuffle(peptide)[0]
        if decoy_peptide and decoy_peptide not in target_peptides:
            decoy_peptides.append(decoy_peptide)
            decoy_precursor_charges.append(charge)
            decoy_collision_energies.append(col_energie)

    for batch_id, decoy_peptides_batch in enumerate(get_predictions(
                            decoy_peptides, decoy_precursor_charges, decoy_collision_energies, True)):
        offset = batch_id * config.prosit_batch_size
        for idx, intensities in enumerate(decoy_peptides_batch['intensities']):
            spectrum = MsmsSpectrum('DECOY_' + str(offset + idx),
                                    mass.fast_mass(sequence=
                                                   decoy_peptides_batch[
                                                       'peptide_sequences'][
                                                       idx],
                                                   ion_type="M",
                                                   charge=
                                                   decoy_peptides_batch[
                                                       'precursor_charges'][
                                                       idx]),
                                    decoy_peptides_batch['precursor_charges'][
                                        idx],
                                    decoy_peptides_batch['mz'][idx],
                                    intensities)

            spectrum.peptide = decoy_peptides_batch['peptide_sequences'][idx]
            spectrum._annotation = [utils._parse_fragment_annotation(
                annotation.decode('utf-8')) for annotation in
                decoy_peptides_batch['annotation'][idx]]
            spectrum.is_decoy = True
            yield spectrum


def read_mztab_ssms(filename: str) -> pd.DataFrame:
    """
    Read SSMs from the given mzTab file.

    Parameters
    ----------
    filename: str
        The mzTab file name from which to read the SSMs.

    Returns
    -------
    pd.DataFrame
        A data frame containing the SSM information from the mzTab file.
    """
    verify_extension(['.mztab'], filename)

    # Skip the header lines.
    skiplines = 0
    with open(filename) as f_in:
        line = next(f_in)
        while line.split('\t', 1)[0] != 'PSH':
            line = next(f_in)
            skiplines += 1

    ssms = pd.read_csv(filename, sep='\t', header=skiplines,
                       index_col='PSM_ID')
    ssms.drop('PSH', 1, inplace=True)

    ssms['opt_ms_run[1]_cv_MS:1002217_decoy_peptide'] =\
        ssms['opt_ms_run[1]_cv_MS:1002217_decoy_peptide'].astype(bool)

    ssms.df_name = os.path.splitext(os.path.basename(filename))[0]

    return ssms
