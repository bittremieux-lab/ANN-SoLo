import collections
import copy
import hashlib
import json
import logging
import math
import multiprocessing
import os
from typing import Dict
from typing import Iterator
from typing import List

import faiss
import numexpr as ne
import numpy as np
import tqdm
from spectrum_utils.spectrum import MsmsSpectrum

from ann_solo import reader
from ann_solo import utils
from ann_solo.config import config
from ann_solo.spectrum import process_spectrum
from ann_solo.spectrum import spectrum_to_vector
from ann_solo.spectrum import SpectrumSpectrumMatch
from ann_solo.spectrum_match import modified_cosine

class SpectralLibrary:
    """
    Spectral library search engine.

    The spectral library search engine identifies unknown query spectra by
    comparing each query spectrum against candidate spectra with a known
    peptide identity in the spectral library. The query spectrum is assigned
    the peptide sequence as its best matching library spectrum.
    """

    # Hyperparameters used to initialize the spectral library.
    _hyperparameters = ['min_mz', 'max_mz', 'bin_size', 'hash_len', 'num_list']

    # File names of the ANN indices for each charge.
    _ann_filenames = {}

    # Lock to allow only a single process to access the active ANN index.
    _ann_index_lock = multiprocessing.Lock()

    def __init__(self, filename: str) -> None:
        """
        Create a spectral library from the given spectral library file.

        New ANN indexes for every charge in the spectral library are created if
        they don't exist yet for the current ANN configuration.

        Parameters
        ----------
        filename : str
            The spectral library file name.

        Raises
        ------
        FileNotFoundError: The given spectral library file wasn't found or
            isn't supported.
        """
        # Library Store
        try:
            self._library_reader = reader.SpectralLibraryReader(
                filename, self._get_hyperparameter_hash())

        except FileNotFoundError as e:
            logging.error(e)
            raise

        self._num_probe = config.num_probe
        self._num_candidates = config.num_candidates
        self._use_gpu = not config.no_gpu and faiss.get_num_gpus()
        if self._use_gpu:
            self._res = faiss.StandardGpuResources()
            # GPU indexes can only handle maximum 1024 probes and neighbors.
            # https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU#limitations
            if self._num_probe > 1024:
                logging.warning('Using num_probe=1024 (maximum supported '
                                'value on the GPU), %d was supplied',
                                self._num_probe)
                self._num_probe = 1024
            if self._num_candidates > 1024:
                logging.warning('Using num_candidates=1024 (maximum supported '
                                'value on the GPU), %d was supplied',
                                self._num_candidates)
                self._num_candidates = 1024

        self._current_index = None, None

        if config.mode == 'ann':
            verify_file_existence = True
            if self._library_reader.is_recreated:
                logging.warning('ANN indexes were created using '
                                'non-compatible settings')
                verify_file_existence = False
            # Check if an ANN index exists for each charge.
            base_filename = f'{os.path.splitext(filename)[0]}_' \
                            f'{self._get_hyperparameter_hash()[:7]}'
            create_ann_charges = []
            # No need to build an ANN index for infrequent precursor charges.
            ann_charges = [charge for charge, charge_info in
                           self._library_reader.spec_info['charge'].items()
                           if len(charge_info['id']) >= config.num_list]
            for charge in sorted(ann_charges):
                self._ann_filenames[charge] = (
                    f'{base_filename}_{charge}.idxann'
                )
                if (not verify_file_existence or
                        not os.path.isfile(self._ann_filenames[charge])):
                    create_ann_charges.append(charge)
                    logging.warning('Missing ANN index for charge %d', charge)

            # Create the missing FAISS indices.
            if create_ann_charges:
                self._create_ann_indexes(create_ann_charges)



    def _get_hyperparameter_hash(self) -> str:
        """
        Get a unique string representation of the hyperparameters used to
        initialize the spectral library.

        Returns
        -------
        str
            A hexadecimal hashed string representing the initialization
            hyperparameters.
        """
        hyperparameters_bytes = json.dumps(
            {hp: config[hp] for hp in self._hyperparameters}).encode('utf-8')
        return hashlib.sha1(hyperparameters_bytes).hexdigest()

    def _create_ann_indexes(self, charges: List[int]) -> None:
        """
        Create FAISS indexes for fast ANN candidate selection.

        Parameters
        ----------
        charges : List[int]
            Charges for which a FAISS index will be created. Sufficient library
            spectra with the corresponding precursor charge should exist.
        """
        logging.debug('Add the spectra to the spectral library ANN indexes')
        # Collect vectors for all spectra per charge.
        charge_vectors = {
            charge: np.zeros((len(self._library_reader.spec_info
                                  ['charge'][charge]['id']), config.hash_len),
                             np.float32)
            for charge in charges}
        # Max batch size to pull data in batches during index creation
        batch_size = 100000
        for charge in tqdm.tqdm(charge_vectors.keys(),
                                desc='Library charge processed', leave=False,
                                unit='charge',
                                smoothing=0.1):
            logging.info(
                'Fetch spectra projections for charge - %d',
                charge)

            # Get the IDs for the current charge
            charge_ids = self._library_reader.spec_info['charge'][charge]['id']
            total_ids = len(charge_ids)
            # Calculate the number of batches
            num_batches = math.ceil(total_ids / batch_size)

            for batch_idx in tqdm.tqdm(range(num_batches),
                                       desc=f'Processing batches for charge {charge}',
                                       leave=False, unit='batch',
                                       total=num_batches,
                                       smoothing=0.1):
                # Determine the start and end of the current batch
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, total_ids)
                # Fetch the spectra projections for the current batch
                batch_ids = charge_ids[start_idx:end_idx]
                spectra_projection = self._library_reader.read_spectra_projections(
                    batch_ids, charge)
                # Insert the batch projections
                charge_vectors[charge][start_idx:end_idx,
                :] = spectra_projection

        # Build an individual FAISS index per charge.
        logging.info('Build the spectral library ANN indexes')
        for charge, vectors in charge_vectors.items():
            logging.debug('Create a new ANN index for charge %d', charge)
            quantizer = faiss.IndexFlatIP(config.hash_len)
            # TODO: Use HNSW as quantizer?
            #       https://github.com/facebookresearch/faiss/blob/master/benchs/bench_hnsw.py#L136
            # quantizer = faiss.IndexHNSWFlat(config.hash_len, 32)
            # quantizer.hnsw.efSearch = 64
            # ann_index -> faiss.METRIC_L2
            # ann_index.quantizer_trains_alone = 2
            ann_index = faiss.IndexIVFFlat(quantizer, config.hash_len,
                                           config.num_list,
                                           faiss.METRIC_INNER_PRODUCT)
            # noinspection PyArgumentList
            ann_index.train(vectors)
            # noinspection PyArgumentList
            ann_index.add(vectors)
            faiss.write_index(ann_index, self._ann_filenames[charge])

        logging.debug('Finished creating the spectral library ANN indexes')

    def shutdown(self) -> None:
        """
        Release any resources to gracefully shut down.
        """
        if self._current_index[1] is not None:
            self._current_index[1].reset()

    def search(self, query_filename: str) -> List[SpectrumSpectrumMatch]:
        """
        Identify all unknown spectra in the given query file.

        Parameters
        ----------
        query_filename : str
            The query file name.

        Returns
        -------
        List[SpectrumSpectrumMatch]
            A list of identified `SpectrumSpectrumMatch`es between the query
            spectra and library spectra below the given FDR threshold
            (specified in the config).
        """
        logging.info('Process file %s', query_filename)
        # Read all spectra in the query file and
        # split based on their precursor charge
        query_spectra = collections.defaultdict(list)
        for query_spectrum in tqdm.tqdm(
                reader.read_query_file(query_filename), desc='Query spectra '
                    'read',leave=False, unit='spectra', smoothing=0.7):
            # For queries with an unknown charge, try all possible charges.
            if query_spectrum.precursor_charge != 0:
                query_spectra_charge = [query_spectrum]
            else:
                query_spectra_charge = []
                for charge in (2, 3):
                    query_spectra_charge.append(copy.copy(query_spectrum))
                    query_spectra_charge[-1]._inner.precursor_charge = charge
                    query_spectra_charge[-1].is_processed = False
                    query_spectra_charge[-1].index = query_spectrum.index

            for query_spectrum_charge in query_spectra_charge:
                # Discard low-quality spectra.
                processed_query = process_spectrum(query_spectrum_charge, False)
                if processed_query.is_valid:
                    query_spectra[processed_query.precursor_charge].append(
                        processed_query)
        # Identify all query spectra.
        identifications = {}
        do_cascade_open = (
            config.precursor_tolerance_mass_open is not None and
            config.precursor_tolerance_mode_open is not None
        )
        # Cascade level 1: standard search.
        n_spectra_identified = 0
        for ssm in self._search_cascade(query_spectra, 'std'):
            # Only retain confidently identified spectra (below FDR threshold)
            # if we're doing a cascade (open) search.
            if not do_cascade_open or ssm.q < config.fdr:
                identifications[ssm.query_identifier] = ssm
                n_spectra_identified += ssm.q < config.fdr

        logging.info('%d spectra identified after the standard search',
                     n_spectra_identified)
        if do_cascade_open:
            # Collect the remaining query spectra for the second cascade level.
            for charge, query_spectra_charge in query_spectra.items():
                query_spectra[charge] = [
                    spectrum for spectrum in query_spectra_charge
                    if spectrum.identifier not in identifications]
            # Cascade level 2: open search.
            # No FDR filtering necessary here to have an entry for each query
            # spectrum in the output.
            for ssm in self._search_cascade(query_spectra, 'open'):
                identifications[ssm.query_identifier] = ssm
                n_spectra_identified += ssm.q < config.fdr
            logging.info('%d spectra identified after the open search',
                         n_spectra_identified)
        return list(identifications.values())

    def _search_cascade(self, query_spectra: Dict[int, List[MsmsSpectrum]],
                        mode: str) -> Iterator[SpectrumSpectrumMatch]:
        """
        Perform a single level of the cascade search.

        Parameters
        ----------
        query_spectra : Dict[int, List[Spectrum]]
            A dictionary with as keys the different charges and as values lists
            of query spectra for each charge.
        mode : {'std', 'open'}
            The search mode. Either 'std' for a standard search with a small
            precursor mass window, or 'open' for an open search with a wide
            precursor mass window.


        Returns
        -------
        Iterator[SpectrumSpectrumMatch]
            An iterator of spectrum-spectrum matches that are below the FDR
            threshold (specified in the config).
        """
        num_spectra = sum([len(q) for q in query_spectra.values()])
        if mode == 'std':
            logging.debug('Process %d query spectra using a standard search '
                          '(Δm = %s %s)',
                          num_spectra,
                          config.precursor_tolerance_mass,
                          config.precursor_tolerance_mode)
        elif mode == 'open':
            logging.debug('Process %d query spectra using an open search '
                          '(Δm = %s %s)',
                          num_spectra,
                          config.precursor_tolerance_mass_open,
                          config.precursor_tolerance_mode_open)
        ssms = {}
        batch_size = config.batch_size
        with tqdm.tqdm(desc='Query spectra processed', total=num_spectra,
                       leave=False, unit='spectra', smoothing=0.1) as pbar:
            for charge, query_spectra_charge in query_spectra.items():
                for batch_i in range(0, len(query_spectra_charge), batch_size):
                    query_spectra_batch =\
                        query_spectra_charge[batch_i:
                                             min(batch_i + batch_size,
                                                 len(query_spectra_charge))]
                    for ssm in self._search_batch(query_spectra_batch, charge,
                                                  mode):
                        # Make sure we only retain the best identification
                        # (i.e. in case of duplicated spectra
                        # if the precursor charge was unknown).
                        if (ssm is not None and
                                (ssm.query_identifier not in ssms or
                                 (ssm.search_engine_score >
                                  ssms[ssm.query_identifier].search_engine_score))):
                            ssms[ssm.query_identifier] = ssm
                        pbar.update(1)
        # Store the SSMs below the FDR threshold.
        logging.info('Filter the spectrum—spectrum matches on FDR '
                     '(threshold = %s)', config.fdr)

        return utils.score_ssms(
            list(ssms.values()),
            config.fdr,
            config.model if config.model != "none" else None,
            mode == "open"
        )

    def _search_batch(self, query_spectra: List[MsmsSpectrum],
                      charge: int, mode: str)\
            -> Iterator[SpectrumSpectrumMatch]:
        """
        Generate spectrum-spectrum matches for a batch of query spectra with
        the same precursor charge.

        Parameters
        ----------
        query_spectra : List[Spectrum]
            The query spectra for which spectrum-spectrum matches are
            generated.
        charge : int
            The precursor charge of the query spectra.
        mode : {'std', 'open'}
            The search mode. Either 'std' for a standard search with a small
            precursor mass window, or 'open' for an open search with a wide
            precursor mass window.

        Returns
        -------
        Iterator[SpectrumSpectrumMatch]
            An iterator of spectrum-spectrum matches for every query spectrum
            that could be successfully matched to its most similar library
            spectrum.
        """

        # Find all library candidates for each query spectrum.
        for query_spectrum, library_candidates in zip(
                query_spectra, self._get_library_candidates(
                    query_spectra, charge, mode)):
            # Find the best match candidate.
            if library_candidates:
                library_match, score, peak_matches = \
                    self._get_best_match(
                        query_spectrum, library_candidates,
                        config.fragment_mz_tolerance
                    )
                yield SpectrumSpectrumMatch(
                    query_spectrum,
                    library_match,
                    peak_matches=np.asarray(peak_matches),
                    search_engine_score=score
                )


    def _get_best_match(self, query: MsmsSpectrum, candidates: list[
                        MsmsSpectrum], fragment_mz: float):
        """
        Find the best matching spectrum from a list of candidates using the
        modified cosine similarity.

        Parameters
        ----------
        query: MsmsSpectrum
            The query spectrum for which the best match is sought.
        candidates: list[MsmsSpectrum]
            A list of candidate spectra to compare against the query.
        fragment_mz: float
            Search fragment m/z tolerance for comparison.

        Returns:
            tuple: A tuple containing:
                - best_match: The spectrum from the candidates with the highest similarity score.
                - best_score: The highest cosine similarity score among the candidates.
                - peak_matches: A list of matched peak indices between the query and the best match.
        """	
        best_match, best_score, peak_matches = None, None, None
        for idx, candidate in enumerate(candidates):
            result = modified_cosine(query, candidate, fragment_mz)
            if best_match is None or result.score > best_score or (
                    result.score == best_score and best_match.is_decoy
            ):
                best_match = candidate
                best_score = result.score
                peak_matches = [[i, j] for i, j in zip(result.matched_indices,
                                                       result.matched_indices_other)]

        return best_match, best_score, peak_matches



    def _get_library_candidates(self, query_spectra: List[MsmsSpectrum],
                                charge: int, mode: str)\
            -> Iterator[List[MsmsSpectrum]]:
        """
        Get the library spectra to be matched against the query spectra.

        Parameters
        ----------
        query_spectra : List[Spectrum]
            The query spectra for which library candidates are retrieved.
        charge : int
            The precursor charge of the query spectra.
        mode : {'std', 'open'}
            The search mode. Either 'std' for a standard search with a small
            precursor mass window, or 'open' for an open search with a wide
            precursor mass window.

        Returns
        -------
        Iterator[List[Spectrum]]
            An iterator of lists of library candidate spectra for each query
            spectrum.

        Raises
        ------
        ValueError: Invalid search settings:
            - Unsupported search mode (either 'std' or 'open')
            - Unsupported precursor mass tolerance mode (either 'Da' or 'ppm')
        """
        if mode == 'std':
            tol_val = config.precursor_tolerance_mass
            tol_mode = config.precursor_tolerance_mode
        elif mode == 'open':
            tol_val = config.precursor_tolerance_mass_open
            tol_mode = config.precursor_tolerance_mode_open
        else:
            raise ValueError('Unknown search mode')

        # No library matches possible.
        if charge not in self._library_reader.spec_info['charge']:
            return

        library_candidates = self._library_reader.spec_info['charge'][charge]

        # First filter: precursor m/z.
        query_mzs = np.empty((len(query_spectra), 1), float)
        for i, query_spectrum in enumerate(query_spectra):
            query_mzs[i] = query_spectrum.precursor_mz
        library_mzs = library_candidates['precursor_mz'].reshape((1, -1))
        if tol_mode == 'Da':
            candidate_filters = ne.evaluate(
                'abs(query_mzs - library_mzs) * charge <= tol_val')
        elif tol_mode == 'ppm':
            candidate_filters = ne.evaluate(
                'abs(query_mzs - library_mzs) / library_mzs * 10**6'
                '<= tol_val')
        else:
            raise ValueError('Unknown precursor tolerance mode')

        # Second filter: ANN.
        if (config.mode == 'ann' and mode == 'open' and
                charge in self._ann_filenames):
            ann_index = self._get_ann_index(charge)
            query_vectors = np.zeros((len(query_spectra), config.hash_len),
                                     np.float32)
            for i, query_spectrum in enumerate(query_spectra):
                query_vectors[i] = spectrum_to_vector(
                    query_spectrum,
                    config.min_mz,
                    config.max_mz,
                    config.bin_size,
                    int(config.hash_len / 2)
                )
            mask = np.zeros_like(candidate_filters)

            for mask_i, ann_filter in zip(mask, ann_index.search(
                    query_vectors, self._num_candidates)[1]):
                valid_indices = ann_filter[ann_filter != -1]
                mask_i[valid_indices] = True

            candidate_filters = np.logical_and(candidate_filters, mask)

        # Get the library candidates that pass the filter.
        for candidate_filter in candidate_filters:
            yield self._library_reader.read_spectra_candidates(
                library_candidates['id'][candidate_filter], charge)


    def _get_ann_index(self, charge: int) -> faiss.IndexIVF:
        """
        Get the ANN index for the specified charge.

        This allows on-demand loading of the ANN indices and prevents having to
        keep a large amount of data for the index into memory.
        The previously used index is cached to avoid reloading the same index
        (only a single index is cached to prevent using an excessive amount of
        memory). If no index for the specified charge is cached yet, this index
        is loaded from the disk.

        To prevent loading the same index multiple times (incurring a
        significant performance quality) it is CRUCIAL that query spectrum
        processing is partitioned by charge so the previous index can be
        reused.

        Parameters
        ----------
        charge : int
            The charge for which the ANN index is retrieved.

        Returns
        -------
        faiss.IndexIVF
            The ANN index for the specified charge.
        """
        with self._ann_index_lock:
            if self._current_index[0] != charge:
                # Release memory reserved by the previous index.
                if self._current_index[1] is not None:
                    self._current_index[1].reset()
                # Load the new index.
                logging.debug('Load the ANN index for charge %d', charge)
                index = faiss.read_index(self._ann_filenames[charge])
                if self._use_gpu:
                    co = faiss.GpuClonerOptions()
                    co.useFloat16 = True
                    index = faiss.index_cpu_to_gpu(self._res, 0, index, co)
                    index.setNumProbes(self._num_probe)
                else:
                    index.nprobe = self._num_probe
                self._current_index = charge, index

            return self._current_index[1]
