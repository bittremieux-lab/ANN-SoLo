import argparse
import textwrap

import configargparse

class NewlineTextHelpFormatter(argparse.HelpFormatter):

    def _fill_text(self, text, width, indent):
        return '\n'.join(
            textwrap.fill(line, width, initial_indent=indent,
                          subsequent_indent=indent,
                          replace_whitespace=False).strip()
            for line in text.splitlines(keepends=True))


class Config:
    """
    Spectral library configuration.

    Configuration settings can be specified in a config.ini file (by default in
    the working directory), or as command-line arguments.
    """

    def __init__(self) -> None:
        """
        Initialize the configuration settings and provide sensible default
        values if possible.
        """

        self._parser = configargparse.ArgParser(
            description='ANN-SoLo: Approximate nearest neighbor spectral '
                        'library searching\n'
                        '================================================'
                        '=================\n\n'
                        'Bittremieux et al. Fast open modification spectral '
                        'library searching through approximate nearest '
                        'neighbor indexing. Journal of Proteome Research 17, '
                        '3464-3474 (2018).\n\nBittremieux et al. Extremely '
                        'fast and accurate open modification spectral library '
                        'searching of high-resolution mass spectra using '
                        'feature hashing and graphics processing units. '
                        'Journal of Proteome Research 18, 3792-3799 (2019).'
                        '\n\nOfficial code website: '
                        'https://github.com/bittremieux/ANN-SoLo\n\n',
            default_config_files=['config.ini'],
            args_for_setting_config_path=['-c', '--config'],
            formatter_class=NewlineTextHelpFormatter)

        # IO
        self._parser.add_argument(
            'spectral_library_filename', help='spectral library file '
                                              '(supported formats: splib)')

        self._parser.add_argument(
            'query_filename', help='query file (supported formats: mgf, '
                                   'mzml, mzxml)')

        self._parser.add_argument(
            'out_filename',
            help='name of the mzTab output file containing the search results')

        # PREPROCESSING
        # Spectral library resolution to round mass values.
        self._parser.add_argument(
            '--resolution', default=None, type=int,
            help='spectral library resolution; masses will be rounded to the '
                 'given number of decimals (default: no rounding)')

        # Minimum and maximum fragment peak mass values.
        self._parser.add_argument(
            '--min_mz', default=11, type=int,
            help='minimum m/z value (inclusive, default: %(default)s m/z)')
        self._parser.add_argument(
            '--max_mz', default=2010, type=int,
            help='maximum m/z value (inclusive, default: %(default)s m/z)')

        # Remove peaks around the precursor mass from fragment spectra.
        self._parser.add_argument(
            '--remove_precursor', action='store_true',
            help='remove peaks around the precursor mass '
                 '(default: no peaks are removed)')
        self._parser.add_argument(
            '--remove_precursor_tolerance', default=0, type=float,
            help='the window (in m/z) around the precursor mass to remove '
                 'peaks (default: %(default)s m/z)')

        # Minimum fragment peak intensity to filter out noise peaks.
        self._parser.add_argument(
            '--min_intensity', default=0.01, type=float,
            help='remove peaks with a lower intensity relative to the maximum '
                 'intensity (default: %(default)s)')

        # Minimum number of fragment peaks or mass range (m/z).
        self._parser.add_argument(
            '--min_peaks', default=10, type=int,
            help='discard spectra with less peaks (default: %(default)s)')
        self._parser.add_argument(
            '--min_mz_range', default=250, type=float,
            help='discard spectra with a smaller mass range '
                 '(default: %(default)s m/z)')

        # Maximum number of fragment peaks to use for each query spectrum.
        self._parser.add_argument(
            '--max_peaks_used', default=50, type=int,
            help='only use the specified most intense peaks for the query '
                 'spectra (default: %(default)s)')

        # Maximum number of fragment peaks to use for each library spectrum.
        self._parser.add_argument(
            '--max_peaks_used_library', default=50, type=int,
            help='only use the specified most intense peaks for the library '
                 'spectra (default: %(default)s)')

        # Manner in which to scale the peak intensities.
        self._parser.add_argument(
            '--scaling', default='rank', type=str,
            choices=['sqrt', 'rank'],
            help='to reduce the influence of very intense peaks, scale the'
                 ' peaks by their square root or by their rank '
                 '(default: %(default)s)')

        # MATCHING
        # Maximum SSM precursor mass tolerance.
        self._parser.add_argument(
            '--precursor_tolerance_mass', type=float, required=True,
            help='precursor mass tolerance (small window for the first level '
                 'of the cascade search)')
        self._parser.add_argument(
            '--precursor_tolerance_mode', type=str, choices=['Da', 'ppm'],
            required=True,
            help='precursor mass tolerance unit (options: %(choices)s)')
        self._parser.add_argument(
            '--precursor_tolerance_mass_open', type=float,
            help='precursor mass tolerance (wide window for the second level '
                 'of the cascade search)')
        self._parser.add_argument(
            '--precursor_tolerance_mode_open', type=str,
            choices=['Da', 'ppm'],
            help='precursor mass tolerance unit (options: %(choices)s)')

        # Fragment peak matching.
        self._parser.add_argument(
            '--fragment_mz_tolerance', type=float, required=True,
            help='fragment mass tolerance (m/z)')

        # Shifted dot product.
        self._parser.add_argument(
            '--allow_peak_shifts', action='store_true',
            help='use the shifted dot product instead of the standard dot '
                 'product')

        # Maximum FDR.
        self._parser.add_argument(
            '--fdr', default=0.01, type=float,
            help='FDR threshold to accept identifications during the cascade '
                 'search (default: %(default)s)')
        # Machine learning model to use for SSM scoring.
        self._parser.add_argument(
            '--model', default="rf", type=str, choices=["rf", "svm", "none"],
            help='machine learning model for spectrum match scoring: "none" '
                 'for no semi-supervised learning, "svm" for a Percolator-like'
                 ' linear SVM, "rf" for random forest (default: %(default)s)')
        # Minimum group size for sub-group FDR for open search results.
        self._parser.add_argument(
            '--fdr_min_group_size', default=100, type=int,
            help='minimum group size for the group FDR calculation during the '
                 'second cascade level (default: %(default)s)')

        # MODE
        # Use an ANN index or the conventional brute-force mode.
        self._parser.add_argument(
            '--mode', default='ann', type=str, choices=['ann', 'bf'],
            help="search using an approximate nearest neighbors or the "
                 "traditional (brute-force) mode; 'bf': brute-force, 'ann': "
                 "approximate nearest neighbors (default: %(default)s)")

        # Bin size for the ANN index (Dalton).
        self._parser.add_argument(
            '--bin_size', default=0.04, type=float,
            help='ANN vector bin width (default: %(default)s Da)')
        # ANN vector length after hashing.
        self._parser.add_argument(
            '--hash_len', default=800, type=int,
            help='ANN vector length (default: %(default)s)')

        # Number of candidates to retrieve from the ANN index for each query.
        self._parser.add_argument(
            '--num_candidates', default=1024, type=int,
            help='number of candidates to retrieve from the ANN index for each'
                 ' query (default: %(default)s), maximum 1024 when using GPU '
                 'indexing')

        # Number of query spectra to process simultaneously.
        self._parser.add_argument(
            '--batch_size', default=16384, type=int,
            help='number of query spectra to process simultaneously '
                 '(default: %(default)s)')

        # Custom FAISS parameters.
        # Number of lists in the IVF.
        self._parser.add_argument(
            '--num_list', default=256, type=int,
            help='number of partitions in the ANN index '
                 '(default: %(default)s)')
        # Number of lists to inspect during querying.
        self._parser.add_argument(
            '--num_probe', default=128, type=int,
            help='number of partitions in the ANN index to inspect during '
                 'querying (default: %(default)s), maximum 1024 when using '
                 'GPU indexing')
        # Don't try to use the GPU.
        self._parser.add_argument(
            '--no_gpu', action='store_true',
            help="don't use the GPU for ANN searching (default: GPU is used "
                 "if available)")

        # Add decoys.
        self._parser.add_argument(
            '--add_decoys', action='store_true',
            help="generate decoys of the spectral library ")

        self._parser.add_argument(
            '--fragment_tol_mode', type=str, choices=['Da', 'ppm'], default='ppm',
            help='Fragment mass tolerance unit. Either (options: %(choices)s)')

        # Prosit
        # Prosit batch size.
        self._parser.add_argument(
            '--prosit_batch_size', default=1000, type=int,
            help='maximum number of peptides to generate predictions for '
                 'using prosit server (default: %(default)s)')
        # Prosit server URL.
        self._parser.add_argument(
            '--prosit_server_url', default="koina.proteomicsdb.org:443",
            type=str, help='koina server url (default: %(default)s)')
        # Prosit model name.
        self._parser.add_argument(
            '--prosit_model_name', default="Prosit_2020_intensity_HCD",
            type=str,
            help='Prosit model name to use for predictions (default: %('
                 'default)s)')
        # min_precursor_charge.
        self._parser.add_argument(
            '--min_precursor_charge', type=int, default=2,
            help='min precursor charge value used for spectra Prosit '
                 'predictions (default: %(default)s)')
        # max_precursor_charge.
        self._parser.add_argument(
            '--max_precursor_charge', type=int, default=3,
            help='max precursor charge value used for spectra Prosit '
                 'predictions (default: %(default)s)')
        # Collision energies.
        self._parser.add_argument(
            '--collision_energies', nargs='+', type=int, default=[32],
            help='list of collision energies used by prosit to generate '
                 'spectrum predictions of given peptides (default: %(default)s)')
        # missed cleavages.
        self._parser.add_argument(
            '--missed_cleavages', type=int, default=2,
            help='missed cleavages value used to cleave the protein sequence '
                 'and retrieve peptides (default: %(default)s)')
        # protease.
        self._parser.add_argument(
            '--protease', type=str, default='trypsin',
            help='protease used to cleave the protein sequence '
                 'and retrieve peptides (default: %(default)s)')

        # Filled in 'parse', contains the specified settings.
        self._namespace = None

    def parse(self, args_str: str = None) -> None:
        """
        Parse the configuration settings.

        Parameters
        ----------
            args_str : str
                If None, the arguments are taken from sys.argv. Arguments that
                are not explicitly specified are taken from the configuration
                file.
        """
        self._namespace = vars(self._parser.parse_args(args_str))

    def __getattr__(self, option):
        if self._namespace is None:
            raise RuntimeError('The configuration has not been initialized')
        return self._namespace[option]

    def __getitem__(self, item):
        return self.__getattr__(item)


config = Config()
