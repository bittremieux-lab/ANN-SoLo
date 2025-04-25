import logging
import os
import sys
from typing import List, Union

# Initialize all random seeds before importing any packages.
from ann_solo import rndm
rndm.set_seeds()

from ann_solo import spectral_library
from ann_solo import writer
from ann_solo.config import config


def ann_solo(spectral_library_filename: str, query_filename: str,
             out_filename: str, **kwargs: Union[bool, float, int, str]) -> int:
    """
    Run ANN-SoLo with the specified search settings.

    Values for search settings that are not explicitly specified will be taken
    from the config file (if present) or take their default values.

    The identified PSMs will be stored in the given file.

    Parameters
    ----------
    spectral_library_filename : str
        The spectral library file name.
    query_filename : str
        The query spectra file name.
    out_filename : str
        The mzTab output file name.
    **kwargs : Union[bool, float, int, str]
        Additional search settings. Keys MUST match the command line
        arguments (excluding the '--' prefix;
        https://github.com/bittremieux/ANN-SoLo/wiki/Parameters). Values
        MUST be the argument values. Boolean flags can be toggled by
        specifying True or False (ex: no_gpu=True).

    Returns
    -------
    int
        The error code from running ANN-SoLo.
    """
    # Convert kwargs dictionary to list for main().
    # 'args' contains arguments with values.
    # 'flags' contains boolean flags to include
    args = sum([['--' + k, str(v)] for k, v in kwargs.items()
                if not isinstance(v, bool)], [])
    flags = ['--' + k for k, v in kwargs.items() if v and isinstance(v, bool)]

    # Explicitly set the search parameters when run from Python.
    error_code = main([spectral_library_filename, query_filename, out_filename,
                       *args, *flags])

    return error_code


def main(args: Union[str, List[str]] = None) -> int:
    # Configure logging.
    logging.captureWarnings(True)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(
        '{asctime} {levelname} [{name}/{processName}] {module}.{funcName} : '
        '{message}', style='{'))
    root.addHandler(handler)
    # Disable dependency non-critical log messages.
    logging.getLogger('faiss').setLevel(logging.WARNING)
    logging.getLogger('mokapot').setLevel(logging.WARNING)
    logging.getLogger('numba').setLevel(logging.WARNING)
    logging.getLogger('numexpr').setLevel(logging.WARNING)

    # Lance spilling config
    # Note that the LANCE_BYPASS_SPILLING environment variable can be used to
    # bypass spilling to disk. Setting this to true can avoid memory exhaustion issues.
    os.environ["LANCE_BYPASS_SPILLING"] = "true"

    # Load the configuration.
    config.parse(args)
    # Perform the search.
    spec_lib = spectral_library.SpectralLibrary(
        config.spectral_library_filename)
    identifications = spec_lib.search(config.query_filename)
    writer.write_mztab(identifications, config.out_filename,
                       spec_lib._library_reader)
    spec_lib.shutdown()

    logging.shutdown()

    return 0


if __name__ == '__main__':
    # Use search parameters from sys.argv when run from CMD.
    main()
