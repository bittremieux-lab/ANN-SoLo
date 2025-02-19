from typing import Dict, List

import logging
import numpy as np
import pandas as pd
import tqdm
from koinapy import Koina

from ann_solo.config import config

def get_predictions(peptides: List[str], precursor_charges: List[int],
                    collision_energies: List[int], decoy: bool = False) -> \
        Dict[str, np.ndarray]:
    """
    Predict spectra from the list of peptides.

    Parameters
    ----------
    peptides: List(str)
        List of peptides.
    precursor_charges: List(int)
        Synced list of precursor_charges.
    collision_energies: List(int)
        Synced list of collision_energies.
    decoy: bool = False
        Boolean precising whether the peptides are target or decoys.


    Returns
    -------
    Dict[str, np.ndarray]
        A dictionary of spectra for each peptide, particularly containing
        intensities,  mz,  annotations for each spectrum.
    """

    batch_size = config.prosit_batch_size
    len_inputs = len(peptides)
    model = Koina(config.prosit_model_name, config.prosit_server_url)
    max_retries = 3  # Number of retries for the server call
    for i in tqdm.tqdm(range(0, len_inputs, batch_size),
            desc='Prosit peptides batch prediction:',
            unit=('decoy' if decoy else 'target') + ' peptides'):

        inputs = pd.DataFrame(
            {
                "peptide_sequences": peptides[i:i + batch_size],
                "precursor_charges": precursor_charges[i:i + batch_size],
                "collision_energies": collision_energies[i:i + batch_size],
            }
        )

        attempt = 0
        while attempt < max_retries:
            try:
                koina_predictions = model.predict(inputs, debug=True)
                break  # If successful, exit the retry loop
            except Exception as e:
                attempt += 1
                if attempt == max_retries:
                    raise RuntimeError(f"Failed to get predictions after {max_retries} retries.") from e
                logging.info(
                    f"Retrying prediction (attempt {attempt}/{max_retries}) due to error: {e}")


        grouped_predictions = koina_predictions.groupby(
            ['peptide_sequences', 'precursor_charges', 'collision_energies']
        ).agg(
            {
                'intensities': list,
                'mz': list,
                'annotation': list
            }
        ).reset_index()

        predictions = grouped_predictions.to_dict(orient='list')

        yield predictions
