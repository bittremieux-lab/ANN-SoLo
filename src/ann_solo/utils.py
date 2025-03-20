from __future__ import annotations

import logging
import re
from typing import Dict, Iterator, List, Optional, Union

import mokapot
import numpy as np
import pandas as pd
import scipy.signal
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectorMixin, VarianceThreshold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from spectrum_utils.utils import mass_diff

from ann_solo import spectrum_similarity as sim
from ann_solo.config import config
from ann_solo.spectrum import SpectrumSpectrumMatch, MsmsSpectrum
from spectrum_utils.fragment_annotation import FragmentAnnotation

class CorrelationThreshold(SelectorMixin, BaseEstimator):
    """
    Feature selector that removes correlated features.

    This feature selection algorithm looks only at the features (X), not the
    desired outputs (y), and can thus be used for unsupervised learning.

    Parameters
    ----------
    threshold : float
        For pairwise features with a training-set absolute correlation
        lower than this threshold, one of the features will be removed. The
        default is to keep all features that are not perfectly correlated,
        i.e. remove the features that are identical or opposite in all
        samples.
    """

    def __init__(self, threshold: float = None) -> None:
        self.threshold = threshold if threshold is not None else 1.0

    def fit(self, X, y=None) -> CorrelationThreshold:
        """
        Learn empirical correlations from X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Data from which to compute correlations, where `n_samples` is the
            number of samples and `n_features` is the number of features.
        y : any, default=None
            Ignored. This parameter exists only for compatibility with
            sklearn.pipeline.Pipeline.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        corr = np.abs(np.corrcoef(X, rowvar=False))
        self.mask = ~(np.tril(corr, k=-1) > self.threshold).any(axis=1)
        return self

    def _get_support_mask(self):
        return self.mask


def score_ssms(
    ssms: List[SpectrumSpectrumMatch],
    fdr: float,
    model: Optional[str],
    grouped: bool = False,
    min_group_size: int = 100,
) -> List[SpectrumSpectrumMatch]:
    """
    Score SSMs using semi-supervised learning with mokapot.

    Parameters
    ----------
    ssms : List[SpectrumSpectrumMatch]
        SSMs to be scored.
    fdr : float
        The minimum FDR threshold to accept target SSMs.
    model : Optional[str]
        The type of machine learning model to use. Can be "rf" for a random
        forest classifier, "svm" for a Percolator-like linear SVM, or `None`
        to disable semi-supervised learning.
    grouped : bool
        Compute q-values per SSM group or not (default: False).
    min_group_size : int
        The minimum group size (default: 100). SSMs in smaller groups are
        combined into a residual group. Must be specified if `grouped` is True.

    Returns
    -------
    Iterator[SpectrumSpectrumMatch]
        An iterator of the SSMs with assigned scores and q-values.
    """
    logging.debug(
        "Compute features for semi-supervised scoring from %d SSMs", len(ssms)
    )
    # Create a Dataset with SSM features.
    features = pd.DataFrame(_compute_ssm_features(ssms))
    # Remove invalid feature values (infinity).
    for col in [
        "mse_mz",
        "mse_int",
        "mse_mz_top5",
        "mse_int_top5",
        "manhattan",
        "euclidean",
        "chebyshev",
        "canberra",
    ]:
        mask = np.isfinite(features[col])
        features.loc[~mask, col] = features.loc[mask, col].max()
    features["group"] = _get_ssm_groups(ssms, min_group_size) if grouped else 0
    dataset = mokapot.dataset.LinearPsmDataset(
        features,
        target_column="is_target",
        spectrum_columns="index",
        peptide_column="sequence",
        group_column="group",
    )
    if grouped:
        logging.debug(
            "Partitioned %d SSMs into %d groups",
            len(ssms),
            features["group"].nunique(),
        )
    # Define the mokapot model.
    #   - Choice between a random forest, linear SVM, or no semi-supervised
    #     learning.
    #   - Features are preprocessed by standardizing them, removing
    #     zero-variance features, and removing highly correlated features.
    #   - We perform minimal tuning of the class weights and the random forest
    #     tree depth.
    if model is None:
        logging.debug("Calculate q-values based on the cosine similarity")
        # Calculate q-values based on the cosine similarity.
        confidences = dataset.assign_confidence(features["cosine"], True)
    else:
        logging.debug(
            "Train semi-supervised %s model and score SSMs", model.upper()
        )
        scaler = make_pipeline(
            StandardScaler(),
            VarianceThreshold(),
            CorrelationThreshold(0.95),
        )
        if model == "svm":
            clf = mokapot.model.PercolatorModel(
                scaler, train_fdr=fdr, direction="cosine"
            )
        elif model == "rf":
            clf = mokapot.Model(
                GridSearchCV(
                    RandomForestClassifier(random_state=1),
                    param_grid={
                        "max_depth": [3, 5, 7, 9, None],
                        "class_weight": [
                            None,
                            {0: 0.1, 1: 1},
                            {0: 0.1, 1: 10},
                            {0: 1, 1: 0.1},
                            {0: 1, 1: 10},
                            {0: 10, 1: 0.1},
                            {0: 10, 1: 1},
                        ],
                    },
                    refit=False,
                    cv=3,
                    n_jobs=-1,
                ),
                scaler,
                train_fdr=fdr,
                direction="cosine",
            )
        else:
            raise ValueError(
                "Unknown semi-supervised machine learning model given"
            )
        # Train the mokapot model and combine the SSMs for all groups.
        confidences, _ = mokapot.brew(dataset, clf, fdr)
    ssm_scores = pd.concat(
        [
            confidences.group_confidence_estimates[group].psms
            for group in confidences.groups
        ],
        ignore_index=True,
    )
    # Assign the scores to the SSM objects.
    for i, score, q in zip(
        ssm_scores["index"],
        ssm_scores["mokapot score"],
        ssm_scores["mokapot q-value"],
    ):
        ssms[i].search_engine_score = score
        ssms[i].q = q
    return ssms


def _get_ssm_groups(
    ssms: List[SpectrumSpectrumMatch], min_group_size: int
) -> pd.Series:
    """
    Get SSM group labels based on the precursor mass differences.

    SSMs are grouped by finding peaks in histograms centered around each
    nominal mass difference and assigning SSMs to the nearest peak.

    Parameters
    ----------
    ssms : Iterator[SpectrumSpectrumMatch]
        SSMs to be grouped.
    min_group_size : int
        The minimum group size. SSMs in smaller groups are combined into a
        residual group.

    Returns
    -------
    pd.Series
        A Series with group labels for all SSMs.
    """
    # Group SSMs in a window around each nominal mass difference.
    mass_diffs = np.asarray(
        [
            (ssm.exp_mass_to_charge - ssm.calc_mass_to_charge) * ssm.charge
            for ssm in ssms
        ]
    )
    order = np.argsort(mass_diffs)
    groups, group = -np.ones(len(ssms), np.int32), 0
    group_md, group_i = np.nan, []
    for counter, (md, i) in enumerate(zip(mass_diffs[order], order)):
        if round(md) != group_md or counter == len(mass_diffs) - 1:
            if round(md) == group_md:
                group_i.append(i)
            if len(group_i) > 0:
                # Assign groups within the current interval.
                # Create a mass difference histogram and find peaks.
                bins = np.linspace(group_md - 0.5, group_md + 0.5, 101)
                hist, _ = np.histogram(mass_diffs[group_i], bins=bins)
                peaks_bin_i, prominences = scipy.signal.find_peaks(
                    hist, prominence=(None, None)
                )
                if len(peaks_bin_i) > 0:
                    # Assign mass differences to their closest peak.
                    for md_j, j in zip(mass_diffs[group_i], group_i):
                        peak_assignment = -1, np.inf
                        for peak_i, peak in enumerate(bins[peaks_bin_i]):
                            distance_to_peak = abs(peak - md_j)
                            if (
                                bins[prominences["left_bases"][peak_i]]
                                < md_j
                                < bins[prominences["right_bases"][peak_i]]
                                and distance_to_peak < peak_assignment[1]
                            ):
                                peak_assignment = peak_i, distance_to_peak
                        if peak_assignment[0] != -1:
                            groups[j] = group + peak_assignment[0]
                group += len(peaks_bin_i)
            # Start a new interval.
            group_i = []
        group_i.append(i)
        group_md = round(md)
    groups = pd.Series(groups)
    # Reassign small groups to a residual group (accurate FDR calculation is
    # not possible in overly small groups).
    group_counts = groups.value_counts()
    groups[groups.isin(group_counts.index[group_counts < min_group_size])] = -1
    return groups


def _compute_ssm_features(
    ssms: Iterator[SpectrumSpectrumMatch],
) -> Dict[str, Union[bool, float, int, str]]:
    """
    Compute spectrum-spectrum match features.

    Parameters
    ----------
    ssms : Iterator[SpectrumSpectrumMatch]
        The SSMs whose features are computed.

    Returns
    -------
    Dict[str, Union[bool, float, int, str]]
        A dictionary with features for all SSMs. The dictionary must contain
        "index", "sequence", "is_target", and "group" keys with the
        corresponding SSM metadata. Additional keys contain SSM features.
    """
    features = {
        "index": [],
        "sequence": [],
        "sequence_len": [],
        "precursor_charge_2": [],
        "precursor_charge_3": [],
        "precursor_charge_4": [],
        "precursor_charge_5": [],
        "query_prec_mz": [],
        "lib_prec_mz": [],
        "mz_diff_ppm": [],
        "abs_mz_diff_ppm": [],
        "mz_diff_da": [],
        "abs_mz_diff_da": [],
        "cosine": [],
        "cosine_top5": [],
        "n_matched_peaks": [],
        "frac_n_peaks_query": [],
        "frac_n_peaks_lib": [],
        "frac_n_peaks_lib_top5": [],
        "frac_int_query": [],
        "frac_int_lib": [],
        "frac_int_lib_top5": [],
        "mse_mz": [],
        "mse_mz_top5": [],
        "mse_int": [],
        "mse_int_top5": [],
        "contrast_angle": [],
        "contrast_angle_top5": [],
        "hypergeometric_score": [],
        "kendalltau": [],
        "ms_for_id_v1": [],
        "ms_for_id_v2": [],
        "entropy_unweighted": [],
        "entropy_weighted": [],
        "scribe_fragment_acc": [],
        "scribe_fragment_acc_top5": [],
        "manhattan": [],
        "euclidean": [],
        "chebyshev": [],
        "pearsonr": [],
        "pearsonr_top5": [],
        "spearmanr": [],
        "spearmanr_top5": [],
        "braycurtis": [],
        "canberra": [],
        "ruzicka": [],
        "is_target": [],
    }
    for i, ssm in enumerate(ssms):
        if len(ssm.peak_matches) == 0:
            continue

        sim_calc = sim.SpectrumSimilarityCalculator(ssm)
        sim_calc_top = sim.SpectrumSimilarityCalculator(ssm, 5)

        features["index"].append(i)
        features["sequence"].append(ssm.sequence)
        features["sequence_len"].append(len(ssm.sequence))
        if ssm.query_spectrum.precursor_charge <= 2:
            features["precursor_charge_2"].append(1)
            features["precursor_charge_3"].append(0)
            features["precursor_charge_4"].append(0)
            features["precursor_charge_5"].append(0)
        elif ssm.query_spectrum.precursor_charge == 3:
            features["precursor_charge_2"].append(0)
            features["precursor_charge_3"].append(1)
            features["precursor_charge_4"].append(0)
            features["precursor_charge_5"].append(0)
        elif ssm.query_spectrum.precursor_charge == 4:
            features["precursor_charge_2"].append(0)
            features["precursor_charge_3"].append(0)
            features["precursor_charge_4"].append(1)
            features["precursor_charge_5"].append(0)
        elif ssm.query_spectrum.precursor_charge >= 5:
            features["precursor_charge_2"].append(0)
            features["precursor_charge_3"].append(0)
            features["precursor_charge_4"].append(0)
            features["precursor_charge_5"].append(1)
        features["query_prec_mz"].append(ssm.query_spectrum.precursor_mz)
        features["lib_prec_mz"].append(ssm.library_spectrum.precursor_mz)
        features["mz_diff_ppm"].append(
            mass_diff(
                ssm.query_spectrum.precursor_mz,
                ssm.library_spectrum.precursor_mz,
                False,
            )
        )
        features["abs_mz_diff_ppm"].append(
            abs(
                mass_diff(
                    ssm.query_spectrum.precursor_mz,
                    ssm.library_spectrum.precursor_mz,
                    False,
                )
            )
        )
        features["mz_diff_da"].append(
            mass_diff(
                ssm.query_spectrum.precursor_mz,
                ssm.library_spectrum.precursor_mz,
                True,
            )
        )
        features["abs_mz_diff_da"].append(
            abs(
                mass_diff(
                    ssm.query_spectrum.precursor_mz,
                    ssm.library_spectrum.precursor_mz,
                    True,
                )
            )
        )
        features["cosine"].append(sim_calc.cosine())
        features["cosine_top5"].append(sim_calc_top.cosine())
        features["n_matched_peaks"].append(sim_calc.n_matched_peaks())
        features["frac_n_peaks_query"].append(sim_calc.frac_n_peaks_query())
        features["frac_n_peaks_lib"].append(sim_calc.frac_n_peaks_library())
        features["frac_n_peaks_lib_top5"].append(
            sim_calc_top.frac_n_peaks_library()
        )
        features["frac_int_query"].append(sim_calc.frac_intensity_query())
        features["frac_int_lib"].append(sim_calc.frac_intensity_library())
        features["frac_int_lib_top5"].append(
            sim_calc_top.frac_intensity_library()
        )
        features["mse_mz"].append(sim_calc.mean_squared_error("mz"))
        features["mse_mz_top5"].append(sim_calc_top.mean_squared_error("mz"))
        features["mse_int"].append(sim_calc.mean_squared_error("intensity"))
        features["mse_int_top5"].append(
            sim_calc_top.mean_squared_error("intensity")
        )
        features["contrast_angle"].append(sim_calc.spectral_contrast_angle())
        features["contrast_angle_top5"].append(
            sim_calc_top.spectral_contrast_angle()
        )
        features["hypergeometric_score"].append(
            sim_calc.hypergeometric_score(
                min_mz=config.min_mz,
                max_mz=config.max_mz,
                fragment_mz_tol=config.bin_size,
            )
        )
        features["kendalltau"].append(sim_calc.kendalltau())
        features["ms_for_id_v1"].append(sim_calc.ms_for_id_v1())
        features["ms_for_id_v2"].append(sim_calc.ms_for_id_v2())
        features["entropy_unweighted"].append(sim_calc.entropy(False))
        features["entropy_weighted"].append(sim_calc.entropy(True))
        features["scribe_fragment_acc"].append(sim_calc.scribe_fragment_acc())
        features["scribe_fragment_acc_top5"].append(
            sim_calc_top.scribe_fragment_acc()
        )
        features["manhattan"].append(sim_calc.manhattan())
        features["euclidean"].append(sim_calc.euclidean())
        features["chebyshev"].append(sim_calc.chebyshev())
        features["pearsonr"].append(sim_calc.pearsonr())
        features["pearsonr_top5"].append(sim_calc_top.pearsonr())
        features["spearmanr"].append(sim_calc.spearmanr())
        features["spearmanr_top5"].append(sim_calc_top.spearmanr())
        features["braycurtis"].append(sim_calc.braycurtis())
        features["canberra"].append(sim_calc.canberra())
        features["ruzicka"].append(sim_calc.ruzicka())
        features["is_target"].append(not ssm.is_decoy)
    return features


def _parse_fragment_annotation(annotation: str) -> \
        FragmentAnnotation:
    """
    Takes an ion peak anotaion line and parse to retrieve: ion_type,
    ion_index, and charge.

    Parameters
    ----------
    annotation : str
        Raw annotation line.

    Returns
    -------
    FragmentAnnotation
        An FragmentAnnotation object.
    """
    ion_type = annotation[0]
    if ion_type in 'abyp':
        index_charge = annotation[1:].split('/', 1)[0].split('^')
        ion_index = re.search(r'^\d+', index_charge[0])
        if len(index_charge) == 1:
            charge, ion_index = (
            1 if ion_index.group(0) == index_charge[0] else -1,
            int(ion_index.group(0)))
        else:
            charge = re.search(r'^\d+', index_charge[1])
            charge, ion_index = int(
                charge.group(0)) if charge else -1, int(
                ion_index.group(0)) if ion_index else 1
        return FragmentAnnotation(str(ion_type)+str(ion_index),
                                  charge=abs(charge))
    else:
        return None
