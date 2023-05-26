# -*- coding: utf-8 -*-
"""Classifier Experiments: code to run experiments as an alternative to orchestration.

This file is configured for runs of the main method with command line arguments, or for
single debugging runs. Results are written in a standard format.
"""

__author__ = ["TonyBagnall"]

import os

os.environ["MKL_NUM_THREADS"] = "1"  # must be done before numpy import!!
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # must be done before numpy import!!
os.environ["OMP_NUM_THREADS"] = "1"  # must be done before numpy import!!

import sys

import numba
from aeon.clustering.k_shapes import TimeSeriesKShapes
from aeon.datasets import load_from_tsfile as load_ts

from aeon.clustering.experiment.dba_experiment import get_distance_defaults, _results_present_full_path, run_clustering_experiment
if __name__ == "__main__":
    """Example simple usage, with args input via script or hard coded for testing."""
    numba.set_num_threads(1)

    tune = False
    normalise = True
    if (
            sys.argv is not None and sys.argv.__len__() > 1
    ):  # cluster run, this is fragile, requires all args atm
        data_dir = sys.argv[1]
        results_dir = sys.argv[2]
        dataset = sys.argv[3]
        resample = int(sys.argv[4])
        init = sys.argv[5]
        if len(sys.argv) > 6:
            normalise = sys.argv[6].lower() == "true"
    else:  # Local run
        print(" Local Run")  # noqa
        data_dir = "/home/chris/Documents/Datasets/Univariate_ts/"
        dataset = "Chinatown"
        init = "random"
        results_dir = "/home/chris/Documents/Results/temp/"
        resample = 0
        normalise = True

    if _results_present_full_path(results_dir + "/", dataset, resample):
        print(
            f"Ignoring dataset{dataset}, results already present at {results_dir}"
        )  # noqa
    else:
        print(  # noqa
            f" Running {dataset} resample {resample} normalised = {normalise} "  # noqa
            f"results path = {results_dir}"
        )  # noqa

    train_X, train_Y = load_ts(f"{data_dir}/{dataset}/{dataset}_TRAIN.ts")
    test_X, test_Y = load_ts(f"{data_dir}/{dataset}/{dataset}_TEST.ts")
    test_X = test_X.squeeze()
    train_X = train_X.squeeze()

    if normalise:
        from sklearn.preprocessing import StandardScaler

        s = StandardScaler()
        train_X = s.fit_transform(train_X.T)
        train_X = train_X.T
        test_X = s.fit_transform(test_X.T)
        test_X = test_X.T

    clst = TimeSeriesKShapes(
        n_clusters=len(set(train_Y)),
        init_algorithm=init,
        n_init=10,
        max_iter=300,
        random_state=resample + 1,
    )
    print(f"Running kshapes with {clst.get_params()}")  # noqa
    run_clustering_experiment(
        train_X,
        clst,
        results_path=results_dir,
        trainY=train_Y,
        testX=test_X,
        testY=test_Y,
        cls_name="kshapes",
        dataset_name=dataset,
        resample_id=resample,
        overwrite=False,
    )
    print("done")  # noqa
