import os
import matplotlib.pyplot as plt
import numpy as np
from aeon.clustering.metrics.averaging._dba import dba
from aeon.datasets import load_from_tsfile as load_ts

if __name__ == "__main__":

    dataset = "Beef"
    c = "1"
    plt.figure()

    fig, axs = plt.subplots(4, 2, sharey=True, sharex=True, figsize=(8, 6))

    data_dir = "/home/chris/Documents/Datasets/Univariate_ts/"

    x_train, y_train = load_ts(f"{data_dir}/{dataset}/{dataset}_TRAIN.ts")

    # test_x, test_y = load_from_tsfile(
    #     os.path.join(f"../../../../ajb/Data/{dataset}/{dataset}_TEST.ts")
    # )
    _, _, len_ts = x_train.shape

    x = range(0, len_ts)

    idxs = np.where(y_train == c)

    for i in x_train[idxs]:
        axs[0, 0].plot(x, i[0], lw=0.2)
        axs[1, 0].plot(x, i[0], lw=0.2)
        axs[2, 0].plot(x, i[0], lw=0.2)
        axs[3, 0].plot(x, i[0], lw=0.2)


    series_avg = np.mean(np.array(x_train[idxs]), axis=0)[0]

    axs[0, 1].plot(x, series_avg, color="red")

    series_mba = dba(
        x_train[idxs],
        metric="msm"
    )

    axs[1, 1].plot(x, series_mba[0, :])

    series_twe = dba(
        x_train[idxs],
        metric="twe"
    )

    axs[2, 1].plot(x, series_twe[0, :])

    series_dba = dba(
        x_train[idxs],
        metric="dtw"
    )

    axs[3, 1].plot(x, series_dba[0, :])



    fig.suptitle(f"{dataset} - Class {c}")
    axs[0, 0].set_title("Original time series")
    axs[0, 1].set_title("Averaging")
    axs[1, 0].set_title("Original time series")
    axs[1, 1].set_title("MBA")
    axs[2, 0].set_title("Original time series")
    axs[2, 1].set_title("TBA")
    axs[3, 0].set_title("Original time series")
    axs[3, 1].set_title("DBA")
    fig.tight_layout()
    fig.show()