import numpy as np
import os
import pandas
from typing import Tuple


# save some global variables
small_datasets = [
    "challenger",
    "fertility",
    "slump",
    "automobile",
    "servo",
    "cancer",
    "hardware",
    "yacht",
    "autompg",
    "housing",
    "forest",
    "stock",
    "pendulum",
    "energy",
    "concrete",
    "solar",
    "airfoil",
    "wine",
]
intermediate_datasets = [
    "gas",
    "skillcraft",
    "sml",
    "parkinsons",
    "pumadyn",
    "poletele",
    "elevators",
    "kin40k",
    "protein",
    "kegg",
    "keggu",
    "ctslice",
]
large_datasets = ["3droad", "song", "buzz", "electric"]
all_datasets = small_datasets + intermediate_datasets + large_datasets


def load_dataset(
    dataset: str, print_stats: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    load dataset

    Inputs:
        dataset: string
            name of the dataset to load. This can be either the name of the directory
            that the dataset is in OR the identifier used in papers. For example you can
            specify dataset='houseelectric' OR dataset='electric' and it will give you the
            same thing. This allows for convienent abbreviations.
        print_stats: if true then will print stuff about the dataset

    Outputs:
        x: dataset inputs/features. Numpy ndarray of size (N,d).
        y: dataset outputs/responses. Numpy ndarray of size (N,1).
        test_mask: dataset test mask. Numpy bool ndarray of size (N,10).
        train_mask: dataset train mask. Numpy bool ndarray of size (N,10).

    Notes:
        * train_mask and test mask are opposite, i.e. test_mask = np.logical_not(train_mask)
    """
    assert isinstance(dataset, str), "dataset must be a string"
    dataset = dataset.lower()  # convert to lowercase
    dataset = dataset.replace(" ", "")  # remove whitespace
    dataset = dataset.replace("_", "")  # remove underscores

    # get the identifier to directory map (NOTE: this may be incomplete)
    id_map = {
        "slump": "concreteslump",
        "automobile": "autos",
        "cancer": "breastcancer",
        "hardware": "machine",
        "forestfires": "forest",
        "solarflare": "solar",
        "gassensor": "gas",
        "poletele": "pol",
        "kegg": "keggdirected",
        "keggu": "keggundirected",
        "ctslice": "slice",
        "electric": "houseelectric",
        "pumadyn": "pumadyn32nm",
    }
    if dataset in id_map:
        dataset = id_map[dataset]

    # get the directory this file is in and load the dataset
    path = os.path.split(os.path.abspath(__file__))[0] + "/../"
    try:
        test_mask = np.loadtxt(
            fname=path + "/" + dataset + "/test_mask.csv.gz", dtype=bool, delimiter=","
        )
        data = np.loadtxt(fname=path + "/" + dataset + "/data.csv.gz", delimiter=",")
    except:
        print("Load failed, maybe dataset string is not correct.")
        raise

    # generate the train_mask
    train_mask = np.logical_not(test_mask)

    # extract the inputs and reponse
    x = data[:, :-1]
    y = data[:, -1].reshape((-1, 1))

    # print stats
    if print_stats:
        print("%s dataset, N=%d, d=%d" % (dataset, x.shape[0], x.shape[1]))
    return x, y, test_mask, train_mask


def csv_results(
    fname,
    runstr,
    i_split,
    rmse=None,
    rmse_norm=None,
    time=None,
    notes=None,
    N=None,
    d=None,
):
    """
    save results to csv file

    Inputs:
        fname : csv filename to save the file to/append results to
        runstr : identifier for the current run. Typically relates to a dataset with specific parameter settings
        i_split : the index of the train/test split (0 to 9)
        rmse : root mean squared error on test set with un-normalized responses
        rmse_norm : root mean squared error on test set with normalized responses
        time : train time
        notes : any other notes you want to add
        N : number of points (typically including both the train and test set)
        d : input dimensionality

    Outputs:
        df : dataframe with results. Results are also saved to file.
    """
    # check if the csv file exists, and if not then create it
    columns = (
        ["N", "d", "Time", "RMSE", "RMSE Normalized", "Notes"]
        + ["time_%d" % i for i in range(10)]
        + ["rmse_%d" % i for i in range(10)]
        + ["rmse_norm_%d" % i for i in range(10)]
    )
    if os.path.isfile(fname):
        df = pandas.read_csv(fname, index_col=0)
    else:  # create a new dataframe
        df = pandas.DataFrame(columns=columns)

    # input the data
    if N is not None:
        df.loc[runstr, "N"] = "%d" % N
    if d is not None:
        df.loc[runstr, "d"] = "%d" % d
    if rmse is not None:
        df.loc[runstr, "rmse_%d" % i_split] = rmse
    if rmse_norm is not None:
        df.loc[runstr, "rmse_norm_%d" % i_split] = rmse_norm
    if time is not None:
        df.loc[runstr, "time_%d" % i_split] = time
    if notes is not None:
        df.loc[runstr, "Notes"] = notes

    # update the means and stds
    for pres_col, data_col in [("RMSE", "rmse"), ("RMSE Normalized", "rmse_norm")]:
        df.loc[runstr, pres_col] = "$%g \pm %g$" % (
            np.around(
                np.nanmean(
                    [df.loc[runstr, "%s_%d" % (data_col, i)] for i in range(10)]
                ),
                decimals=3,
            ),
            np.around(
                np.nanstd([df.loc[runstr, "%s_%d" % (data_col, i)] for i in range(10)]),
                decimals=3,
            ),
        )
    df.loc[runstr, "Time"] = "%g" % np.around(
        np.nanmean([df.loc[runstr, "time_%d" % i] for i in range(10)]), decimals=0
    )

    # save to file, do this in a robust way since often multiple people/servers write to the file at the same time which can cause issues
    n_failed = 0
    while True:
        try:
            df.to_csv(fname)
            break
        except:
            if n_failed > 10:
                raise
            else:
                n_failed += 1
    return df

