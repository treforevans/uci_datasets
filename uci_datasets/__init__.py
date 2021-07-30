import numpy as np
import os
import pandas
from typing import Tuple


# save some global variables
# Identify the small, intermediate, and large datasets from Yang et al., 2016 by the
# name of the dataset reported in that paper.
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
# also Identify all of the datasets, note that it is just in alphabetical order
all_datasets = [
    "3droad",
    "autompg",
    "bike",
    "challenger",
    "concreteslump",
    "energy",
    "forest",
    "houseelectric",
    "keggdirected",
    "kin40k",
    "parkinsons",
    "pol",
    "pumadyn32nm",
    "slice",
    "solar",
    "stock",
    "yacht",
    "airfoil",
    "autos",
    "breastcancer",
    "buzz",
    "concrete",
    "elevators",
    "fertility",
    "gas",
    "housing",
    "keggundirected",
    "machine",
    "pendulum",
    "protein",
    "servo",
    "skillcraft",
    "sml",
    "song",
    "tamielectric",
    "wine",
]


class Dataset:
    def __init__(self, dataset: str, dtype=np.float64, print_stats: bool = True):
        """
        load dataset

        Inputs:
            dataset: string
                name of the dataset to load. This can be either the name of the directory
                that the dataset is in OR the identifier used in papers. For example you can
                specify dataset='houseelectric' OR dataset='electric' and it will give you the
                same thing. This allows for convienent abbreviations.
            print_stats: if true then will print stuff about the dataset

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
            self.test_mask = np.loadtxt(
                fname=path + "/" + dataset + "/test_mask.csv.gz",
                dtype=bool,
                delimiter=",",
            )
            data = np.loadtxt(
                fname=path + "/" + dataset + "/data.csv.gz", dtype=dtype, delimiter=",",
            )
        except:
            print("Load failed, maybe dataset string is not correct.")
            raise

        # generate the train_mask
        # train_mask and test mask are opposite, i.e. test_mask = np.logical_not(train_mask)
        self.train_mask = np.logical_not(self.test_mask)

        # extract the inputs and reponse
        self.x = data[:, :-1]
        self.y = data[:, -1, None]

        # print stats
        if print_stats:
            print(
                "%s dataset, N=%d, d=%d" % (dataset, self.x.shape[0], self.x.shape[1])
            )

    def get_split(
        self, split: int = 0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the test and train points for the specified split.
        Inputs:
            split : (int) index of the requested split. There are 10 test train splits 
            for each dataset so this value can be any integer from 0 to 9 (inclusive).

        Outputs:
            x_train: training dataset inputs/features. Numpy ndarray of size (n,d).
            y_train: training dataset outputs/responses. Numpy ndarray of size (n,1).
            x_test: testing dataset inputs/features. Numpy ndarray of size (m,d).
            y_test: testing dataset outputs/responses. Numpy ndarray of size (m,1).
        """
        assert isinstance(split, int)
        assert split >= 0
        assert split < 10
        x_test = self.x[self.test_mask[:, split], :]
        x_train = self.x[self.train_mask[:, split], :]
        y_test = self.y[self.test_mask[:, split], :]
        y_train = self.y[self.train_mask[:, split], :]
        return x_train, y_train, x_test, y_test


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

