from uci_datasets import Dataset, all_datasets


def test_datasets():
    # loop through all datasets, make sure they load and run some tests
    for dataset_name in all_datasets.keys():
        data = Dataset(dataset_name)
        x_train, y_train, x_test, y_test = data.get_split(split=0)
        assert x_train.shape[1] == x_test.shape[1]
        assert y_train.shape[1] == y_test.shape[1] == 1
        assert x_train.shape[0] == y_train.shape[0]
        assert x_test.shape[0] == y_test.shape[0]
