Regression datasets with test-train splits from the [UCI machine learning repository](https://archive.ics.uci.edu).

# Setup
Simply clone and setup:
```bash
git clone git@github.com:treforevans/uci_datasets.git
cd uci_datasets
python setup.py develop
```
*Note that you must use `develop` in the above line, not `install`.*

# Usage
The following code gets the first test-train split (i.e., `split=0`) of the `challenger` dataset:
```python
from uci_datasets import Dataset
data = Dataset("challenger")
x_train, y_train, x_test, y_test = data.get_split(split=0)
```
There are 10 test train splits for each dataset (as in 10-fold cross validation) with 90% of the dataset being training points and 10% being testing points in each split.
The `split` parameter of `get_split` method accepts integers from 0 to 9 (inclusive).
The dataset can be referenced by the name of its respective folder in the repository.
