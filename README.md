Regression datasets from the [UCI machine learning repository](https://archive.ics.uci.edu) prepared for benchmarking studies with test-train splits.

# Setup
This repository uses [git large file storage](https://git-lfs.github.com/) so you must first [install git LFS](https://github.com/git-lfs/git-lfs/wiki/Installation) otherwise the cloned repo will only contain pointer files rather than the data files.
The size of the repository is about 319 Mb.

After installing git LFS you can simply clone and setup the python package as follows:
```bash
git clone https://github.com/treforevans/uci_datasets.git
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
There are 10 test-train splits for each dataset (as in 10-fold cross validation) with 90% of the dataset being training points and 10% being testing points in each split.
The `split` parameter of the `Dataset.get_split` method accepts integers from 0 to 9 (inclusive).
The dataset can be retrieved by the name of its respective folder in the repository.
A list of all the dataset names can be obtained as follows:
```python
from uci_datasets import all_datasets # list of all dataset names
```

# Papers using these datasets
The following papers use the same datasets and test-train splits present in this repository.
* [Yang, Zichao, et al. "A la carte–learning fast kernels." Artificial Intelligence and Statistics. 2015.](https://proceedings.mlr.press/v38/yang15b.html)
* [Wilson, Andrew Gordon, et al. "Deep kernel learning." Artificial Intelligence and Statistics. 2016.](https://proceedings.mlr.press/v51/wilson16.html)
* [Evans, Trefor W., and Prasanth B. Nair. "Scalable Gaussian processes with grid-structured eigenfunctions (GP-GRIEF)." International Conference on Machine Learning. 2018.](https://arxiv.org/abs/1807.02125)
* [Evans, Trefor W., and Prasanth B. Nair. "Discretely relaxing continuous variables for tractable variational inference." Neural Information Processing Systems. 2018.](https://arxiv.org/abs/1809.04279)
* [Audouze, Christophe, and Prasanth B. Nair. "Sparse low-rank separated representation models for learning from data." Proceedings of the Royal Society A 475.2221 (2019).](https://royalsocietypublishing.org/doi/full/10.1098/rspa.2018.0490)
* [Evans, Trefor W., and Prasanth B. Nair. "Quadruply stochastic gaussian processes." arXiv preprint arXiv:2006.03015 (2020).](https://arxiv.org/abs/2006.03015)

