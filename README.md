# UCI datasets

Regression datasets from the [UCI machine learning repository](https://archive.ics.uci.edu) prepared for benchmarking studies with test-train splits.

## Installation

Install using pip (the download size is about 312 Mb):

```bash
python -m pip install git+https://github.com/treforevans/uci_datasets.git
```

## Usage

The following code gets the first test-train split (i.e., `split=0`) of the `challenger` dataset:

```python
from uci_datasets import Dataset
data = Dataset("challenger")
x_train, y_train, x_test, y_test = data.get_split(split=0)
```

There are 10 test-train splits for each dataset (as in 10-fold cross validation) with 90% of the dataset being training points and 10% being testing points in each split.
The `split` parameter of the `Dataset.get_split` method accepts integers from 0 to 9 (inclusive).

## Datasets

The below table contains the size (number of observations) and the number of input dimensions of each dataset.
All datasets have a single output dimension.

| Dataset name     | Number of observations | Input dimension |
| :--------------- | ---------------------: | --------------: |
| `3droad`         |                 434874 |               3 |
| `autompg`        |                    392 |               7 |
| `bike`           |                  17379 |              17 |
| `challenger`     |                     23 |               4 |
| `concreteslump`  |                    103 |               7 |
| `energy`         |                    768 |               8 |
| `forest`         |                    517 |              12 |
| `houseelectric`  |                2049280 |              11 |
| `keggdirected`   |                  48827 |              20 |
| `kin40k`         |                  40000 |               8 |
| `parkinsons`     |                   5875 |              20 |
| `pol`            |                  15000 |              26 |
| `pumadyn32nm`    |                   8192 |              32 |
| `slice`          |                  53500 |             385 |
| `solar`          |                   1066 |              10 |
| `stock`          |                    536 |              11 |
| `yacht`          |                    308 |               6 |
| `airfoil`        |                   1503 |               5 |
| `autos`          |                    159 |              25 |
| `breastcancer`   |                    194 |              33 |
| `buzz`           |                 583250 |              77 |
| `concrete`       |                   1030 |               8 |
| `elevators`      |                  16599 |              18 |
| `fertility`      |                    100 |               9 |
| `gas`            |                   2565 |             128 |
| `housing`        |                    506 |              13 |
| `keggundirected` |                  63608 |              27 |
| `machine`        |                    209 |               7 |
| `pendulum`       |                    630 |               9 |
| `protein`        |                  45730 |               9 |
| `servo`          |                    167 |               4 |
| `skillcraft`     |                   3338 |              19 |
| `sml`            |                   4137 |              26 |
| `song`           |                 515345 |              90 |
| `tamielectric`   |                  45781 |               3 |
| `wine`           |                   1599 |              11 |

Dataset information can be obtained from the `all_datasets` dictionary.
For example, to obtain a list of all datasets with fewer than 1000 observations, execute the following:

```python
from uci_datasets import all_datasets
[name for name, (n_observations, n_dimensions) in all_datasets.items() if n_observations < 1000]
```

## Papers using these datasets

The following papers use the same datasets and test-train splits present in this repository.

- [Yang, Zichao, et al. "A la carte–learning fast kernels." Artificial Intelligence and Statistics. 2015.](https://proceedings.mlr.press/v38/yang15b.html)
- [Wilson, Andrew Gordon, et al. "Deep kernel learning." Artificial Intelligence and Statistics. 2016.](https://proceedings.mlr.press/v51/wilson16.html)
- [Evans, Trefor W., and Prasanth B. Nair. "Scalable Gaussian processes with grid-structured eigenfunctions (GP-GRIEF)." International Conference on Machine Learning. 2018.](https://arxiv.org/abs/1807.02125)
- [Evans, Trefor W., and Prasanth B. Nair. "Discretely relaxing continuous variables for tractable variational inference." Neural Information Processing Systems. 2018.](https://arxiv.org/abs/1809.04279)
- [Audouze, Christophe, and Prasanth B. Nair. "Sparse low-rank separated representation models for learning from data." Proceedings of the Royal Society A 475.2221 (2019).](https://royalsocietypublishing.org/doi/full/10.1098/rspa.2018.0490)
- [Evans, Trefor W., and Prasanth B. Nair. "Quadruply stochastic gaussian processes." arXiv preprint arXiv:2006.03015 (2020).](https://arxiv.org/abs/2006.03015)
