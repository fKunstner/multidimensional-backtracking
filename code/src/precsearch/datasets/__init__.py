import os
from dataclasses import dataclass, field
from typing import Tuple

import dsdl
import numpy as np
from dsdl import config as dsdlconfig
from numpy.typing import NDArray

import precsearch.config

DSDL_DATASETS = [
    "a1a",
    "cpusmall",
    "eunite2001",
    "california-housing",
    "pyrim",
    "mpg",
    "mg",
    "cadata",
    "concrete",
    "power-plant",
    "naval-propulsion",
    "bodyfat",
    "E2006-E2006-tfidf",
    "E2006-log1p",
    "colon-cancer",
    "SVHN",
    "cifar10",
    "real-sim",
    "criteo",
    "kdd2010raw",
    "url",
    "webspam",
    "abalone",
    "yacht",
    "breast-cancer",
    "ionosphere",
    "australian",
    "diabetes",
    "heart",
    "duke-breast-cancer",
    "rcv1.binary",
    "news20.binary",
]
ALL_DATASETS = {"dsdl": DSDL_DATASETS}


@dataclass
class Dataset:
    dataset_name: str
    _X: NDArray = field(init=False, repr=False)
    _y: NDArray = field(init=False, repr=False)
    _n: NDArray = field(init=False, repr=False)
    _d: NDArray = field(init=False, repr=False)

    def uname(self):
        return self.dataset_name

    def _load(self):
        dname = self.dataset_name

        dsdlconfig.Config.set(
            "DATA_ROOT", os.path.join(precsearch.config.base_workspace(), "datasets")
        )

        if dname in DSDL_DATASETS:
            return dsdl.load(dname).get_train()
        else:
            raise ValueError(f"Unknown dataset {dname}. Expected one of {ALL_DATASETS}")

    def load(self) -> Tuple[NDArray, NDArray]:
        """
        Loads the data into memory.

        Caches the results.

        Returns:
            A tuple containing `X` and `y`, the feature matrix and target vector.
        """
        if not hasattr(self, "_X") or not hasattr(self, "_y"):
            self._X, self._y = self._load()
            self._n, self._d = self._X.shape

        K = self._X

        labels = np.unique(self._y)
        y = self._y
        if len(labels) in [2]:
            y[y == labels[0]] = 0.0
            y[y == labels[1]] = 1.0

        return K, y
