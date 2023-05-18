# -*- coding: utf-8 -*-
from typing import Tuple, List

import numpy as np
from numba import njit
from numba.typed import List as NumbaList

NpList = List[np.ndarray]
NumbaNpList = NumbaList[np.ndarray]


@njit(cache=True, fastmath=True)
def _reshape_ndarray_pairwise_to_3d_ndarray(X: np.ndarray) -> np.ndarray:
    if X.ndim == 3:
        return X
    if X.ndim == 2:
        return X.reshape((X.shape[0], 1, X.shape[1]))
    raise ValueError("x and y must be 2D or 3D arrays")


@njit(cache=True, fastmath=True)
def _reshape_ndarray_for_multiple_to_multiple(
        x: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    if x.ndim == y.ndim:
        if y.ndim == 3:
            return x, y
        if y.ndim == 2:
            _x = x.reshape((x.shape[0], 1, x.shape[1]))
            _y = y.reshape((y.shape[0], 1, y.shape[1]))
            return _x, _y
        if y.ndim == 1:
            _x = x.reshape((1, 1, x.shape[0]))
            _y = y.reshape((1, 1, y.shape[0]))
            return _x, _y
        raise ValueError("x and y must be 1D, 2D, or 3D arrays")
    else:
        if x.ndim == 3 and y.ndim == 2:
            _y = y.reshape((1, y.shape[0], y.shape[1]))
            return x, _y
        if y.ndim == 3 and x.ndim == 2:
            _x = x.reshape((1, x.shape[0], x.shape[1]))
            return _x, y
        if x.ndim == 2 and y.ndim == 1:
            _x = x.reshape((x.shape[0], 1, x.shape[1]))
            _y = y.reshape((1, 1, y.shape[0]))
            return _x, _y
        if y.ndim == 2 and x.ndim == 1:
            _x = x.reshape((1, 1, x.shape[0]))
            _y = y.reshape((y.shape[0], 1, y.shape[1]))
            return _x, _y
        raise ValueError("x and y must be 2D or 3D arrays")


@njit(cache=True)
def python_list_to_numba_list(x: List) -> NumbaList:
    nb_list = NumbaList()
    for i in range(len(x)):
        nb_list.append(x[i])

    return nb_list


@njit(cache=True)
def _reshape_np_list_to_2d_np_list(x: NpList) -> NumbaNpList:
    new_arr = NumbaList()
    for i in range(len(x)):
        new_arr.append(x[i].reshape((1, x[i].shape[0])))
    return new_arr
