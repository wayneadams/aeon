from numba.typed import List

from typing import Tuple
import numpy as np
from numba import njit


def reshape_distance_ts(arr: np.ndarray):
    if arr.ndim == 2:
        return arr.reshape((1, arr.shape[0], arr.shape[1]))
    elif arr.ndim == 1:
        return arr.reshape((1, 1, arr.shape[0]))
    raise ValueError("Time series must be 1D or 2D")


@njit(cache=True, fastmath=True)
def reshape_pairwise_to_multiple(
        x: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    if x.ndim == y.ndim:
        if y.ndim == 3 and x.ndim == 3:
            return x, y
        if y.ndim == 2 and x.ndim == 2:
            _x = x.reshape((x.shape[0], 1, x.shape[1]))
            _y = y.reshape((y.shape[0], 1, y.shape[1]))
            return _x, _y
        if y.ndim == 1 and x.ndim == 1:
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


@njit(cache=True, fastmath=True)
def reshape_pairwise_np_list(np_list: List[np.ndarray]):
    """Reshape a np.ndarray np-list to a 2d np.ndarray np-list.

    Parameters
    ----------
    np_list : numba.typed.List[np.ndarray]
        A list of np.ndarrays (which can be different lengths).

    Returns
    -------
    new_arr : numba.typed.List[np.ndarray], where the np.ndarrays are of shape (1, n_i).
        A list of 2d np.ndarrays (which can be different lengths)

    Examples
    --------
    >>> from aeon.distances._utils import reshape_pairwise_np_list
    >>> np_list_1d = List([np.array([1, 2, 3]), np.array([4, 5, 6, 7])])
    >>> reshape_pairwise_np_list(np_list_1d)
    ListType[array(int64, 2d, C)]([[[1 2 3]], [[4 5 6 7]]])
    """
    if np_list[0].ndim == 2:
        return np_list
    elif np_list[0].ndim == 1:
        new_arr = List()
        for item in np_list:
            new_arr.append(item.reshape((1, item.shape[0])))
        return new_arr
    elif np_list[0].ndim == 0:
        new_arr = List()
        for item in np_list:
            new_arr.append(item.reshape((1, item.shape[0])))
    else:
        raise ValueError("x and y must be 1D, 2D, or 3D arrays")
