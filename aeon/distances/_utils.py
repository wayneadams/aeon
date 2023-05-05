from numba import njit
from numba.typed import List
import numpy as np


def reshape_pairwise_ts(arr: np.ndarray):
    if arr.ndim == 3:
        return arr
    elif arr.ndim == 2:
        return arr.reshape((1, arr.shape[0], arr.shape[1]))
    elif arr.ndim == 1:
        return arr.reshape((1, 1, arr.shape[0]))
    raise ValueError("Time series must be 1D, 2D or 3D")


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
