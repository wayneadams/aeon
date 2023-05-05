from typing import Tuple, Union

from numba.typed import List
import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def reshape_pairwise_to_multiple(
        x: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Reshape two collection of time series for pairwise distance computation.

    Parameters
    ----------
    x: np.ndarray, of shape (n_instances, n_channels, n_timepoints) or
            (n_instances, n_timepoints) or (n_timepoints,) or List[np.ndarray]
        A collection of time series instances.
    y: np.ndarray, of shape (m_instances, m_channels, m_timepoints) or
            (m_instances, m_timepoints) or (m_timepoints,) or List[np.ndarray]
        A collection of time series instances.

    Returns
    -------
    np.ndarray
        Reshaped x.
    np.ndarray
        Reshaped y.

    Raises
    ------
    ValueError
        If x and y are not 1D, 2D or 3D arrays.
    """
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
def reshape_pairwise_1d_np_list(np_list: List[np.ndarray]):
    new_arr = List()
    for item in np_list:
        new_arr.append(item.reshape((1, item.shape[0])))
    return new_arr


@njit(cache=True, fastmath=True)
def reshape_pairwise_to_multiple_np_list(
        x: List[np.ndarray], y: List[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    x_dims = x[0].ndim
    y_dims = y[0].ndim
    if x_dims == y_dims:
        if y_dims == 2 and x_dims == 2:
            return x, y
        if y_dims == 1 and x_dims == 1:
            _x = reshape_pairwise_1d_np_list(x)
            _y = reshape_pairwise_1d_np_list(y)
            return _x, _y
        if y_dims == 0 and x_dims == 0:
            _x = List(np.array([[x]]))
            _y = List(np.array([[y]]))
            return _x, _y
        raise ValueError("x and y must be 1D, 2D, or 3D arrays")
    if x_dims == 2 and y_dims == 1:
        _y = reshape_pairwise_1d_np_list(y)
        return x, _y
    if y_dims == 2 and x_dims == 1:
        _x = reshape_pairwise_1d_np_list(x)
        return _x, y
    if x_dims == 1 and y_dims == 0:
        _x = reshape_pairwise_1d_np_list(x)
        _y = List(np.array([[y]]))
        return _x, _y
    if y_dims == 1 and x_dims == 0:
        _y = reshape_pairwise_1d_np_list(y)
        _x = List(np.array([[x]]))
        return _x, _y
    raise ValueError("x and y must be 2D or 3D arrays")
