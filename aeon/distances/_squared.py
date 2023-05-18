# -*- coding: utf-8 -*-
__author__ = ["chrisholder", "tonybagnall"]

import numpy as np
from numba import njit

from aeon.distances._utils import (
    _reshape_ndarray_for_multiple_to_multiple,
    _reshape_np_list_to_2d_np_list,
    python_list_to_numba_list
)


@njit(cache=True, fastmath=True)
def squared_distance(x: np.ndarray, y: np.ndarray) -> float:
    r"""Compute the squared distance between two time series.

    The squared distance between two time series is defined as:
    .. math::
        sd(x, y) = \sum_{i=1}^{n} (x_i - y_i)^2

    Parameters
    ----------
    x: np.ndarray, of shape (n_channels, n_timepoints) or (n_timepoints,)
        First time series.
    y: np.ndarray, of shape (m_channels, m_timepoints) or (m_timepoints,)
        Second time series.

    Returns
    -------
    float
        Squared distance between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import squared_distance
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
    >>> squared_distance(x, y)
    1000.0
    """
    if x.ndim == 1 and y.ndim == 1:
        return _univariate_squared_distance(x, y)
    if x.ndim == 2 and y.ndim == 2:
        return _squared_distance(x, y)
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def _squared_distance(x: np.ndarray, y: np.ndarray) -> float:
    distance = 0.0
    min_val = min(x.shape[0], y.shape[0])
    for i in range(min_val):
        distance += _univariate_squared_distance(x[i], y[i])
    return distance


@njit(cache=True, fastmath=True)
def _univariate_squared_distance(x: np.ndarray, y: np.ndarray) -> float:
    distance = 0.0
    min_length = min(x.shape[0], y.shape[0])
    for i in range(min_length):
        difference = x[i] - y[i]
        distance += difference * difference
    return distance


@njit(cache=True, fastmath=True)
def squared_pairwise_distance(X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
    """Compute the squared pairwise distance between a set of time series.

    Parameters
    ----------
    X: np.ndarray, of shape (n_instances, n_channels, n_timepoints) or
            (n_instances, n_timepoints) or (n_timepoints,)
        A collection of time series instances.
    y: np.ndarray, of shape (m_instances, m_channels, m_timepoints) or
            (m_instances, m_timepoints) or (m_timepoints,), default=None
        A collection of time series instances.


    Returns
    -------
    np.ndarray (n_instances, n_instances)
        squared pairwise matrix between the instances of X.

    Raises
    ------
    ValueError
        If X is not 2D or 3D array when only passing X.
        If X and y are not 1D, 2D or 3D arrays when passing both X and y.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import squared_pairwise_distance
    >>> X = np.array([[[1, 2, 3, 4]],[[4, 5, 6, 3]], [[7, 8, 9, 3]]])
    >>> squared_pairwise_distance(X)
    array([[  0.,  28., 109.],
           [ 28.,   0.,  27.],
           [109.,  27.,   0.]])

    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> y = np.array([[[11, 12, 13]],[[14, 15, 16]], [[17, 18, 19]]])
    >>> squared_pairwise_distance(X, y)
    array([[300., 507., 768.],
           [147., 300., 507.],
           [ 48., 147., 300.]])

    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> y_univariate = np.array([[11, 12, 13],[14, 15, 16], [17, 18, 19]])
    >>> squared_pairwise_distance(X, y_univariate)
    array([[300.],
           [147.],
           [ 48.]])
    """
    if y is None:
        # To self
        if isinstance(X, list):
            x_dims = X[0].ndim
            if x_dims == 2:
                _X = python_list_to_numba_list(X)
                return _squared_pairwise_distance(_X)
            if x_dims == 1:
                _X = _reshape_np_list_to_2d_np_list(X)
                return _squared_pairwise_distance(_X)
            raise ValueError("X must be 2D or 3D array")
        else:
            if X.ndim == 3:
                return _squared_pairwise_distance(X)
            if X.ndim == 2:
                _X = X.reshape((X.shape[0], 1, X.shape[1]))
                return _squared_pairwise_distance(_X)
        raise ValueError("X must be 2D or 3D array")

    if isinstance(X, list):
        x_dims = X[0].ndim
        y_dims = y[0].ndim
        if x_dims == y_dims:
            if x_dims == 2:
                _X = python_list_to_numba_list(X)
                _y = python_list_to_numba_list(y)
                return _squared_from_multiple_to_multiple_distance(_X, _y)
            if x_dims == 1:
                _X = _reshape_np_list_to_2d_np_list(X)
                _y = _reshape_np_list_to_2d_np_list(y)
                return _squared_from_multiple_to_multiple_distance(_X, _y)
            raise ValueError("X must be 2D or 3D array")
        else:
            if x_dims == 2 and y_dims == 1:
                _X = python_list_to_numba_list(X)
                _y = _reshape_np_list_to_2d_np_list(y)
                return _squared_from_multiple_to_multiple_distance(_X, _y)
            if x_dims == 1 and y_dims == 2:
                _X = _reshape_np_list_to_2d_np_list(X)
                _y = python_list_to_numba_list(y)
                return _squared_from_multiple_to_multiple_distance(_X, _y)
            if x_dims == 1 and y_dims == 0:
                pass
            if x_dims == 0 and y_dims == 1:
                pass

            raise ValueError("X and y must have the same number of dimensions")
    else:
        _x, _y = _reshape_ndarray_for_multiple_to_multiple(X, y)
    return _squared_from_multiple_to_multiple_distance(_x, _y)


@njit(cache=True, fastmath=True)
def _squared_pairwise_distance(X: np.ndarray) -> np.ndarray:
    n_instances = len(X)
    distances = np.zeros((n_instances, n_instances))

    for i in range(n_instances):
        for j in range(i + 1, n_instances):
            distances[i, j] = squared_distance(X[i], X[j])
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True, fastmath=True)
def _squared_from_multiple_to_multiple_distance(
        x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    n_instances = len(x)
    m_instances = len(y)
    distances = np.zeros((n_instances, m_instances))

    for i in range(n_instances):
        for j in range(m_instances):
            distances[i, j] = squared_distance(x[i], y[j])
    return distances
