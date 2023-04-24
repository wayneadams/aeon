# -*- coding: utf-8 -*-
r"""Longest common subsequence (LCSS) between two time series.

The LCSS distance for time series is based on the solution to the
longest common subsequence problem in pattern matching [1]. The typical problem
is to find the longest subsequence that is common to two discrete series based on
the edit distance. This approach can be extended to consider real-valued time series
by using a distance threshold epsilon, which defines the maximum difference
between a pair of values that is allowed for them to be considered a match.
LCSS finds the optimal alignment between two series by find the greatest number
of matching pairs. The LCSS distance uses a matrix :math:'L' that records the
sequence of matches over valid warpings. For two series :math:'a = a_1,... a_m
and b = b_1,... b_m, L' is found by iterating over all valid windows (i.e.
where bounding_matrix is not infinity, which by default is the constant band
:math:'|i-j|<w*m', where :math:'w' is the window parameter value and m is series
length), then calculating

::math
if(|a_i - b_j| < espilon) \\
        &L_{i,j} \leftarrow L_{i-1,j-1}+1 \\
else\\
        &L_{i,j} \leftarrow \max(L_{i,j-1}, L_{i-1,j})\\

The distance is an inverse function of the final LCSS.
::math
d_{LCSS}({\bf a,b}) = 1- \frac{LCSS({\bf a,b})}{m}.\]

Note that series a and b need not be equal length.

References
----------
.. [1] D. Hirschberg, Algorithms for the longest common subsequence problem, Journal
of the ACM 24(4), 664--675, 1977
"""
__author__ = ["chrisholder", "TonyBagnall"]

from typing import List, Tuple
import numpy as np
from numba import njit

from aeon.distances._alignment_paths import compute_lcss_return_path
from aeon.distances._bounding_matrix import create_bounding_matrix
from aeon.distances._squared import _univariate_squared_distance


@njit(cache=True)
def lcss_distance(
        x: np.ndarray, y: np.ndarray, window: float = None, epsilon: float = 1.0
) -> float:
    r"""Return the lcss distance between x and y.

    LCSS attempts to find the longest common sequence between two time series and
    returns a value that is the percentage that longest common sequence assumes.
    Originally present in [1]_, LCSS is computed by matching indexes that are
    similar up until a defined threshold (epsilon).

    The value returned will be between 0.0 and 1.0, where 0.0 means the two time series
    are exactly the same and 1.0 means they are complete opposites.

    Parameters
    ----------
    x: np.ndarray, of shape (n_channels, n_timepoints) or (n_timepoints,) or
            (n_instances, n_channels, n_timepoints)
        First time series.
    y: np.ndarray, of shape (m_channels, m_timepoints) or (m_timepoints,) or
            (m_instances, m_channels, m_timepoints)
        Second time series.
    window : float, defaults=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    epsilon: float, defaults=1.
        Matching threshold to determine if two subsequences are considered close
        enough to be considered 'common'. The default is 1.

    Returns
    -------
    float
        The lcss distance between x and y.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import lcss_distance
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> lcss_distance(x, y)
    0.0

    Raises
    ------
    ValueError
        If x and y are not 1D, 2D, or 3D arrays.

    References
    ----------
    .. [1] M. Vlachos, D. Gunopoulos, and G. Kollios. 2002. "Discovering
        Similar Multidimensional Trajectories", In Proceedings of the
        18th International Conference on Data Engineering (ICDE '02).
        IEEE Computer Society, USA, 673.
    """
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(_x.shape[1], _y.shape[1], window)
        return _lcss_distance(_x, _y, bounding_matrix, epsilon)
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(x.shape[1], y.shape[1], window)
        return _lcss_distance(x, y, bounding_matrix, epsilon)
    if x.ndim == 3 and y.ndim == 3:
        distance = 0
        bounding_matrix = create_bounding_matrix(x.shape[2], y.shape[2], window)
        for curr_x, curr_y in zip(x, y):
            distance += _lcss_distance(curr_x, curr_y, bounding_matrix, epsilon)
        return distance
    raise ValueError("x and y must be 1D, 2D, or 3D arrays")


@njit(cache=True)
def lcss_cost_matrix(
        x: np.ndarray, y: np.ndarray, window: float = None, epsilon: float = 1.0
) -> np.ndarray:
    r"""Return the lcss cost matrix between x and y.

    Parameters
    ----------
    x: np.ndarray, of shape (n_channels, n_timepoints) or (n_timepoints,) or
            (n_instances, n_channels, n_timepoints)
        First time series.
    y: np.ndarray, of shape (m_channels, m_timepoints) or (m_timepoints,) or
            (m_instances, m_channels, m_timepoints)
        Second time series.
    window : float, defaults=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    epsilon: float, defaults=1.
        Matching threshold to determine if two subsequences are considered close
        enough to be considered 'common'. The default is 1.

    Returns
    -------
    np.ndarray
        The lcss cost matrix between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D, 2D, or 3D arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import lcss_cost_matrix
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> lcss_cost_matrix(x, y)
    array([[ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
           [ 1.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.],
           [ 1.,  2.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.],
           [ 1.,  2.,  3.,  4.,  4.,  4.,  4.,  4.,  4.,  4.],
           [ 1.,  2.,  3.,  4.,  5.,  5.,  5.,  5.,  5.,  5.],
           [ 1.,  2.,  3.,  4.,  5.,  6.,  6.,  6.,  6.,  6.],
           [ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  7.,  7.,  7.],
           [ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  8.,  8.],
           [ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,  9.],
           [ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.]])
    """
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(_x.shape[1], _y.shape[1], window)
        return _lcss_cost_matrix(_x, _y, bounding_matrix, epsilon)
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(x.shape[1], y.shape[1], window)
        return _lcss_cost_matrix(x, y, bounding_matrix, epsilon)
    if x.ndim == 3 and y.ndim == 3:
        bounding_matrix = create_bounding_matrix(x.shape[2], y.shape[2], window)
        cost_matrix = np.zeros((x.shape[2], y.shape[2]))
        for curr_x, curr_y in zip(x, y):
            cost_matrix = np.add(
                cost_matrix, _lcss_cost_matrix(curr_x, curr_y, bounding_matrix, epsilon)
            )
        return cost_matrix
    raise ValueError("x and y must be 1D, 2D, or 3D arrays")


@njit(cache=True)
def _lcss_distance(
        x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, epsilon: float
) -> float:
    distance = _lcss_cost_matrix(x, y, bounding_matrix, epsilon)[
        x.shape[1] - 1, y.shape[1] - 1
    ]
    return 1 - float(distance / min(x.shape[1], y.shape[1]))


@njit(cache=True)
def _lcss_cost_matrix(
        x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, epsilon
) -> np.ndarray:
    x_size = x.shape[1]
    y_size = y.shape[1]

    cost_matrix = np.zeros((x_size + 1, y_size + 1))

    for i in range(1, x_size + 1):
        for j in range(1, y_size + 1):
            if bounding_matrix[i - 1, j - 1]:
                squared_distance = _univariate_squared_distance(x[:, i - 1],
                                                                y[:, j - 1])
                if squared_distance <= epsilon:
                    cost_matrix[i, j] = 1 + cost_matrix[i - 1, j - 1]
                else:
                    cost_matrix[i, j] = max(
                        cost_matrix[i, j - 1], cost_matrix[i - 1, j]
                    )

    return cost_matrix[1:, 1:]


@njit(cache=True)
def lcss_pairwise_distance(
        X: np.ndarray, window: float = None, epsilon: float = 1.0
) -> np.ndarray:
    """Compute the lcss pairwise distance between a set of time series.

    Parameters
    ----------
    X: np.ndarray, of shape (n_instances, n_channels, n_timepoints) or
            (n_instances, n_timepoints)
        A collection of time series instances.
    window: float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    epsilon: float, defaults=1.
        Matching threshold to determine if two subsequences are considered close
        enough to be considered 'common'. The default is 1.

    Returns
    -------
    np.ndarray (n_instances, n_instances)
        lcss pairwise matrix between the instances of X.

    Raises
    ------
    ValueError
        If x and y are not 2D or 3D arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import lcss_pairwise_distance
    >>> X = np.array([[[1, 2, 3, 4]],[[4, 5, 6, 3]], [[7, 8, 9, 3]]])
    >>> lcss_pairwise_distance(X)
    array([[0.  , 0.5 , 0.75],
           [0.5 , 0.  , 0.5 ],
           [0.75, 0.5 , 0.  ]])
    """
    if X.ndim == 3:
        return _lcss_pairwise_distance(X, window, epsilon)
    if X.ndim == 2:
        _X = X.reshape((X.shape[0], 1, X.shape[1]))
        return _lcss_pairwise_distance(_X, window, epsilon)

    raise ValueError("x and y must be 2D or 3D arrays")


@njit(cache=True)
def _lcss_pairwise_distance(
        X: np.ndarray, window: float, epsilon: float
) -> np.ndarray:
    n_instances = X.shape[0]
    distances = np.zeros((n_instances, n_instances))
    bounding_matrix = create_bounding_matrix(X.shape[2], X.shape[2], window)

    for i in range(n_instances):
        for j in range(i + 1, n_instances):
            distances[i, j] = _lcss_distance(X[i], X[j], bounding_matrix, epsilon)
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True)
def lcss_from_single_to_multiple_distance(
        x: np.ndarray, y: np.ndarray, window: float = None, epsilon: float = 1.0
) -> np.ndarray:
    """Compute the lcss distance between a single time series and multiple.

    Parameters
    ----------
    x: np.ndarray, (n_channels, n_timepoints) or (n_timepoints,)
        Single time series.
    y: np.ndarray, of shape (m_instances, m_channels, m_timepoints) or
            (m_instances, m_timepoints)
        A collection of time series instances.
    window: float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    epsilon: float, defaults=1.
        Matching threshold to determine if two subsequences are considered close
        enough to be considered 'common'. The default is 1.

    Returns
    -------
    np.ndarray (n_instances)
        lcss distance between the collection of instances in y and the time series x.

    Raises
    ------
    ValueError
        If x and y are not 2D or 3D arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import lcss_from_single_to_multiple_distance
    >>> x = np.array([[1, 2, 3, 6]])
    >>> y = np.array([[[1, 2, 3, 4]],[[4, 5, 6, 3]], [[7, 8, 9, 3]]])
    >>> lcss_from_single_to_multiple_distance(x, y)
    array([0.25, 0.5 , 0.75])
    """
    if y.ndim == 3 and x.ndim == 2:
        return _lcss_from_single_to_multiple_distance(x, y, window, epsilon)
    if y.ndim == 2 and x.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((y.shape[0], 1, y.shape[1]))
        return _lcss_from_single_to_multiple_distance(_x, _y, window, epsilon)
    else:
        raise ValueError("x and y must be 2D or 3D arrays")


@njit(cache=True)
def _lcss_from_single_to_multiple_distance(
        x: np.ndarray, y: np.ndarray, window: float, epsilon: float
) -> np.ndarray:
    n_instances = y.shape[0]
    distances = np.zeros(n_instances)
    bounding_matrix = create_bounding_matrix(x.shape[1], y.shape[2], window)

    for i in range(n_instances):
        distances[i] = _lcss_distance(x, y[i], bounding_matrix, epsilon)

    return distances


@njit(cache=True)
def lcss_from_multiple_to_multiple_distance(
        x: np.ndarray, y: np.ndarray, window: float = None, epsilon: float = 1.0
) -> np.ndarray:
    """Compute the lcss distance between two sets of time series.

    If x and y are the same then you should use lcss_pairwise_distance.

    Parameters
    ----------
    x: np.ndarray, of shape (n_instances, n_channels, n_timepoints) or
            (n_instances, n_timepoints) or (n_timepoints,)
        A collection of time series instances.
    y: np.ndarray, of shape (m_instances, m_channels, m_timepoints) or
            (m_instances, m_timepoints) or (m_timepoints,)
        A collection of time series instances.
    window: float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    epsilon: float, defaults=1.
        Matching threshold to determine if two subsequences are considered close
        enough to be considered 'common'. The default is 1.

    Returns
    -------
    np.ndarray (n_instances, m_instances)
        lcss distance between two collections of time series, x and y.

    Raises
    ------
    ValueError
        If x and y are not 2D or 3D arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import lcss_from_multiple_to_multiple_distance
    >>> x = np.array([[[1, 2, 3, 3]],[[4, 5, 6, 9]], [[7, 8, 9, 22]]])
    >>> y = np.array([[[11, 12, 13, 2]],[[14, 15, 16, 1]], [[17, 18, 19, 10]]])
    >>> lcss_from_multiple_to_multiple_distance(x, y)
    array([[0.75, 0.75, 1.  ],
           [1.  , 1.  , 0.75],
           [1.  , 1.  , 0.75]])
    """
    if y.ndim == 3 and x.ndim == 3:
        return _lcss_from_multiple_to_multiple_distance(x, y, window, epsilon)
    if y.ndim == 2 and x.ndim == 2:
        _x = x.reshape((x.shape[0], 1, x.shape[1]))
        _y = y.reshape((y.shape[0], 1, y.shape[1]))
        return _lcss_from_multiple_to_multiple_distance(_x, _y, window, epsilon)
    if y.ndim == 1 and x.ndim == 1:
        _x = x.reshape((1, 1, x.shape[0]))
        _y = y.reshape((1, 1, y.shape[0]))
        return _lcss_from_multiple_to_multiple_distance(_x, _y, window, epsilon)
    raise ValueError("x and y must be 1D, 2D, or 3D arrays")


@njit(cache=True)
def _lcss_from_multiple_to_multiple_distance(
        x: np.ndarray, y: np.ndarray, window: float, epsilon: float
) -> np.ndarray:
    n_instances = x.shape[0]
    m_instances = y.shape[0]
    distances = np.zeros((n_instances, m_instances))
    bounding_matrix = create_bounding_matrix(x.shape[2], y.shape[2], window)

    for i in range(n_instances):
        for j in range(m_instances):
            distances[i, j] = _lcss_distance(x[i], y[j], bounding_matrix, epsilon)
    return distances


@njit(cache=True)
def lcss_alignment_path(
        x: np.ndarray, y: np.ndarray, window: float = None, epsilon: float = 1.0
) -> Tuple[List[Tuple[int, int]], float]:
    """Compute the lcss alignment path between two time series.

    Parameters
    ----------
    x: np.ndarray, of shape (n_channels, n_timepoints) or (n_timepoints,) or
            (n_instances, n_channels, n_timepoints)
        First time series.
    y: np.ndarray, of shape (m_channels, m_timepoints) or (m_timepoints,) or
            (m_instances, m_channels, m_timepoints)
        Second time series.
    window: float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    epsilon: float, defaults=1.
        Matching threshold to determine if two subsequences are considered close
        enough to be considered 'common'. The default is 1.


    Returns
    -------
    List[Tuple[int, int]]
        The alignment path between the two time series where each element is a tuple
        of the index in x and the index in y that have the best alignment according
        to the cost matrix.
    float
        The lcss distance between the two time series.

    Raises
    ------
    ValueError
        If x and y are not 1D, 2D, or 3D arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import lcss_alignment_path
    >>> x = np.array([[1, 2, 3, 6]])
    >>> y = np.array([[1, 2, 3, 4]])
    >>> lcss_alignment_path(x, y)
    ([(0, 0), (1, 1), (2, 2)], 0.25)
    """
    x_size = x.shape[-1]
    y_size = y.shape[-1]
    cost_matrix = lcss_cost_matrix(x, y, window, epsilon)
    distance = 1 - float(cost_matrix[x_size - 1, y_size - 1] / min(x_size, y_size))
    bounding_matrix = create_bounding_matrix(x_size, y_size, window)
    return (
        compute_lcss_return_path(x, y, epsilon, bounding_matrix, cost_matrix),
        distance,
    )
