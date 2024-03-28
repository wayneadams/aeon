"""Auto-correlation transformations.

Module :mod:`aeon.transformations` implements auto-correlation
transformers.
"""

__maintainer__ = ["TonyBagnall"]
__all__ = ["AutoCorrelationTransformer", "PartialAutoCorrelationTransformer"]

import numpy as np
import pandas as pd
from numba import njit

from aeon.transformations.series import BaseSeriesTransformer


class AutoCorrelationTransformer(BaseSeriesTransformer):
    """Auto-correlation transformer.

    The autocorrelation function measures how correlated a timeseries is
    with itself at different lags. The AutocorrelationTransformer returns
    these values as a series for each lag up to the `n_lags` specified.

    Parameters
    ----------
    adjusted : bool, default=False
        If True, then denominators for autocovariance are n-k, otherwise n.

    n_lags : int, default=None
        Number of lags to return autocorrelation for. If None,
        statsmodels acf function uses min(10 * np.log10(nobs), nobs - 1).

    See Also
    --------
    PartialAutoCorrelationTransformer

    Notes
    -----
    Provides wrapper around statsmodels
    `acf <https://www.statsmodels.org/devel/generated/
    statsmodels.tsa.stattools.acf.html>`_ function.

    Examples
    --------
    >>> from aeon.transformations.acf import AutoCorrelationTransformer
    >>> from aeon.datasets import load_airline
    >>> y = load_airline()  # doctest: +SKIP
    >>> transformer = AutoCorrelationTransformer(n_lags=12)  # doctest: +SKIP
    >>> y_hat = transformer.fit_transform(y)  # doctest: +SKIP
    """

    _tags = {
        "X_inner_type": "np.ndarray",
        "capability:multivariate": True,
        "fit_is_empty": True,
    }

    def __init__(
        self,
        n_lags=None,
    ):
        self.n_lags = n_lags
        super().__init__(axis=1)

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing the core logic, called from transform

        Parameters
        ----------
        X : np.ndarray
            Data to be transformed, shape (n_channels, n_timepoints)
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        transformed version of X
        """
        return self._acf(X, max_lag=self.n_lags)

    @staticmethod
    @njit(cache=True, fastmath=True)
    def _acf(X, max_lag):
        n_channels, length = X.shape

        X_t = np.zeros((n_channels, max_lag))
        for i, x in enumerate(X):
            for lag in range(1, max_lag + 1):
                lag_length = length - lag
                x1, x2 = x[:-lag], x[lag:]
                s1 = np.sum(x1)
                s2 = np.sum(x2)
                m1 = s1 / lag_length
                m2 = s2 / lag_length
                ss1 = np.sum(x1 * x1)
                ss2 = np.sum(x2 * x2)
                v1 = ss1 - s1 * m1
                v2 = ss2 - s2 * m2
                v1_is_zero, v2_is_zero = v1 <= 1e-9, v2 <= 1e-9
                if v1_is_zero and v2_is_zero:  # Both zero variance,
                    # so must be 100% correlated
                    X_t[i][lag - 1] = 1
                elif v1_is_zero or v2_is_zero:  # One zero variance
                    # the other not
                    X_t[i][lag - 1] = 0
                else:
                    X_t[i][lag - 1] = np.sum((x1 - m1) * (x2 - m2)) / np.sqrt(v1 * v2)

        return X_t

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.


        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        return [{}, {"n_lags": 1}]


class PartialAutoCorrelationTransformer(BaseSeriesTransformer):
    """Partial auto-correlation transformer.

    The partial autocorrelation function measures the conditional correlation
    between a timeseries and its self at different lags. In particular,
    the correlation between a time period and a lag, is calculated conditional
    on all the points between the time period and the lag.

    The PartialAutoCorrelationTransformer returns
    these values as a series for each lag up to the `n_lags` specified.

    Parameters
    ----------
    n_lags : int, default=None
        Number of lags to return partial autocorrelation for. If None,
        statsmodels acf function uses min(10 * np.log10(nobs), nobs // 2 - 1).

    method : str, default="ywadjusted"
        Specifies which method for the calculations to use.

        - "yw" or "ywadjusted" : Yule-Walker with sample-size adjustment in
          denominator for acovf. Default.
        - "ywm" or "ywmle" : Yule-Walker without adjustment.
        - "ols" : regression of time series on lags of it and on constant.
        - "ols-inefficient" : regression of time series on lags using a single
          common sample to estimate all pacf coefficients.
        - "ols-adjusted" : regression of time series on lags with a bias
          adjustment.
        - "ld" or "ldadjusted" : Levinson-Durbin recursion with bias
          correction.
        - "ldb" or "ldbiased" : Levinson-Durbin recursion without bias
          correction.

    See Also
    --------
    AutoCorrelationTransformer

    Notes
    -----
    Provides wrapper around statsmodels
    `pacf <https://www.statsmodels.org/devel/generated/
    statsmodels.tsa.stattools.pacf.html>`_ function.


    Examples
    --------
    >>> from aeon.transformations.acf import PartialAutoCorrelationTransformer
    >>> from aeon.datasets import load_airline
    >>> y = load_airline()  # doctest: +SKIP
    >>> transformer = PartialAutoCorrelationTransformer(n_lags=12)  # doctest: +SKIP
    >>> y_hat = transformer.fit_transform(y)  # doctest: +SKIP
    """

    _tags = {
        "input_data_type": "Series",
        # what is the abstract type of X: Series, or Panel
        "output_data_type": "Series",
        "X_inner_type": "pd.Series",
        "univariate-only": True,
        "fit_is_empty": True,
        "python_dependencies": "statsmodels",
    }

    def __init__(
        self,
        n_lags=None,
        method="ywadjusted",
    ):
        self.n_lags = n_lags
        self.method = method
        super().__init__()

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing the core logic, called from transform

        Parameters
        ----------
        X : pd.Series
            Data to be transformed
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        transformed version of X
        """
        from statsmodels.tsa.stattools import pacf

        # Passing an alpha values other than None would return confidence intervals
        # and break the signature of the series-to-series transformer
        zt = pacf(X, nlags=self.n_lags, method=self.method, alpha=None)
        return pd.Series(zt)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.


        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        return [{}, {"n_lags": 1}]