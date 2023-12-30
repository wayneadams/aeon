"""Metrics classes to assess performance on forecasting task.

Classes named as ``*Score`` return a value to maximize: the higher the better.
Classes named as ``*Error`` or ``*Loss`` return a value to minimize:
the lower the better.
"""
from copy import deepcopy
from warnings import warn

import numpy as np
import pandas as pd
from sklearn.utils import check_array

from aeon.datatypes import VectorizedDF, check_is_scitype, convert_to
from aeon.performance_metrics.base import BaseMetric

__author__ = ["mloning", "Tomasz Chodakowski", "RNKuhns", "fkiraly"]
__all__ = [
    "BaseForecastingErrorMetric",
]


def _coerce_to_scalar(obj):
    """Coerce obj to scalar, from polymorphic input scalar or pandas."""
    if isinstance(obj, pd.DataFrame):
        assert len(obj) == 1
        assert len(obj.columns) == 1
        return obj.iloc[0, 0]
    if isinstance(obj, pd.Series):
        assert len(obj) == 1
        return obj.iloc[0]
    return obj


def _coerce_to_df(obj):
    """Coerce to pd.DataFrame, from polymorphic input scalar or pandas."""
    return pd.DataFrame(obj)


class BaseForecastingErrorMetric(BaseMetric):
    """Base class for defining forecasting error metrics in aeon.

    Extends aeon's BaseMetric to the forecasting interface. Forecasting error
    metrics measure the error (loss) between forecasts and true values.

    `multioutput` and `multilevel` parameters can be used to control averaging
    across variables (`multioutput`) and (non-temporal) hierarchy levels (`multilevel`).

    Parameters
    ----------
    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines whether and how to aggregate metric for across variables.
        If 'uniform_average' (default), errors are mean-averaged across variables.
        If array-like, errors are weighted averaged across variables, values as weights.
        If 'raw_values', does not average errors across variables, columns are retained.

    multilevel : {'raw_values', 'uniform_average', 'uniform_average_time'}
        Defines how to aggregate metric for hierarchical data (with levels).
        If 'uniform_average' (default), errors are mean-averaged across levels.
        If 'uniform_average_time', errors are mean-averaged across rows.
        If 'raw_values', does not average errors across levels, hierarchy is retained.
    """

    _tags = {
        "requires-y-train": False,
        "requires-y-pred-benchmark": False,
        "univariate-only": False,
        "lower_is_better": True,
        # "y_inner_type": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"]
        "inner_implements_multilevel": False,
    }

    def __init__(self, multioutput="uniform_average", multilevel="uniform_average"):
        self.multioutput = multioutput
        self.multilevel = multilevel

        if not hasattr(self, "name"):
            self.name = type(self).__name__

        super(BaseForecastingErrorMetric, self).__init__()

    def __call__(self, y_true, y_pred, **kwargs):
        """Calculate metric value using underlying metric function.

        Parameters
        ----------
        y_true : time series in aeon compatible data container format
            Ground truth (correct) target values
            y can be in one of the following formats:
            Series type: pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            Panel type: pd.DataFrame with 2-level row MultiIndex,
                3D np.ndarray, list of Series pd.DataFrame, or nested pd.DataFrame
            Hierarchical type: pd.DataFrame with 3 or more level row MultiIndex
        y_pred :time series in aeon compatible data container format
            Forecasted values to evaluate
            must be of same format as y_true, same indices and columns if indexed

        Returns
        -------
        loss : float, np.ndarray, or pd.DataFrame
            Calculated metric, averaged or by variable.
            float if self.multioutput="uniform_average" or array-like
                and self.multilevel="uniform_average" or "uniform_average_time"
                value is metric averaged over variables and levels (see class docstring)
            np.ndarray of shape (y_true.columns,) if self.multioutput="raw_values"
                and self.multilevel="uniform_average" or "uniform_average_time"
                i-th entry is metric calculated for i-th variable
            pd.DataFrame if self.multilevel=raw.values
                of shape (n_levels, ) if self.multioutput = "uniform_average" or array
                of shape (n_levels, y_true.columns) if self.multioutput="raw_values"
                metric is applied per level, row averaging (yes/no) as in multioutput
        """
        return self.evaluate(y_true, y_pred, **kwargs)

    def evaluate(self, y_true, y_pred, **kwargs):
        """Evaluate the desired metric on given inputs.

        Parameters
        ----------
        y_true : time series in aeon compatible data container format
            Ground truth (correct) target values
            y can be in one of the following formats:
            Series type: pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            Panel type: pd.DataFrame with 2-level row MultiIndex,
                3D np.ndarray, list of Series pd.DataFrame, or nested pd.DataFrame
            Hierarchical type: pd.DataFrame with 3 or more level row MultiIndex
        y_pred :time series in aeon compatible data container format
            Forecasted values to evaluate
            must be of same format as y_true, same indices and columns if indexed

        Returns
        -------
        loss : float, np.ndarray, or pd.DataFrame
            Calculated metric, averaged or by variable.
            float if self.multioutput="uniform_average" or array-like
                and self.multilevel="uniform_average" or "uniform_average_time"
                value is metric averaged over variables and levels (see class docstring)
            np.ndarray of shape (y_true.columns,) if self.multioutput="raw_values"
                and self.multilevel="uniform_average" or "uniform_average_time"
                i-th entry is metric calculated for i-th variable
            pd.DataFrame if self.multilevel=raw.values
                of shape (n_levels, ) if self.multioutput = "uniform_average" or array
                of shape (n_levels, y_true.columns) if self.multioutput="raw_values"
                metric is applied per level, row averaging (yes/no) as in multioutput
        """
        multioutput = self.multioutput
        multilevel = self.multilevel
        # Input checks and conversions
        y_true_inner, y_pred_inner, multioutput, multilevel, kwargs = self._check_ys(
            y_true, y_pred, multioutput, multilevel, **kwargs
        )

        requires_vectorization = isinstance(y_true_inner, VectorizedDF)
        if not requires_vectorization:
            # pass to inner function
            out_df = self._evaluate(y_true=y_true_inner, y_pred=y_pred_inner, **kwargs)
        else:
            out_df = self._evaluate_vectorized(
                y_true=y_true_inner, y_pred=y_pred_inner, **kwargs
            )
            if multilevel == "uniform_average":
                out_df = out_df.mean(axis=0)
                # if level is averaged, but not variables, return numpy
                if isinstance(multioutput, str) and multioutput == "raw_values":
                    out_df = out_df.values

        if (
            multilevel == "uniform_average"
            and isinstance(multioutput, str)
            and multioutput == "uniform_average"
        ):
            out_df = _coerce_to_scalar(out_df)
        if multilevel == "raw_values":
            out_df = _coerce_to_df(out_df)

        return out_df

    def _evaluate(self, y_true, y_pred, **kwargs):
        """Evaluate the desired metric on given inputs.

        private _evaluate containing core logic, called from evaluate

        By default this uses evaluate_by_index, taking arithmetic mean over time points.

        Parameters
        ----------
        y_true : time series in aeon compatible data container format
            Ground truth (correct) target values
            y can be in one of the following formats:
            Series type: pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            Panel type: pd.DataFrame with 2-level row MultiIndex,
                3D np.ndarray, list of Series pd.DataFrame, or nested pd.DataFrame
            Hierarchical type: pd.DataFrame with 3 or more level row MultiIndex
        y_pred :time series in aeon compatible data container format
            Forecasted values to evaluate
            must be of same format as y_true, same indices and columns if indexed

        Returns
        -------
        loss : float or np.ndarray
            Calculated metric, averaged or by variable.
            float if self.multioutput="uniform_average" or array-like
                value is metric averaged over variables (see class docstring)
            np.ndarray of shape (y_true.columns,) if self.multioutput="raw_values"
                i-th entry is metric calculated for i-th variable
        """
        # multioutput = self.multioutput
        # multilevel = self.multilevel
        try:
            index_df = self._evaluate_by_index(y_true, y_pred, **kwargs)
            return index_df.mean(axis=0)
        except RecursionError:
            RecursionError("Must implement one of _evaluate or _evaluate_by_index")

    def _evaluate_vectorized(self, y_true, y_pred, **kwargs):
        """Vectorized version of _evaluate.

        Runs _evaluate for all instances in y_true, y_pred,
        and returns results in a hierarchical pandas.DataFrame.

        Parameters
        ----------
        y_true : pandas.DataFrame with MultiIndex, last level time-like
        y_pred : pandas.DataFrame with MultiIndex, last level time-like
        non-time-like instanceso of y_true, y_pred must be identical
        """
        kwargsi = deepcopy(kwargs)
        n_batches = len(y_true)
        res = []
        for i in range(n_batches):
            if "y_train" in kwargs:
                kwargsi["y_train"] = kwargs["y_train"][i]
            if "y_pred_benchmark" in kwargs:
                kwargsi["y_pred_benchmark"] = kwargs["y_pred_benchmark"][i]
            resi = self._evaluate(y_true=y_true[i], y_pred=y_pred[i], **kwargsi)
            if isinstance(resi, float):
                resi = pd.Series(resi)
            if self.multioutput == "raw_values":
                assert isinstance(resi, np.ndarray)
                df = pd.DataFrame(columns=y_true.X.columns)
                df.loc[0] = resi
                resi = df
            res += [resi]
        out_df = y_true.reconstruct(res)
        if out_df.index.nlevels == y_true.X.index.nlevels:
            out_df.index = out_df.index.droplevel(-1)

        return out_df

    def evaluate_by_index(self, y_true, y_pred, **kwargs):
        """Return the metric evaluated at each time point.

        Parameters
        ----------
        y_true : time series in aeon compatible data container format
            Ground truth (correct) target values
            y can be in one of the following formats:
            Series type: pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            Panel type: pd.DataFrame with 2-level row MultiIndex,
                3D np.ndarray, list of Series pd.DataFrame, or nested pd.DataFrame
            Hierarchical type: pd.DataFrame with 3 or more level row MultiIndex
        y_pred :time series in aeon compatible data container format
            Forecasted values to evaluate
            must be of same format as y_true, same indices and columns if indexed

        Returns
        -------
        loss : pd.Series or pd.DataFrame
            Calculated metric, by time point (default=jackknife pseudo-values).
            pd.Series if self.multioutput="uniform_average" or array-like
                index is equal to index of y_true
                entry at index i is metric at time i, averaged over variables
            pd.DataFrame if self.multioutput="raw_values"
                index and columns equal to those of y_true
                i,j-th entry is metric at time i, at variable j
        """
        multioutput = self.multioutput
        multilevel = self.multilevel
        # Input checks and conversions
        y_true_inner, y_pred_inner, multioutput, multilevel, kwargs = self._check_ys(
            y_true, y_pred, multioutput, multilevel, **kwargs
        )
        # pass to inner function
        out_df = self._evaluate_by_index(y_true_inner, y_pred_inner, **kwargs)

        return out_df

    def _evaluate_by_index(self, y_true, y_pred, **kwargs):
        """Return the metric evaluated at each time point.

        private _evaluate_by_index containing core logic, called from evaluate_by_index

        By default this uses _evaluate to find jackknifed pseudosamples.
        This yields estimates for the metric at each of the time points.
        Caution: this is only sensible for differentiable statistics,
        i.e., not for medians, quantiles or median/quantile based statistics.

        Parameters
        ----------
        y_true : time series in aeon compatible data container format
            Ground truth (correct) target values
            y can be in one of the following formats:
            Series type: pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            Panel type: pd.DataFrame with 2-level row MultiIndex,
                3D np.ndarray, list of Series pd.DataFrame, or nested pd.DataFrame
            Hierarchical type: pd.DataFrame with 3 or more level row MultiIndex
        y_pred :time series in aeon compatible data container format
            Forecasted values to evaluate
            must be of same format as y_true, same indices and columns if indexed

        Returns
        -------
        loss : pd.Series or pd.DataFrame
            Calculated metric, by time point (default=jackknife pseudo-values).
            pd.Series if self.multioutput="uniform_average" or array-like
                index is equal to index of y_true
                entry at index i is metric at time i, averaged over variables
            pd.DataFrame if self.multioutput="raw_values"
                index and columns equal to those of y_true
                i,j-th entry is metric at time i, at variable j
        """
        multioutput = self.multioutput
        n = y_true.shape[0]
        if multioutput == "raw_values":
            out_series = pd.DataFrame(index=y_true.index, columns=y_true.columns)
        else:
            out_series = pd.Series(index=y_true.index)
        try:
            x_bar = self.evaluate(y_true, y_pred, **kwargs)
            for i in range(n):
                idx = y_true.index[i]
                pseudovalue = n * x_bar - (n - 1) * self.evaluate(
                    y_true.drop(idx),
                    y_pred.drop(idx),
                )
                out_series.loc[idx] = pseudovalue
            return out_series
        except RecursionError:
            RecursionError("Must implement one of _evaluate or _evaluate_by_index")

    def _check_consistent_input(self, y_true, y_pred, multioutput, multilevel):
        y_true_orig = y_true
        y_pred_orig = y_pred

        # unwrap y_true, y_pred, if wrapped in VectorizedDF
        if isinstance(y_true, VectorizedDF):
            y_true = y_true.X
        if isinstance(y_pred, VectorizedDF):
            y_pred = y_pred.X

        # check row and column indices if y_true vs y_pred
        same_rows = y_true.index.equals(y_pred.index)
        same_row_num = len(y_true.index) == len(y_pred.index)
        same_cols = y_true.columns.equals(y_pred.columns)
        same_col_num = len(y_true.columns) == len(y_pred.columns)

        if not same_row_num:
            raise ValueError("y_pred and y_true do not have the same number of rows.")
        if not same_col_num:
            raise ValueError(
                "y_pred and y_true do not have the same number of columns."
            )

        if not same_rows:
            warn(
                "y_pred and y_true do not have the same row index. "
                "This may indicate incorrect objects passed to the metric. "
                "Indices of y_true will be used for y_pred."
            )
            y_pred_orig = y_pred_orig.copy()
            if isinstance(y_pred_orig, VectorizedDF):
                y_pred_orig.X.index = y_true.index
            else:
                y_pred_orig.index = y_true.index
        if not same_cols:
            warn(
                "y_pred and y_true do not have the same column index. "
                "This may indicate incorrect objects passed to the metric. "
                "Indices of y_true will be used for y_pred."
            )
            y_pred_orig = y_pred_orig.copy()
            if isinstance(y_pred_orig, VectorizedDF):
                y_pred_orig.X.columns = y_true.columns
            else:
                y_pred_orig.columns = y_true.columns
        # check multioutput arg
        # add this back when variance_weighted is supported
        # ("raw_values", "uniform_average", "variance_weighted")
        allowed_multioutput_str = ("raw_values", "uniform_average")

        if isinstance(multioutput, str):
            if multioutput not in allowed_multioutput_str:
                raise ValueError(
                    f"Allowed 'multioutput' values are {allowed_multioutput_str}, "
                    f"but found multioutput={multioutput}"
                )
        else:
            multioutput = check_array(multioutput, ensure_2d=False)
            if len(y_pred.columns) != len(multioutput):
                raise ValueError(
                    "There must be equally many custom weights (%d) as outputs (%d)."
                    % (len(multioutput), len(y_pred.columns))
                )

        # check multilevel arg
        allowed_multilevel_str = (
            "raw_values",
            "uniform_average",
            "uniform_average_time",
        )

        if not isinstance(multilevel, str):
            raise ValueError(f"multilevel must be a str, but found {type(multilevel)}")
        if multilevel not in allowed_multilevel_str:
            raise ValueError(
                f"Allowed 'multilevel' values are {allowed_multilevel_str}, "
                f"but found multilevel={multilevel}"
            )

        return y_true_orig, y_pred_orig, multioutput, multilevel

    def _check_ys(self, y_true, y_pred, multioutput, multilevel, **kwargs):
        types = ["Series", "Panel", "Hierarchical"]
        inner_types = ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"]

        def _coerce_to_df(y, var_name="y"):
            valid, msg, metadata = check_is_scitype(
                y, scitype=types, return_metadata=True, var_name=var_name
            )
            if not valid:
                raise TypeError(msg)
            y_inner = convert_to(y, to_type=inner_types)

            type = metadata["scitype"]
            ignore_index = multilevel == "uniform_average_time"
            if type in ["Panel", "Hierarchical"] and not ignore_index:
                y_inner = VectorizedDF(y_inner, is_scitype=type)
            return y_inner

        y_true = _coerce_to_df(y_true, var_name="y_true")
        y_pred = _coerce_to_df(y_pred, var_name="y_pred")
        if "y_train" in kwargs.keys():
            kwargs["y_train"] = _coerce_to_df(kwargs["y_train"], var_name="y_train")
        if "y_pred_benchmark" in kwargs.keys():
            kwargs["y_pred_benchmark"] = _coerce_to_df(
                kwargs["y_pred_benchmark"], var_name="y_pred_benchmark"
            )

        y_true, y_pred, multioutput, multilevel = self._check_consistent_input(
            y_true, y_pred, multioutput, multilevel
        )

        return y_true, y_pred, multioutput, multilevel, kwargs
