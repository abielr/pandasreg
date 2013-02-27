import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from pandas.tseries import frequencies
from pandas.tseries.offsets import DateOffset, CacheableOffset
from pandas.tseries.frequencies import to_offset
from pandas.tseries import offsets
from datetime import datetime, timedelta
from collections import defaultdict
import os
import glob
import subprocess
import uuid
import pandas.lib as lib

from pandasreg.rperiod import RPeriodIndex, RFrequency, RPeriod

def trim(series):
    """Trim trailing and leading NaN values"""

    ix = np.where(np.isfinite(series))[0]
    if len(ix) == 0:
        return series[0:0]
    return series[ix[0]:(ix[-1]+1)]

def fill(series):
    """Makes a series regularly spaced if it is not so already"""

    if series.index.is_full:
        return series
    ix = RPeriodIndex(start=series.index[0], end=series.index[-1], 
        freq=series.index.freq)
    return series.reindex(ix)

def _agg_first(x):
    ix = np.where(np.isfinite(x))[0]
    if len(ix) == 0:
        return np.nan
    return x[ix[0]]

def _agg_last(x):
    ix = np.where(np.isfinite(x))[0]
    if len(ix) == 0:
        return np.nan
    return x[ix[-1]]

def resample(input, freq, how=None):
    """
    Resample (convert) a time series to another frequency.

    """

    # TODO: allow/disallow partial periods in aggregation
    # TODO: disaggregation performance is too slow, avoid groupby and transform
    #   at least for base cases

    if not isinstance(input.index, RPeriodIndex):
        raise ValueError("Index must be of type RPeriodIndex")

    if isinstance(freq, basestring):
        freq = RFrequency.init(freq)

    if how is None:
        how = input.index.observed

    aggfuncs = {
        "sum": np.sum,
        "mean": np.mean,
        "first": _agg_first,
        "last": _agg_last,
        "min": np.min,
        "max": np.max
    }

    def disagg_start(x):
        x[0] = x[-1]
        x[-1] = np.NaN
        return x

    disaggfuncs = {
        "sum": lambda x: x.fillna(method="backfill")/len(x),
        "mean": lambda x: x.fillna(method="backfill"),
        "first": disagg_start,
        "last": lambda x: x,
        "min": lambda x: x.fillna(method="backfill"),
        "max": lambda x: x.fillna(method="backfill")
    }

    if input.index.freq < freq: # disaggregation
        start = input.index[0].asfreq(freq, how='S')
        end = input.index[-1].asfreq(freq)
        index = RPeriodIndex(start=start, end=end, freq=freq)
        if isinstance(input, pd.Series):
            s = pd.Series(input.values, index=input.index.asfreq(freq))
        elif isinstance(input, pd.DataFrame):
            s = pd.DataFrame(input.values, columns=input.columns, 
                index=input.index.asfreq(freq))

        idx = np.empty(len(s.index)+1, dtype=int)
        idx[0] = start.ordinal-1
        idx[1:] = s.index.values
        lengths = np.diff(idx)
        groups = np.repeat(np.arange(len(lengths)), lengths)

        if isinstance(how, basestring):
            try:
                how = disaggfuncs[how]
            except KeyError:
                raise KeyError("Invalid disaggregation function '%s'" % how)

        s = s.reindex(index).groupby(groups).transform(how)
        s.index = index # otherwise index is Int64 when using DataFrame
        return s

    elif input.index.freq > freq: # aggregation
        start = input.index[0].asfreq(freq)
        end = input.index[-1].asfreq(freq)
        indexnew = RPeriodIndex(start=start, end=end, freq=freq)
        index =indexnew.asfreq(input.index.freq)
        nbins = end-start+1
        idx = np.empty(nbins+1, dtype=int)
        idx[0] = input.index[0].ordinal-1
        idx[1:] = index.values
        idx[-1] = input.index[-1].ordinal
        lengths = np.diff(idx)
        groups = np.repeat(np.arange(len(lengths)), lengths)

        if isinstance(how, basestring):
            try:
                how = aggfuncs[how]
            except KeyError:
                raise KeyError("Invalid aggregation function '%s'" % how)

        s = input.groupby(groups).agg(how)
        s.index = indexnew
        return s

    return input

def overlay(series, replace=True):
    """
    Overlay a list of series on top of each other

    Example: overlay([s1,s2,s3])
    If series overlap, the series that came last in the input list will have
    precedence if replace=True. If replace=false, a series coming later in the 
    list will replace only NA values in the existing list.
    """

    if not isinstance(series, list) and not isinstance(series, tuple):
        raise ValueError("series argument should be list or tuple")

    if len(series) == 0:
        return None

    if len(set([s.index.freq.freqstr for s in series])) > 1:
        raise ValueError("Can only overlay series with the same frequencies")

    start = min([s.index[0].ordinal for s in series])
    end = max([s.index[-1].ordinal for s in series])

    index = RPeriodIndex(start=start, end=end, freq=series[0].index.freq)
    new_series = pd.Series(np.empty(len(index)), index=index)
    new_series[:] = np.nan

    for s in series:
        if replace:
            new_series[s.index[0]:s.index[-1]] = s
        else:
            new_series[s.index[0]:s.index[-1]][np.isnan(new_series[s.index[0]:s.index[-1]])] = \
            s[np.isnan(new_series[s.index[0]:s.index[-1]])]

    return new_series

def extend(input, extender, direction="forward", extender_type="index"):
    """
    Extend a series forward and/or backward using another series or an array.

    Arguments:
        direction: forward, backward
        extender_type: index, pc, pca, diff. If type = pc or pca, the percent
            changes should be in decimal form, i.e. 4% = .04
    """

    # TODO make this makes nicely with DataFrame

    if not isinstance(input, pd.Series) and not isinstance(input, pd.DataFrame):
        raise ValueError("Input must be Series or DataFrame")

    input = trim(input)

    if not direction in ["forward", "backward"]:
        raise ValueError("direction must be 'forward' or 'backward'")

    if not extender_type in ["index","pc","pca","diff"]:
        raise ValueError("extender_type must be 'index', 'pc', 'pca', or 'diff'")

    if isinstance(extender, list):
        extender = np.array(extender)

    if (isinstance(extender, pd.Series) and
        isinstance(extender.index, RPeriodIndex) and
        input.index.freq != input.index.freq):
        raise ValueError("Series and extender series must have same frequency")

    if extender_type == "index":
        if direction == "forward":
            if not input.index[-1] in extender.index:
                return input
            tmp = extender/extender[input.index[-1]]*input[-1]
            return overlay((input, tmp[input.index[-1]+1:]))
        elif direction == "backward":
            if not input.index[0] in extender.index:
                return input
            tmp = extender/extender[input.index[0]]*input[0]
            return overlay((input, tmp[:input.index[0]]))

    elif extender_type == "pc":
        index = RPeriodIndex(start=extender.index[0]-1, periods=len(extender)+1,
            freq=extender.index.freq)
        tmp = pd.Series(np.empty(len(index), dtype=input.dtype), index)
        tmp[0] = 100
        tmp[1:] = 1+extender
        tmp = np.cumprod(tmp)
        return extend(input, tmp, direction=direction, extender_type='index')

    elif extender_type == "pca":
        if (not isinstance(extender, pd.Series) or 
            not isinstance(extender.index, RPeriodIndex)):
            raise ValueError("Can only use extender_type = pca with a time \
                series that has an RPeriodIndex")
        index = RPeriodIndex(start=extender.index[0]-1, periods=len(extender)+1,
            freq=extender.index.freq)
        tmp = pd.Series(np.empty(len(index), dtype=input.dtype), index)
        tmp[0] = 100
        tmp[1:] = (1+extender)**(1.0/input.index.freq.periodicity)
        tmp = np.cumprod(tmp)
        return extend(input, tmp, direction=direction, extender_type='index')

    elif extender_type == "diff":
        if direction == "forward":
            if not input.index[-1]+1 in extender.index:
                return input
            tmp = np.cumsum(extender[input.index[-1]+1:])+input[-1]
            return overlay((input, tmp[input.index[-1]+1:]))
        elif direction == "backward":
            if not input.index[0]-1 in extender.index:
                return input
            tmp = (np.cumsum(-extender[:input.index[0]-1][::-1])+input[0])[::-1]
            return overlay((input, tmp[:input.index[0]-1]))