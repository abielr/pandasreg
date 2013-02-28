import numpy as np
from datetime import datetime, date
import dateutil.parser
from pandas.tseries.index import Int64Index, Index
import pandas as pd
import pandas.core.common as com
import pandas._algos as _algos
from pandas.tseries.tools import parse_time_string
from rfreq import RFrequency

class RPeriod(object):
    """
    Represents a regularly spaced period, which can be incremented, decremented,
    and compared to other periods of the same frequency.
    """

    def __init__(self, value=None, freq=None, ordinal=None):
        """

        Arguments:

            value (RPeriod, datetime, Timestamp, date, str): a date to be
            converted to a period at the given frequency. Either the value or
            ordinal must be supplied, but not both.

            freq (str, RFrequency): frequency of the period

            ordinal (int): an ordinal period at the given frequency. Either the
            value or ordinal must be supplied, but not both.

        """

        if freq is not None:
            if isinstance(freq, basestring):
                self.freq = RFrequency.init(freq)
            elif isinstance(freq, RFrequency):
                self.freq = freq
            else:
                raise ValueError("Frequency must be a string or frequency class")

        if ordinal is not None and value is not None:
            raise ValueError("Only value or ordinal but not both should be given")

        elif ordinal is not None:
            if not com.is_integer(ordinal):
                raise ValueError("Ordinal must be an integer")
            if freq is None:
                raise ValueError("Must supply freq for ordinal value")
            self.ordinal = ordinal

        elif isinstance(value, RPeriod):
            if freq is None or value.freq == self.freq:
                # print("HERE")
                self.ordinal = value.ordinal
                self.freq = value.freq
            else:
                self.ordinal = value.asfreq(self.freq)
        elif isinstance(value, (datetime, pd.Timestamp)):
            if freq is None:
                raise ValueError("Must supply freq for datetime/Timestamp value")
            self.ordinal = self.freq.to_ordinal(value)
        elif isinstance(value, date):
            if freq is None:
                raise ValueError("Must supply freq for datetime value")
            self.ordinal = self.freq.to_ordinal(datetime(value.year, value.month, value.day))
        elif isinstance(value, basestring):
            self.ordinal = _string_to_period(value).asfreq(freq).ordinal
        else:
            raise ValueError("Value must be RPeriod, integer, datetime/date, or valid string")

    def __lt__(self, other):
        if isinstance(other, RPeriod):
            if self.freq != other.freq:
                raise ValueError("Can only compare periods with same frequency")
            return self.ordinal < other.ordinal
        raise ValueError("Can only compare to another RPeriod")

    def __le__(self, other):
        return self < other or self == other

    def __eq__(self, other):
        if isinstance(other, RPeriod):
            return self.ordinal == other.ordinal and self.freq == other.freq
        raise ValueError("Can only compare to another RPeriod")

    def __ne__(self, other):
        return not self == other

    def __ge__(self, other):
        return self > other or self == other

    def __gt__(self, other):
        if isinstance(other, RPeriod):
            if self.freq != other.freq:
                raise ValueError("Can only compare periods with same frequency")
            return self.ordinal > other.ordinal
        raise ValueError("Can only compare to another RPeriod")

    def __hash__(self):
        return hash((self.ordinal, self.freq))

    def __add__(self, other):
        if com.is_integer(other):
            return RPeriod(ordinal=self.ordinal + other, freq=self.freq)
        else:
            raise TypeError(other)

    def __sub__(self, other):
        if com.is_integer(other):
            return RPeriod(ordinal=self.ordinal - other, freq=self.freq)
        if isinstance(other, RPeriod):
            if other.freq != self.freq:
                raise ValueError("Cannot do arithmetic with non-conforming periods")
            return self.ordinal - other.ordinal
        else:
            raise TypeError(other)

    def asfreq(self, freq, how='E', overlap=True):
        return RPeriod(ordinal=self.freq.asfreq(self.ordinal, freq, how, 
            overlap), freq=freq)

    def to_timestamp(self):
        # TODO: support freq, how option
        return self.freq.to_timestamp(self.ordinal)

    def to_datetime(self, freq=None):
        return self.to_timestamp().to_pydatetime()

    def __repr__(self):
        return self.freq.to_timestamp(self.ordinal)

    def __str__(self):
        dt = self.freq.to_timestamp(self.ordinal)
        return self.freq.format(dt)

    def strftime(self, fmt):
        return self.freq.to_timestamp(self.ordinal).strftime(fmt)

class RPeriodIndex(Int64Index):
    """

    This class is based on pandas' PeriodIndex and the initalization
    arguments are almost the same. The one additional argument is `observed`.

    Arguments:
        data: a list of datetimes, Timestamps, or datetime strings

        ordinal: a list of ordinal periods that can be provided instead of the
        data argument.

        freq (str, RFrequency): frequency of the index

        start: starting period

        end: ending period

        periods: # of periods

        name: a name for the index

        observed: this option controls how a Series or DataFrame will be
        resampled if the user does not provide an explicit method. Options can
        be any of those that are provided to the pandas resample function.

    """

    def __new__(cls, data=None, ordinal=None,
                freq=None, start=None, end=None, periods=None,
                name=None, observed=None):

        if freq is None:
            if start is not None and isinstance(start, RPeriod):
                freq = start.freq
            elif end is not None and isinstance(end, RPeriod):
                freq = end.freq
            else:
                raise ValueError("Must supply frequency")

        if isinstance(freq, basestring):
            freq = RFrequency.init(freq)

        if data is None:
            if ordinal is not None:
                data = np.asarray(ordinal, dtype=np.int64)
            else:
                data = cls._get_ordinal_range(start, end, periods, freq)
        else:
            ordinal = cls._from_arraylike(data, freq)
            data = np.array(ordinal, dtype=np.int64, copy=False)

        if observed is None:
            observed = "mean"

        subarr = data.view(cls)
        subarr.name = name
        subarr.freq = freq
        subarr.observed = observed

        return subarr

    @classmethod
    def _get_ordinal_range(cls, start, end, periods, freq):
        if isinstance(start, datetime):
            start = freq.to_ordinal(start)
        elif isinstance(start, RPeriod):
            start = start.ordinal
        elif isinstance(start, basestring):
            start = _string_to_period(start).asfreq(freq).ordinal
        if isinstance(end, datetime):
            end = freq.to_ordinal(end)
        elif isinstance(end, RPeriod):
            end = end.ordinal
        elif isinstance(end, basestring):
            end = _string_to_period(end).asfreq(freq).ordinal

        if periods is not None:
            if start is None:
                data = np.arange(end - periods + 1, end + 1, dtype=np.int64)
            else:
                data = np.arange(start, start + periods, dtype=np.int64)
        else:
            data = np.arange(start, end+1, dtype=np.int64)

        return data

    @classmethod
    def _from_arraylike(cls, data, freq):
        if not isinstance(data, np.ndarray):
            data = [freq.to_ordinal(datetime(x.year, x.month, x.day)) for x in data]
        else:
            if isinstance(data, RPeriodIndex):
                if freq == data.freq:
                    data = data.values
                else:
                    pass
            else:
                pass

        return data

    def asfreq(self, freq, how='E', overlap=True):
        """Convert the periods in the index to another frequency.

        See the RFrequency.asfreq() documention for more information
        """

        if isinstance(freq, basestring):
            freq = RFrequency.init(freq)
        if freq.freqstr == self.freq.freqstr:
            return self

        return type(self)(ordinal=self.freq.np_asfreq(self.values, freq, how, 
            overlap), freq=freq)

    @property
    def freqstr(self):
        """String representation of the index's frequency"""

        return self.freq.freqstr

    def __contains__(self, key):
        if not isinstance(key, RPeriod) or key.freq != self.freq:
            if isinstance(key, basestring):
                try:
                    self.get_loc(key)
                    return True
                except Exception:
                    return False
            return False
        return key.ordinal in self._engine

    @property
    def is_full(self):
        """
        Returns True if there are any missing periods from start to end
        """
        if len(self) == 0:
            return True
        if not self.is_monotonic:
            raise ValueError('Index is not monotonic')
        values = self.values
        return ((values[1:] - values[:-1]) < 2).all()

    def __array_finalize__(self, obj):
        if self.ndim == 0:  # pragma: no cover
            return self.item()

        self.freq = getattr(obj, 'freq', None)
        self.observed = getattr(obj, 'observed', None)

    def map(self, f):
        try:
            return f(self)
        except:
            values = self._get_object_array()
            return _algos.arrmap_object(values, f)

    def shift(self, n):
        if n == 0:
            return self

        return RPeriodIndex(data=self.values + n, freq=self.freq, 
            observed=self.observed)

    def __add__(self, other):
        return RPeriodIndex(ordinal=self.values + other, freq=self.freq, 
            observed=self.observed)

    def __sub__(self, other):
        return RPeriodIndex(ordinal=self.values - other, freq=self.freq, 
            observed=self.observed)

    def __getitem__(self, key):
        arr_idx = self.view(np.ndarray)
        if np.isscalar(key):
            val = arr_idx[key]
            return RPeriod(ordinal=val, freq=self.freq)
        else:
            if com._is_bool_indexer(key):
                key = np.asarray(key)

            result = arr_idx[key]
            if result.ndim > 1:
                # MPL kludge
                # values = np.asarray(list(values), dtype=object)
                # return values.reshape(result.shape)
                return RPeriodIndex(result, name=self.name, freq=self.freq, 
                    observed=self.observed)

            return RPeriodIndex(result, name=self.name, freq=self.freq, 
                observed=self.observed)

    def join(self, other, how='left', level=None, return_indexers=False):
        self._assert_can_do_setop(other)

        result = Int64Index.join(self, other, how=how, level=level,
                                return_indexers=return_indexers)

        if return_indexers:
            result, lidx, ridx = result
            return self._apply_meta(result), lidx, ridx
        else:
            return self._apply_meta(result)

    def _assert_can_do_setop(self, other):
        if not isinstance(other, RPeriodIndex):
            raise ValueError('can only call with other RPeriodIndex-ed objects')

        if self.freq != other.freq:
            raise ValueError('Only like-indexed RPeriodIndexes compatible for join')

    def _wrap_union_result(self, other, result):
        name = self.name if self.name == other.name else None
        result = self._apply_meta(result)
        result.name = name
        return result

    def _apply_meta(self, rawarr):
        idx = rawarr.view(RPeriodIndex)
        idx.freq = self.freq
        idx.observed = self.observed;
        return idx

    def __iter__(self):
        for val in self.values:
#            yield RPeriod(ordinal=val, freq=self.freq)
            yield val

    @property
    def inferred_type(self):
        # b/c data is represented as ints make sure we can't have ambiguous
        # indexing
        return 'period'

    def get_value(self, series, key):
        try:
            return super(RPeriodIndex, self).get_value(series, key)
        except (KeyError, IndexError):
            try:
                period = _string_to_period(key)

                vals = self.values

                # if our data is higher resolution than requested key, slice
                if period.freq < self.freq:
                    ord1 = period.asfreq(self.freq, how='S').ordinal
                    ord2 = period.asfreq(self.freq, how='E').ordinal

                    if ord2 < vals[0] or ord1 > vals[-1]:
                        raise KeyError(key)

                    pos = np.searchsorted(self.values, [ord1, ord2])
                    key = slice(pos[0], pos[1] + 1)
                    return series[key]
                else:
                    key = period.asfreq(self.freq)
                    return self._engine.get_value(series, key.ordinal)
            except TypeError:
                pass
            except KeyError:
                pass

            key = RPeriod(key, self.freq)
            return self._engine.get_value(series, key.ordinal)

    def get_loc(self, key):
        try:
            return self._engine.get_loc(key)
        except KeyError:
            if com.is_integer(key):
                return key
            try:
                key = _string_to_period(key)
            except TypeError:
                pass

            key = RPeriod(key, self.freq).ordinal
            return self._engine.get_loc(key)

    def slice_locs(self, start=None, end=None):
        """
        Index.slice_locs, customized to handle partial ISO-8601 string slicing
        """
        if isinstance(start, basestring) or isinstance(end, basestring):
            try:
                if start:
                    start_loc = self._get_string_slice(start).start
                else:
                    start_loc = 0

                if end:
                    end_loc = self._get_string_slice(end).stop
                else:
                    end_loc = len(self)

                return start_loc, end_loc
            except KeyError:
                pass

        if isinstance(start, datetime) and isinstance(end, datetime):
            ordinals = self.values
            t1 = RPeriod(start, freq=self.freq)
            t2 = RPeriod(end, freq=self.freq)

            left = ordinals.searchsorted(t1.ordinal, side='left')
            right = ordinals.searchsorted(t2.ordinal, side='right')
            return left, right

        return Int64Index.slice_locs(self, start, end)

    def _get_string_slice(self, key):
        if not self.is_monotonic:
            raise ValueError('Partial indexing only valid for '
                             'ordered time series')

        t1 = _string_to_period(key)

        ordinals = self.values

        t2 = t1.asfreq(self.freq, how='end')
        t1 = t1.asfreq(self.freq, how='start')

        left = ordinals.searchsorted(t1.ordinal, side='left')
        right = ordinals.searchsorted(t2.ordinal, side='right')
        return slice(left, right)

    def take(self, indices, axis=None):
        """
        Analogous to ndarray.take
        """
        taken = self.values.take(indices.astype('int32'), axis=axis)
        taken = taken.view(RPeriodIndex)
        taken.freq = self.freq
        taken.name = self.name
        return taken

    def format(self, name=False, formatter=None):
        """
        Render a string representation of the Index
        """
        header = []

        if name:
            header.append(str(self.name) if self.name is not None else '')

        return header + ['%s' % RPeriod(ordinal=x, freq=self.freq) for x in self]

def _validate_end_alias(how):
    how_dict = {'S': 'S', 'E': 'E',
                'START': 'S', 'FINISH': 'E',
                'BEGIN': 'S', 'END': 'E'}
    how = how_dict.get(str(how).upper())
    if how not in set(['S', 'E']):
        raise ValueError('How must be one of S or E')
    return how

def _string_to_period(value, freq=None):
    asdt, parsed, reso = parse_time_string(value, freq=freq)

    if reso == 'year':
        freq = 'A'
    elif reso == 'quarter':
        freq = 'Q'
    elif reso == 'month':
        freq = 'M'
    else:
        freq = 'D'

    return RPeriod(asdt, freq=freq)