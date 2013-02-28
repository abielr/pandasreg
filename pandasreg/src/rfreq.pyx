import pandas.tslib as tslib
import pandas as pd
from datetime import datetime
import pandas.core.common as com
import numpy as np
cimport numpy as np
from numpy cimport int64_t

cdef int EPOCH = 1970
cdef int64_t DAYNANO = 1000000000*3600*24

aliases = {
    'A': (RFrequencyM, 12, 11, 1),
    'A-DEC': (RFrequencyM, 12, 11, 1),
    'A-NOV': (RFrequencyM, 12, 10, 1),
    'A-OCT': (RFrequencyM, 12, 9, 1),
    'A-SEP': (RFrequencyM, 12, 8, 1),
    'A-AUG': (RFrequencyM, 12, 7, 1),
    'A-JUL': (RFrequencyM, 12, 6, 1),
    'A-JUN': (RFrequencyM, 12, 5, 1),
    'A-MAY': (RFrequencyM, 12, 4, 1),
    'A-APR': (RFrequencyM, 12, 3, 1),
    'A-MAR': (RFrequencyM, 12, 2, 1),
    'A-FEB': (RFrequencyM, 12, 1, 1),
    'A-JAN': (RFrequencyM, 12, 0, 1),
    'SA': (RFrequencyM, 6, 5, 2),
    'SA-DEC': (RFrequencyM, 6, 5, 2),
    'SA-NOV': (RFrequencyM, 6, 4, 2),
    'SA-OCT': (RFrequencyM, 6, 3, 2),
    'SA-SEP': (RFrequencyM, 6, 2, 2),
    'SA-AUG': (RFrequencyM, 6, 1, 2),
    'SA-JUL': (RFrequencyM, 6, 0, 2),
    'Q': (RFrequencyM, 3, 2, 4),
    'Q-DEC': (RFrequencyM, 3, 2, 4),
    'Q-NOV': (RFrequencyM, 3, 1, 4),
    'Q-OCT': (RFrequencyM, 3, 0, 4),
    'BM': (RFrequencyM, 2, 1, 6),
    'M': (RFrequencyM, 1, 0, 12),
    'TM': (RFrequencyTM, 1, 0, 24),
    'W': (RFrequencyNS, DAYNANO*7, DAYNANO*3, 52),
    'W-SAT': (RFrequencyNS, DAYNANO*7, DAYNANO*2, 52),
    'W-SUN': (RFrequencyNS, DAYNANO*7, DAYNANO*3, 52),
    'W-MON': (RFrequencyNS, DAYNANO*7, DAYNANO*4, 52),
    'W-TUE': (RFrequencyNS, DAYNANO*7, DAYNANO*5, 52),
    'W-WED': (RFrequencyNS, DAYNANO*7, DAYNANO*6, 52),
    'W-THU': (RFrequencyNS, DAYNANO*7, 0, 52),
    'W-FRI': (RFrequencyNS, DAYNANO*7, DAYNANO*1, 52),
    'B': (RFrequencyB, 1, 0, 262),
    'D': (RFrequencyNS, DAYNANO, 0, 365),
    'Hour': (RFrequencyNS, DAYNANO/24, DAYNANO/24, 21900),
    'Min': (RFrequencyNS, DAYNANO/1440, DAYNANO/1440, 525600),
    'Sec': (RFrequencyNS, 1000000000, 0, 31536000)
}

cdef class RFrequency(object):
    """

    The RFrequency class maps date/time values to ordinal periods at a given
    frequency, and can transform a period at one frequency to a period at
    another frequency.

    """

    cdef int64_t stride
    cdef int64_t anchor
    cdef double _periodicity
    cdef _freqstr

    def __init__(self, int64_t stride, int64_t anchor, double periodicity, object freqstr):
        """Do not call this directly; use init() instead"""

        if stride < 1:
            raise ValueError("Stride must be >= 1")
        if periodicity <=0:
            raise ValueError("Periodicity must be > 0")

        self.stride = stride
        self.anchor = anchor
        self._periodicity = periodicity
        self._freqstr = freqstr

    @classmethod
    def init(cls, alias, int64_t stride=1, object anchor=None, double periodicity=-1):
        """

        Use this function to create a new frequency object.

        Arguments:
            alias (str): The frequency string, such as 'A', 'M', or 'D'

            stride (int): The number of periods at the base frequency between
            consecutive observations. For example, setting alias='M' and
            stride=3 creates an observation every 3 months.

            anchor (int, datetime, Timestamp): force the strided observations to
            pass through the given point. For example, a frequency with
            observations every three months could pass through Jan 2000, Feb
            2000, or Mar 2000. Setting datetime(2000,2,1) would force it to pass
            through Feb 2000.

            periodicity (double): The number of observations per year for the
            frequency. If not specified, the 

        Returns:
            An instance of a class inheriting from RFrequency

        """

        if isinstance(anchor, (datetime, pd.Timestamp)):
            freq = RFrequency.init(alias)
            anchor = RFrequency.init(alias).to_ordinal(anchor)
        elif not com.is_integer(anchor) and not anchor is None:
            raise ValueError("Anchor must be an integer, datetime, or Timestamp")

        try:
            _class, _stride, _anchor, _periodicity = aliases[alias]
            _stride = _stride*stride
            if not anchor is None:
                _anchor = anchor
            if periodicity == -1 and stride > 1:
                _periodicity /= stride
            elif periodicity != -1:
                _periodicity = periodicity
            return _class(_stride, _anchor, _periodicity, alias)
        except KeyError:
            raise ValueError("Frequency alias '%s' is not valid" % alias)

    def to_ordinal(self, object dt):
        if not isinstance(dt, pd.Timestamp):
            dt = pd.Timestamp(dt)
        cdef int64_t ordinal = self._to_ordinal(dt)
        cdef int64_t offset = (self.anchor-ordinal) % self.stride
        return (ordinal+offset-self.anchor) / self.stride

    def to_timestamp(self, int64_t ordinal):
        return self._to_timestamp(self.anchor+self.stride*ordinal)

    def asfreq(self, int64_t ordinal, freq, how='E', overlap=True):
        """
        Convert a period at one frequency to a period of another frequency

        Arguments:
            ordinal (int): Ordinal value

            freq (str, RFrequency): Frequency to convert to

            how (str): 'S' (start) or 'E' (end). Only relevant when converting
            from a lower to higher frequency. Determines whether the period
            returned is at the start or the end of the lower-frequency window.

            overlap (bool): Only relevant when converting from a lower to higher
            frequency. Determines the higher-frequency window be allowed to
            extend beyond the end of the lower-frequency window. For example,
            when converting from monthly to weekly, should the week be the last
            week fully contained in the month, or the last week starting in the
            month (but possibly extending into the next month).

        Returns:
            An ordinal at the new frequency
        """

        if isinstance(freq, basestring):
            freq = RFrequency.init(freq)
        elif not isinstance(freq, RFrequency):
            raise ValueError("Frequency must be a string or RFrequency class")

        how = _validate_end_alias(how)

        dt = self.to_timestamp(ordinal)
        cdef int64_t new_ordinal = freq.to_ordinal(dt)

        if self < freq: # disaggregation
            new_dt = freq.to_timestamp(new_ordinal)

            if how == 'E' and not overlap:
                new_ordinal -= 1

            elif how == 'S':
                # Never uses original dt or new_ordinal, so inefficient
                # Doesn't work like I want for B -> D
                new_dt = self.to_timestamp(ordinal-1)
                new_ordinal = freq.to_ordinal(new_dt)
                new_dt2 = freq.to_timestamp(new_ordinal)

                if new_dt2 <= new_dt:
                    new_ordinal += 1

                if not overlap:
                    new_dt2 = freq.to_timestamp(new_ordinal-1)
                    if new_dt2 < new_dt:
                        new_ordinal += 1

        return new_ordinal

    def np_asfreq(self, np.ndarray[int64_t, ndim=1] ordinal, freq, how='E', overlap=True):
        """Same as asfreq(), but accepts and returns a numpy array of ordinals"""

        cdef int i
        cdef np.ndarray[int64_t, ndim=1] result = np.empty((len(ordinal),), dtype=np.int64)

        for i in range(len(ordinal)):
            result[i] = self.asfreq(ordinal[i], freq, how, overlap)

        return result

    def __richcmp__(RFrequency self, RFrequency other, int op):
        if op == 0: # <
            return (self.group == other.group and self.stride > other.stride) \
                or self.group < other.group

        if op == 1: # <=
            return self < other or self == other

        if op == 2: # ==
            return self.freqstr == other.freqstr and self.stride == other.stride and \
                           (self.anchor-other.anchor)%self.stride == 0

        if op == 3: # !=
            return not self == other

        if op == 4: # >
            return (self.group == other.group and self.stride < other.stride) \
                or self.group > other.group

        if op == 5: # >=
            return self > other or self == other

    @property
    def freqstr(self):
        return self._freqstr

    @property
    def periodicity(self):
        return self._periodicity

    def format(self, val):
        """Return a formatted string for the period"""

        if not isinstance(val, (datetime,pd.Timestamp)):
            val = self.to_timestamp(val)
        output = "%04d-%02d-%02d %02d:%02d:%02d" % (val.year, val.month, val.day, 
            val.hour, val.minute, val.second)
        return output

    def __hash__(self):
        return hash((self.group, self.stride, self.anchor, self.periodicity))

cdef class RFrequencyM(RFrequency):
    """Monthly base frequency"""

    group = 0

    def _to_ordinal(self, dt):
        return (dt.year-EPOCH)*12+dt.month-1

    def _to_timestamp(self, int64_t ordinal):
        cdef int month = ordinal % 12 + 1
        cdef int year = (ordinal-month+1)/12+EPOCH
        cdef int day = tslib.monthrange(year, month)[1]
        return pd.Timestamp(datetime(year, month, day))

    def format(self, val):
        if not isinstance(val, datetime):
            val = self.to_timestamp(val)
        output = "%04d-%02d" % (val.year, val.month)
        return output

cdef class RFrequencyTM(RFrequency):
    """Twice monthly base frequency"""

    group = 1

    def _to_ordinal(self, dt):
        cdef int64_t ordinal = (dt.year-EPOCH)*24+(dt.month-1)*2
        if dt.day > 15:
            ordinal += 1
        return ordinal

    def _to_timestamp(self, int64_t ordinal):
        cdef int64_t offset1 = ordinal % 24
        cdef int64_t offset2 = ordinal % 2
        cdef int64_t year = (ordinal-offset1)/24+EPOCH
        cdef int64_t month = (ordinal-(year-EPOCH)*24-offset2)/2+1
        cdef int64_t day
        if offset2 == 0:
            day = 15
        else:
            day = tslib.monthrange(year, month)[1]
        return pd.Timestamp(datetime(year, month, day))

cdef class RFrequencyB(RFrequency):
    """Business daily (Mon-Fri) base frequency"""

    group = 2

    def _to_ordinal(self, dt):
        cdef int64_t days = dt.value/DAYNANO
        cdef int64_t wday = dt.weekday()
        if wday < 5:
            return (days-wday+3)/7*5+wday-3
        return (days-wday+3)/7*5+2

    def _to_timestamp(self, int64_t ordinal):
        cdef int wday = (ordinal+3) % 5
        return pd.Timestamp(((ordinal-wday+3)/5*7+wday-3)*DAYNANO)

cdef class RFrequencyNS(RFrequency):
    """Nanosecond base frequency"""

    group = 4

    def _to_ordinal(self, dt):
        return pd.Timestamp(dt).value

    def _to_timestamp(self, int64_t ordinal):
        return pd.Timestamp(ordinal)

def _validate_end_alias(how):
    how_dict = {'S': 'S', 'E': 'E',
                'START': 'S', 'FINISH': 'E',
                'BEGIN': 'S', 'END': 'E'}
    how = how_dict.get(str(how).upper())
    if how not in set(['S', 'E']):
        raise ValueError('How must be one of S or E')
    return how