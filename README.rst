pandasReg
---------

pandasReg is an extension for `pandas <http://pandas.pydata.org/>`_ that adds some extra functionality for dealing with regularly-spaced time series data. It adds a new time series index `RPeriodIndex` that behaves very similarly to the built-in pandas index `PeriodIndex` but with some of the flexibility of pandas' `DatetimeIndex`. Essentially, `RPeriodIndex` is a variant of `PeriodIndex` that adds more frequencies and is easier to extend to new frequencies. There is also a function to support resampling between any combination of frequencies (`PeriodIndex` does not allow every possible combination, such as going from monthly to weekly).

For example usage, see the examples/ directory.