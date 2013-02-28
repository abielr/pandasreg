""" 

Shows basic usage the the package. Much of this is a kind of test to make
sure it works with standard pandas functions. 

"""

from datetime import datetime
import numpy as np
from scipy.interpolate import interp1d
import statsmodels.api as sm
import pandas as pd

from pandasreg import RFrequency, RPeriod, RPeriodIndex
import pandasreg as pdr

############################################
# INITIALIZING A TIME SERIES
############################################

# Using a list of dates and values
dates = [datetime(2007,12,31),datetime(2008,12,31),datetime(2009,12,31)]
values = [1,2,3]
s = pd.Series(values, RPeriodIndex(dates, freq="A"))

# Using a starting date and list of values
values = [1,2,3,4,5,6,7,8,9,10]
s = pd.Series(values, RPeriodIndex(start=datetime(2010,1,31), periods=len(values), freq="M"))

# Using a starting and ending dates and list of values
values = [1,2,3,4,5,6,7,8,9,10]
start = RPeriod(datetime(2007,1,1), "D")
end = start+len(values)-1
s = pd.Series(values, RPeriodIndex(start=start, end=end))

# SUPPORTED FREQUENCIES (CAN EASILY BE EXTENDED)
# Annual: A, A-NOV, A-OCT, ..., A-JAN
# Semiannual: SA, SA-NOV, ..., SA-JUL
# Quarterly: Q, Q-NOV, Q-OCT
# Bimontly: BM
# Monthly: M
# Twice-monthly: TM
# Weekly: W-MON, W-TUE, ..., W-SUN
# Business: B
# Daily: D

############################################
# BASIC MATH
############################################

# Setup some series with same frequency but different date ranges
s1 = pd.Series(np.random.randn(100), RPeriodIndex(start=datetime(2007,1,31), periods=100, freq="M"))
s2 = pd.Series(np.random.randn(50), RPeriodIndex(start=datetime(2007,1,31), periods=50, freq="M"))

s1+s2
s1-s2
s1/s2
s1*s2
s1**s2
0.75*s1+0.25*s2

############################################
# ACCESSING AND SETTING PARTS OF A SERIES
############################################

s1[0]										# Get first observation
s1[-1]										# Get last observation
s1[:5]										# Get first 5 observations
s1[2:5]										# Get observations 3 through 5
s1[-5:]										# Get last 5 observations
s1[datetime(2007,1,31)]						# Get Jan-2007
s1[datetime(2007,1,1)]						# Get Jan-2007
s1["2007-01"]								# Get Jan-2007
s1["2007Q1"]								# Get Jan-Mar 2007
s1[datetime(2007,1,31):datetime(2007,6,30)]	# Get Jan-2007 through Jun-2008
s1["2007-01":"2008-06"]						# Get Jan-2007 through Jun-2008
s1["2007Q1":"2008Q2"]						# Get Jan-2007 through Jun-2008
s1[s1>0]									# Get observations larger than 0

s1[0] = 100
s1[2:5] = [1,2,3]
s1[datetime(2007,1,31)] = 100
s1["2007-01"] = 100
s1["2007-01":"2007-03"] = [1,2,3]
s1[s1>0] = 100

############################################
# SPLICING
############################################

# OVERLAY SERIES TWO OR MORE SERIES
ix1 = RPeriodIndex(start=datetime(2000,1,1), periods=6, freq="M")
ix2 = RPeriodIndex(start=datetime(2000,4,1), periods=6, freq="M")
s1 = pd.Series(np.array([1,2,3,4,5,6], dtype=np.float64), ix1)
s2 = pd.Series(np.array([1,2,3,4,5,6], dtype=np.float64), ix2)

pdr.overlay([s1,s2])				# Splice and overwrite s1 with s2 where possible
pdr.overlay([s1,s2], False)			# Splice and overwrite s1 with s2 if s1 is NA

# EXTEND A SERIES FORWARD OR BACKWARD USING ANOTHER SERIES

ix1 = RPeriodIndex(start=datetime(2000,1,1), periods=12, freq="M")
ix2 = RPeriodIndex(start=datetime(2000,9,1), periods=12, freq="M")
ix3 = RPeriodIndex(start=datetime(1999,3,1), periods=12, freq="M")
s1 = pd.Series(np.arange(1, 13, dtype=np.float64), ix1)
s2 = pd.Series(np.arange(1, 13, dtype=np.float64), ix2)
s3 = pd.Series(np.arange(1, 13, dtype=np.float64), ix3)

# Grow a series out using the growth rate of another series, For example,
# extend an index or money series using another index or money series
pdr.extend(s1, s2, direction="forward", extender_type="index")
pdr.extend(s1, s3, direction="backward", extender_type="index")

# Use an extender series that is defined in percent change form
s2 = pd.Series(np.array([.2,1.1,-.3,3.2,.5,.5,.1,-1.1,.4,-0.7,.1,.5]), ix2)
pdr.extend(s1, s2, direction="forward", extender_type="pc")

# Use an extender series that is defined in annualized percent change form
pdr.extend(s1, s2, direction="forward", extender_type="pca")

# Use an extender series that is defined as level differences
pdr.extend(s1, s2, direction="forward", extender_type="diff")

############################################
# FREQUENCY CONVERSIONS
############################################

s1 = pd.Series(np.arange(24), RPeriodIndex(start=datetime(2007,1,31), periods=24, freq="M"))

pdr.resample(s1, "Q", how='sum')
pdr.resample(s1, "Q", how='mean')
pdr.resample(s1, "Q", how='min')
pdr.resample(s1, "Q", how='max')
pdr.resample(s1, "Q", how='first')
pdr.resample(s1, "Q", how='last')

pdr.resample(s1, "D", how='sum')
pdr.resample(s1, "D", how='mean')
pdr.resample(s1, "D", how='min')
pdr.resample(s1, "D", how='max')
pdr.resample(s1, "D", how='first')
pdr.resample(s1, "D", how='last')

############################################
# ROLLING FUNCTIONS
############################################

s1 = pd.Series(np.random.randn(100), RPeriodIndex(start=datetime(2007,1,31), periods=100, freq="M"))
s2 = pd.Series(np.random.randn(50), RPeriodIndex(start=datetime(2007,1,31), periods=50, freq="M"))

pd.rolling_count(s1, 5)					# Number of non-NA observations
pd.rolling_sum(s1, 5, min_periods=1)	# Rolling sum, at least one observation must be non-NA
pd.rolling_mean(s1, 5, min_periods=5)	# Rolling mean, all observations must be non-NA
pd.rolling_median(s1, 5)				# Rolling median
pd.rolling_min(s1, 5)					# Rolling min
pd.rolling_max(s1, 5)					# Rolling max
pd.rolling_std(s1, 5)					# Rolling standard deviation
pd.rolling_var(s1, 5)					# Rolling variance
pd.rolling_skew(s1, 5)					# Rolling skew
pd.rolling_kurt(s1, 5)					# Rolling kurtosis
pd.rolling_quantile(s1, 5, 0.3)			# Rolling quantile
pd.rolling_cov(s1, s2, 5)				# Rolling covariance
pd.rolling_corr(s1, s2, 5)				# Rolling correlation

############################################
# EXPONENTIALLY WEIGHTED MOMENT FUNCTIONS
############################################

pd.ewma(s1, 0.5)						# EW weighted average
pd.ewmvar(s1, 0.5)						# EW weighted variance
pd.ewmstd(s1, 0.5)						# EW moving standard deviation
pd.ewmcorr(s1, s2, 0.5)					# EW moving correlation
pd.ewmcov(s1, s2, 0.5)					# EW moving covariance

############################################
# BASIC STATS
############################################

ix = RPeriodIndex(start=datetime(1990,1,1), periods=100, freq="M")
s1 = pd.Series(np.random.randn(len(ix)), ix)

s1.shift(1)							# Lagging/leading a series
s1.cov(s2)							# Coveriance between series
s2.corr(s2)							# Correlation between series
pdr.d(s1, 1)						# Difference
pdr.da(s1, 1)						# Difference, annualized
pdr.dy(s1, 1)						# Difference over N years
pdr.dya(s1, 1)						# Difference over N years, annualized
pdr.pc(s1, 1)						# Percent change
pdr.pca(s1, 1)						# Percent change, annualized
pdr.pcy(s1, 1)						# Percent change over N years
pdr.pcya(s1, 1)						# Percent change over N years, annualized
pdr.logd(s1, 1)						# Log difference
pdr.logda(s1, 1)					# Log difference, annualized
pdr.logdy(s1, 1)					# Log difference over N years
pdr.logdya(s1, 1)					# Log difference over N years, annualized

# See also http://docs.scipy.org/doc/numpy/reference/routines.statistics.html

############################################
# FILL MISSING VALUES
############################################

ix = RPeriodIndex(start=datetime(1990,1,1), periods=6, freq="M")
s1 = pd.Series(np.array([1.2,np.nan,4.5,2.6,np.nan,6.7]), ix)

s1.fillna(method='ffill')			# Forward fill
s1.fillna(method='bfill')			# Backward fill
s1.interpolate()					# Linear interpolation

############################################
# SEASONAL ADJUSTMENT
############################################

ix = RPeriodIndex(start=datetime(1990,1,1), periods=120, freq="M")
s1 = pd.Series(np.arange(1,121), ix)

# To run this, you must have X-12 installed and provide its location
# pdr.x12(s1, [EXECUTABLE PATH], [TMP DIR PATH])

############################################
# REGRESSIONS
############################################

s1 = pd.Series(np.random.randn(100), RPeriodIndex(start=datetime(2007,1,31), periods=100, freq="M"))
s2 = pd.Series(np.random.randn(50), RPeriodIndex(start=datetime(2007,1,31), periods=50, freq="M"))
s3 = pd.Series(np.random.randn(75), RPeriodIndex(start=datetime(2007,1,31), periods=75, freq="M"))

# Ordinary linear regression on time series
ols = pd.ols(y=s1, x={'x1': s2, 'x2': s3})
# Useful output parameters include ols.beta, ols.y_fitted; do print(ols) for summary

# Rolling regression
pd.ols(y=s1, x={'x1': s2, 'x2': s3}, window='24')

# Expanding regression using minimum of 12 periods
pd.ols(y=s1, x={'x1': s2, 'x2': s3}, window_type='expanding', min_periods=12)

# Regression will continue to work with missing data
s2[0] = np.nan
pd.ols(y=s1, x={'x1': s2, 'x2': s3})

############################################
# OTHER STATISTICAL PROCEDURES
############################################

ix1 = RPeriodIndex(start=datetime(1990,1,1), periods=100, freq="M")
s1 = pd.Series(np.random.randn(len(ix1)), ix1)

# Hodrick-Prescott Filter
cycle, trend = sm.tsa.filters.hpfilter(s1, 1600)