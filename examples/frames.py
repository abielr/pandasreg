"""
Some examples using the pandas DataFrame
"""

from datetime import datetime
import numpy as np
import pandas as pd

from pandasreg import RFrequency, RPeriod, RPeriodIndex
import pandasreg as pdr

############################################
# Example: create a series of regional industrial production indices
############################################

countries = ['a','b','c','d','e','f','g']
regions = {
	'region1': ['a','c','f'],
	'region2': ['b','c','g'],
	'region3': ['c','d','e','f','g']
}

# Weights will be nominal GDP, annual frequency
ix = RPeriodIndex(start="1/1/2000", end="12/31/2009", freq="A")
gdp = pd.DataFrame(np.random.randint(1,10,(len(ix), len(countries))), 
	index=ix, columns=countries)

# Create IP growth rates, monthly frequency
ix = RPeriodIndex(start="1/1/2000", end="12/31/2010", freq="M")
ip = pd.DataFrame(np.random.uniform(-.2, .2, (len(ix), len(countries))), 
	index=ix, columns=countries)

# Out original GDP weights are annual and don't extend through 2010.
gdpm = pdr.resample(gdp, "M") # Convert to monthly
gdpm = gdpm.reindex(ip.index) # Align GDP time index with IP
gdpm = gdpm.fillna(method='ffill') # Forward fill: set 2010 GDP weights equal to 2009

# Compute regional IP index for each region
for name, region in regions.items():
	gdpwt = gdpm[region].divide(gdpm[region].sum(axis=1), axis=0) # Compute regional weights
	changes = (ip[region]*gdpwt).sum(axis=1, skipna=False) # Sum of weighted changes
	changes = pdr.trim(changes) # Trim any NAs from start and end of series

	# Create base period and extend it out with percent changes
	# Percent changes begin in 2000, so start of index will be Dec 1999
	regionip = pd.Series(100.0, RPeriodIndex(start=changes.index[0]-1, periods=1,
		freq="M"))
	regionip = pdr.extend(regionip, changes, extender_type='pc')

# Suppose there is NAs in our IP data, and that for each region we want the
# 	growth rate for the region to be equal to the growth rate of only the
# 	available indexes on each date (easy to extend this to taking the growth rate
#	of a subset of countries who report early)

# Make the last five observations of some of our series NA
ip['a'][-5:] = np.nan
ip['f'][-5:] = np.nan

# Basically same as before, but we adjust the weights
for name, region in regions.items():
	gdpm_tmp = gdpm[region].copy()
	gdpm_tmp[np.isnan(ip[region])] = np.nan
	gdpwt = gdpm_tmp.divide(gdpm_tmp.sum(axis=1), axis=0)
	changes = (ip[region]*gdpwt).sum(axis=1, skipna=True)
	changes = pdr.trim(changes)
	regionip = pd.Series(100.0, RPeriodIndex(start=changes.index[0]-1, periods=1, freq="M"))
	regionip = pdr.extend(regionip, changes, extender_type='pc')