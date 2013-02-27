import numpy as np
import numpy.testing as npt
from nose.tools import *
from datetime import datetime
import pandas as pd

from pandasreg.rperiod import RFrequency, RPeriod, RPeriodIndex

class TestClass:
	def setUp(self):
		pass

	def tearDown(self):
		pass

	def test_periods(self):
		dt = datetime(2013,1,1)
		assert RPeriod(dt, freq="D").to_timestamp() == pd.Timestamp(datetime(2013,1,1))
		assert RPeriod(dt, freq="B").to_timestamp() == pd.Timestamp(datetime(2013,1,1))
		assert RPeriod(dt, freq="W-MON").to_timestamp() == pd.Timestamp(datetime(2013,1,7))
		assert RPeriod(dt, freq="W-TUE").to_timestamp() == pd.Timestamp(datetime(2013,1,1))
		assert RPeriod(dt, freq="W-WED").to_timestamp() == pd.Timestamp(datetime(2013,1,2))
		assert RPeriod(dt, freq="W-THU").to_timestamp() == pd.Timestamp(datetime(2013,1,3))
		assert RPeriod(dt, freq="W-FRI").to_timestamp() == pd.Timestamp(datetime(2013,1,4))
		assert RPeriod(dt, freq="W-SAT").to_timestamp() == pd.Timestamp(datetime(2013,1,5))
		assert RPeriod(dt, freq="W-SUN").to_timestamp() == pd.Timestamp(datetime(2013,1,6))
		assert RPeriod(dt, freq="TM").to_timestamp() == pd.Timestamp(datetime(2013,1,15))
		assert RPeriod(dt, freq="M").to_timestamp() == pd.Timestamp(datetime(2013,1,31))
		assert RPeriod(dt, freq="Q").to_timestamp() == pd.Timestamp(datetime(2013,3,31))
		assert RPeriod(dt, freq="SA").to_timestamp() == pd.Timestamp(datetime(2013,6,30))
		assert RPeriod(dt, freq="A").to_timestamp() == pd.Timestamp(datetime(2013,12,31))

	def test_asfreq(self):
		dt = datetime(2013,1,1)
		assert RPeriod(dt, freq="D").asfreq("A").to_timestamp() == pd.Timestamp(datetime(2013,12,31))
		assert RPeriod(dt, freq="B").asfreq("A").to_timestamp() == pd.Timestamp(datetime(2013,12,31))
		assert RPeriod(dt, freq="W-MON").asfreq("A").to_timestamp() == pd.Timestamp(datetime(2013,12,31))
		assert RPeriod(dt, freq="W-TUE").asfreq("A").to_timestamp() == pd.Timestamp(datetime(2013,12,31))
		assert RPeriod(dt, freq="W-WED").asfreq("A").to_timestamp() == pd.Timestamp(datetime(2013,12,31))
		assert RPeriod(dt, freq="W-THU").asfreq("A").to_timestamp() == pd.Timestamp(datetime(2013,12,31))
		assert RPeriod(dt, freq="W-FRI").asfreq("A").to_timestamp() == pd.Timestamp(datetime(2013,12,31))
		assert RPeriod(dt, freq="W-SAT").asfreq("A").to_timestamp() == pd.Timestamp(datetime(2013,12,31))
		assert RPeriod(dt, freq="W-SUN").asfreq("A").to_timestamp() == pd.Timestamp(datetime(2013,12,31))
		assert RPeriod(dt, freq="TM").asfreq("A").to_timestamp() == pd.Timestamp(datetime(2013,12,31))
		assert RPeriod(dt, freq="M").asfreq("A").to_timestamp() == pd.Timestamp(datetime(2013,12,31))
		assert RPeriod(dt, freq="BM").asfreq("A").to_timestamp() == pd.Timestamp(datetime(2013,12,31))
		assert RPeriod(dt, freq="Q").asfreq("A").to_timestamp() == pd.Timestamp(datetime(2013,12,31))
		assert RPeriod(dt, freq="A").asfreq("A").to_timestamp() == pd.Timestamp(datetime(2013,12,31))

		assert RPeriod(dt, freq="A").asfreq("D", how='E').to_timestamp() == pd.Timestamp(datetime(2013,12,31))
		assert RPeriod(dt, freq="A").asfreq("D", how='S').to_timestamp() == pd.Timestamp(datetime(2013,1,1))
		assert RPeriod(dt, freq="A").asfreq("B", how='E').to_timestamp() == pd.Timestamp(datetime(2013,12,31))
		assert RPeriod(dt, freq="A").asfreq("B", how='S').to_timestamp() == pd.Timestamp(datetime(2013,1,1))
		assert RPeriod(dt, freq="A").asfreq("W-MON", how='E').to_timestamp() == pd.Timestamp(datetime(2014,1,6))
		assert RPeriod(dt, freq="A").asfreq("W-MON", how='S').to_timestamp() == pd.Timestamp(datetime(2013,1,7))

	def test_indexing(self):
		index = RPeriodIndex(start=datetime(2013,1,1), periods=50, freq="M")
		s = pd.Series(np.arange(len(index)), index)

		assert s[0] == 0
		assert s[-1] == 49
		assert s[datetime(2013,1,1)] == 0
		assert s["2013-01"] == 0
		assert s[RPeriod(datetime(2013,1,1), freq="M")] == 0

	@raises(KeyError)
	def test_indexing_exception(self):
		index = RPeriodIndex(start=datetime(2013,1,1), periods=50, freq="M")
		s = pd.Series(np.arange(len(index)), index)
		s[datetime(2020,1,1)]

	def test_rperiodindex(self):
		ix = RPeriodIndex(start=datetime(2010,1,1), periods=50, freq="M")
		ix = RPeriodIndex(start=RPeriod(datetime(2010,1,1), freq="M"), periods=50)
		ix = RPeriodIndex(start="1/1/2000", periods=50, freq="M")
		ix = RPeriodIndex(start="2000-01-01", periods=50, freq="M")

if __name__ == "__main__":
	import nose
	nose.run(argv=["-w", __file__,"--with-coverage"])