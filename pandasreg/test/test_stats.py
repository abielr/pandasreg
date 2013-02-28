from datetime import datetime
import numpy as np
import pandas as pd
from pandasreg.rperiod import RFrequency, RPeriod, RPeriodIndex
import pandasreg as pdr

class TestClass:
	def setUp(self):
		pass

	def tearDown(self):
		pass

	def test_basic(self):
		index = RPeriodIndex(start=datetime(1970,1,1), periods=50, freq="M")
		s1 = pd.Series(np.arange(1, len(index)+1, dtype=np.float64), index)

		s = pdr.d(s1)
		assert s[1] == s1[1]-s1[0]
		s = pdr.da(s1)
		assert s[1] == (s1[1]-s1[0])*12
		s = pdr.dy(s1)
		assert s[12] == (s1[12]-s1[0])
		s = pdr.dya(s1,2)
		assert s[24] == (s1[24]-s1[0])/2

		s = pdr.logd(s1)
		assert s[1] == (np.log(s1[1])-np.log(s1[0]))*100
		s = pdr.logda(s1)
		assert s[1] == (np.log(s1[1])-np.log(s1[0]))*1200
		s = pdr.logdy(s1)
		assert s[12] == (np.log(s1[12])-np.log(s1[0]))*100
		s = pdr.logdya(s1,2)
		assert s[24] == (np.log(s1[24])-np.log(s1[0]))*100/2

		s = pdr.pc(s1)
		assert s[1] == (s1[1]/s1[0]-1)*100
		s = pdr.pca(s1)
		assert s[1] == ((s1[1]/s1[0])**12-1)*100
		s = pdr.pcy(s1)
		assert s[12] == (s1[12]/s1[0]-1)*100
		s = pdr.pcya(s1,2)
		assert s[24] == ((s1[24]/s1[0])**.5-1)*100

if __name__ == "__main__":
	import nose
	nose.run(argv=["-w", __file__,"--nocapture"])