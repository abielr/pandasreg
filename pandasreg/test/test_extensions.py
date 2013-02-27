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

	def test_overlay(self):
		ix1 = RPeriodIndex(start=datetime(1970,1,1), periods=6, freq="M")
		ix2 = RPeriodIndex(start=datetime(1970,4,1), periods=6, freq="M")
		ix3 = RPeriodIndex(start=datetime(1971,4,1), periods=6, freq="M")
		s1 = pd.Series(np.arange(6), ix1)
		s2 = pd.Series(np.arange(6), ix2)
		s3 = pd.Series(np.arange(6), ix3)

		pdr.overlay([s1, s2])
		pdr.overlay([s1, s2], False)
		pdr.overlay([s1,s2,s3])
		pdr.overlay([s1,s3])

	def test_extend(self):
		ix1 = RPeriodIndex(start=datetime(1970,1,1), periods=6, freq="M")
		ix2 = RPeriodIndex(start=datetime(1962,4,1), periods=6, freq="M")
		ix3 = RPeriodIndex(start=datetime(1969,10,1), periods=18, freq="M")
		s1 = pd.Series(np.array([100,101,102,99,100,102], dtype=np.float), ix1)
		s2 = pd.Series(np.array([100,101,102,99,100,102], dtype=np.float), ix2)
		s3 = pd.Series(np.array([100,101,102,99,100,102,100,101,102,99,100,
			102,100,101,102,99,100,102], dtype=np.float), ix3)

		assert all(pdr.extend(s1, s2, extender_type="diff")==s1)

if __name__ == "__main__":
	import nose
	nose.run(argv=["-w",__file__])