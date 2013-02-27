from datetime import datetime
import numpy as np
import pandas as pd
from pandasreg.rperiod import RFrequency, RPeriod, RPeriodIndex

class TestClass:
	def setUp(self):
		pass

	def tearDown(self):
		pass

	def test_basic(self):
		index = RPeriodIndex(start=datetime(1970,1,1), periods=50, freq="M")
		s = pd.Series(np.arange(len(index)), index)

if __name__ == "__main__":
	import nose
	nose.run(argv=["-w", __file__])