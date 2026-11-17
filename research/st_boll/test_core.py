import unittest
import numpy as np
from . import daily, indicators


class ResearchChecks(unittest.TestCase):
    def test_next_open_and_fees(self):
        # Signal at close 0 must enter at open 1=100, not at the earlier 50.
        old_fee, old_fund = daily.FEE, daily.FUND8H
        try:
            daily.FEE, daily.FUND8H = .001, 0.
            eq, _, trades = daily.run_fast(np.array([50.,100.,110.,120.]),
                np.array([51.,101.,111.,121.]), np.array([49.,99.,109.,119.]),
                np.array([50.,100.,110.,120.]), np.array([1.,1.,-1.,-1.]), 'L', 86400000)
            self.assertAlmostEqual(eq[-1], .999*1.2*.999)
            self.assertEqual(len(trades), 1)
        finally:
            daily.FEE, daily.FUND8H = old_fee, old_fund

    def test_indicator_prefix_causality(self):
        c = np.array([10.,11.,9.,12.,13.,8.,7.,14.,12.,16.])
        full = indicators.st(c+1, c-1, c, 3, 2.)
        short = indicators.st(c[:7]+1, c[:7]-1, c[:7], 3, 2.)
        for a,b in zip(full,short):
            np.testing.assert_allclose(a[:7], b, equal_nan=True)

    def test_incomplete_daily_bar_is_dropped(self):
        t = np.arange(25, dtype='int64')*3600000
        c = np.arange(25, dtype=float)+100
        result = indicators.resample(t,c,c+1,c-1,c,'1d')
        self.assertEqual(len(result[0]), 1)
        self.assertEqual(result[-1][0], 123.)


if __name__ == '__main__':
    unittest.main()
