"""Regression: a funded gate must keep cycling after TP cancels reserve margin."""
import math
import unittest

import numpy as np
try:
    from . import native_core as N
except ImportError:
    N = None

START = 1780272000000


def example():
    return np.array([[START,100,100,72.4,72.4,100000],
        [START+N.HOUR,72.4,72.4,72.4,72.4,100000]] +
        [[START+i*N.HOUR,72.4+(i-2)*10,82.4+(i-2)*10,
          72.4+(i-2)*10,82.4+(i-2)*10,100000] for i in range(2,6)], dtype=float)


@unittest.skipIf(N is None, "NautilusTrader environment required")
class RestartMarginTest(unittest.TestCase):
    def run_example(self, close_hour):
        windows=[dict(open_ms=START,close_ms=START+N.HOUR),
                 dict(open_ms=START+2*N.HOUR,close_ms=START+close_hour*N.HOUR)]
        return N.run_native('MOCKUSDT',example(),np.empty((0,2)),np.empty((0,2)),
            resize=False,windows=windows,path='OLH',step=.005,tp=.01,amount=1.1,levels=20)

    def test_loss_then_monotone_rise_keeps_reopening(self):
        result,trace=self.run_example(6)
        trades=[f for f in trace['fills'] if f['time_ns']>=(START+2*N.HOUR)*1_000_000]
        expected=math.floor(math.log(112.4/72.4)/math.log(1.01))
        self.assertEqual(sum(f['kind']=='TP' for f in trades),expected)
        self.assertGreater(result['open_quantity'],0)
        # Each new base uses the observed last TP price, with no stale-price fill.
        for a,b in zip(trades,trades[1:]):
            if a['kind']=='TP' and b['kind']=='BASE':
                self.assertAlmostEqual(a['price'],b['price'],places=9)

    def test_retry_does_not_reopen_after_gate_close(self):
        result,trace=self.run_example(3)
        self.assertEqual(result['open_quantity'],0)
        self.assertFalse(any(f['side']=='BUY' and f['time_ns']>=(START+3*N.HOUR)*1_000_000
                             for f in trace['fills']))


if __name__=='__main__':
    unittest.main()
