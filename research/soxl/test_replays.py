"""Golden end-to-end checks against frozen input bytes; optional raw-market replay checks."""
from pathlib import Path
import argparse,hashlib,json,unittest
import numpy as np
if __package__:
    from . import run,kdj_core,grid_core
else:
    import run,kdj_core,grid_core

LEGACY=None

class InputChecks(unittest.TestCase):
    def test_non_5m_input_fails(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.json';p.write_text(json.dumps([[i*900000,100,101,99,100] for i in range(160)]))
            with self.assertRaisesRegex(ValueError,'continuous 5m'):run.load_candles(p)

@unittest.skipUnless(LEGACY is not None,'Pass --legacy-data-root for historical golden checks')
class FrozenReplays(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p=LEGACY/'backtest/reports/soxl_ema5m_st3x_2026-11-11/SOXLUSDT.klines_5m.json'
        cls.rows=run.load_candles(cls.p);cls.f=run.read_json(LEGACY/'backtest/sox_grid/data/SOXLUSDT.funding.json')
        inp=run.read_json(LEGACY/'backtest/reports/soxl_sar_grid_1h_2026-11-12/inputs.json');cls.tiers=[(x['cap'],x['rate'],x['cum']) for x in inp['tiers']]
    def test_kdj_terminal_position_and_fixed_costs(self):
        a=kdj_core.evaluate(self.rows,self.f);b=kdj_core.evaluate(self.rows,self.f,.0008)
        self.assertEqual(a['simsig']['trades'],116);self.assertEqual(a['closed_only']['trades'],115)
        self.assertAlmostEqual(a['simsig']['mean_net_pct'],.4049562598461735,places=10)
        self.assertAlmostEqual(a['closed_only_full_notional_return_pct'],57.24111751275667,places=8)
        self.assertAlmostEqual(a['simsig']['mean_net_pct']-b['simsig']['mean_net_pct'],.08,places=10)
        cross=kdj_core.evaluate(self.rows,self.f,j_cross_only=True)
        self.assertEqual(a['simsig'],cross['simsig'])
    def test_rebuilt_hourly_signals_and_grid(self):
        import pandas as pd
        expected=pd.read_csv(LEGACY/'backtest/reports/soxl_sar_grid_1h_2026-11-12/signals_1h.csv').dropna(subset=['lower','upper']).loc[lambda x:x.sar_direction!=0].reset_index(drop=True)
        actual=grid_core.derive_signals(self.rows)
        for col in ('known_time','st_direction','sar_direction'):np.testing.assert_array_equal(actual[col],expected[col])
        for col in ('lower','upper'):np.testing.assert_allclose(actual[col],expected[col],rtol=0,atol=1e-10)
        old=grid_core.evaluate(self.rows,self.f,self.tiers,stop=None);new=grid_core.evaluate(self.rows,self.f,self.tiers)
        self.assertAlmostEqual(old['return_pct'],111.424742726539,places=8)
        self.assertAlmostEqual(new['return_pct'],110.02177352445197,places=8)
        self.assertAlmostEqual(new['dd_pct'],25.89253213124223,places=8)
        self.assertEqual(new['cycles'],89)
    def test_indicators_do_not_look_forward(self):
        full=grid_core.derive_signals(self.rows);prefix=grid_core.derive_signals(self.rows[:10000]);aligned=full[full.known_time<=prefix.known_time.iloc[-1]].reset_index(drop=True)
        for col in ('known_time','lower','upper','st_direction','sar_direction'):np.testing.assert_allclose(aligned[col],prefix[col],rtol=0,atol=1e-10)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--legacy-data-root',type=Path);args,rest=parser.parse_known_args()
    if args.legacy_data_root:LEGACY=run.legacy_repo(args.legacy_data_root)
    # unittest decorators are evaluated at import time; update class skip after parsing.
    FrozenReplays.__unittest_skip__=LEGACY is None
    unittest.main(argv=['test_replays.py']+rest)
