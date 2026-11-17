"""Financial regression checks for the migrated DCA and confirmed signal boundary."""
import unittest
import numpy as np
from . import engine as E
from .signals import CoinGate,make_gate,HOUR
from .account import acct
class DcaCheck(unittest.TestCase):
 def test_gate_exit_does_not_count_returned_principal_as_profit(self):
  x=np.full(4,100.);p=E.Params(step=.01,gx=1.,amt=1.,tp=.015,n=1,lev=1.,base=1000.)
  r=E.simulate(x,x,x,x,np.zeros(4),make_gate(np.array([1,1,0,0]),'exit'),p,10000.,detail=True)
  expected=9000.+1000.*(1.-.0005)**2 # Buy fee reduces acquired quantity; sell fee reduces proceeds.
  self.assertAlmostEqual(r['eq'][-1],expected,places=7)
  self.assertAlmostEqual(sum(z['pnl'] for z in r['rounds_detail']),expected-10000.,places=7)
 def test_future_prices_cannot_change_confirmed_signal_prefix(self):
  ts=np.arange(440,dtype='int64')*HOUR;c=100+np.sin(np.arange(440)/30)*10+np.arange(440)*.01
  a=CoinGate(ts[:400],c[:400]).state('4h',10,30,80)
  c[400:]=1e9;b=CoinGate(ts,c).state('4h',10,30,80)
  np.testing.assert_array_equal(a[0],b[0][:400]);np.testing.assert_array_equal(a[1],b[1][:400])
 def test_capacity_and_five_percent_allocation(self):
  u=np.array([1.,1.1]);e,info=acct([(0,1,u,False),(0,1,u,False)],0,2,cap=1)
  self.assertAlmostEqual(e[-1],1.005);self.assertEqual(info['used'],1);self.assertEqual(info['skipped'],1)
if __name__=='__main__':unittest.main()

