import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from profile_body_infer import warm_summary

class TimingTests(unittest.TestCase):
    def test_five_warmups_twenty_measurements_and_nearest_rank_p95(self):
        result=warm_summary([100]*5+list(range(1,21)),5)
        self.assertEqual(result['count'],20)
        self.assertEqual(result['median_seconds'],10.5)
        self.assertEqual(result['p95_seconds'],19)
        self.assertEqual(result['maximum_seconds'],20)
    def test_invalid_timings(self):
        for values,warmup in [([],1),([1],1),([1,2],0),([1,float('nan')],1),([1,-1],1)]:
            with self.assertRaises(ValueError):warm_summary(values,warmup)
