import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('suite_runner',Path(__file__).resolve().parents[1]/'scripts/run_ood99.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)

class RetryTests(unittest.TestCase):
    def test_zero_ack_but_unknown_physical_advance_cannot_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);(out/'run').mkdir()
            row={'task':'t','status':'infrastructure_failed','steps':0,'run':'run'}
            data={'acknowledged_steps':0,'control_summary':{'steps':0},'rounds':[{'ack_steps':0,'unacknowledged_submissions':1}]}
            (out/'run/acceptance.json').write_text(json.dumps(data))
            self.assertFalse(runner.retry_has_no_submitted_actions(row,out))
            data['rounds']=[];(out/'run/acceptance.json').write_text(json.dumps(data))
            self.assertTrue(runner.retry_has_no_submitted_actions(row,out))

    def test_retry_keeps_prior_evidence_without_double_counting_task(self):
        rows=[{'task':'t','status':'infrastructure_failed','steps':0}]
        runner.record(rows,{'task':'t','status':'success','steps':32})
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['previous_attempts'][0]['status'],'infrastructure_failed')

if __name__=='__main__':unittest.main()
