import unittest
from urllib.error import URLError, HTTPError
from ap_harness.policy import retry_model_call

class ModelWaitTests(unittest.TestCase):
    def test_more_than_two_timeouts_then_success(self):
        calls=[];sleeps=[]
        def call():
            calls.append(1)
            if len(calls)<6:raise URLError('timed out')
            return 'response'
        self.assertEqual(retry_model_call(call,sleep=sleeps.append),'response')
        self.assertEqual(len(calls),6)
        self.assertEqual(len(sleeps),5)
    def test_bad_contract_does_not_retry(self):
        def call():raise ValueError('wrong action dimensions')
        with self.assertRaises(ValueError):retry_model_call(call,sleep=lambda _:self.fail('must not retry'))
    def test_authorization_error_is_not_transport_timeout(self):
        def call():raise HTTPError('url',403,'forbidden',{},None)
        with self.assertRaises(HTTPError):retry_model_call(call,sleep=lambda _:self.fail('must not retry'))
