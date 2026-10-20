import unittest
from unittest.mock import patch
from ap_harness.policy import PinnedRemotePolicy, CHECKPOINT

class IdentityTests(unittest.TestCase):
    def test_unverified_checkpoint_rejected(self):
        with self.assertRaises(ValueError):
            PinnedRemotePolicy(checkpoint=CHECKPOINT.replace('010000','999999'))
    def test_server_cannot_switch_pinned_checkpoint(self):
        policy=PinnedRemotePolicy(checkpoint=CHECKPOINT.replace('010000','024000'))
        with patch('vla_http_client.RemotePi05Executor.health',return_value={'checkpoint':CHECKPOINT}):
            with self.assertRaises(ValueError):policy.health()
