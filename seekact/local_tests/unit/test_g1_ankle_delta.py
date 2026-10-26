import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from services.g1_sonic.ankle_delta import amplify_windows, LEGS


@pytest.mark.parametrize('side', ['left', 'right'])
def test_identity_anchor_scope_and_rate(side):
    poses = np.zeros((80, 10, 21, 3))
    poses[:, :, :, 2] = .03
    ids = np.array(LEGS[side]) - 1
    poses[10:, :, ids, 0] = .5
    original = poses.copy()
    identity, zero = amplify_windows(poses, side, 1.)
    np.testing.assert_array_equal(identity, poses)
    assert not np.any(zero)
    changed, delta = amplify_windows(poses, side, 2.)
    np.testing.assert_array_equal(changed[0], poses[0])
    unowned = sorted(set(range(21)) - set(ids))
    np.testing.assert_array_equal(changed[:, :, unowned], poses[:, :, unowned])
    np.testing.assert_array_equal(poses, original)
    assert 0.1 < np.linalg.norm(delta, axis=-1).max() <= .35 + 1e-12
    angular_steps = (R.from_rotvec(delta[:-1].reshape(-1, 3)).inv() *
                     R.from_rotvec(delta[1:].reshape(-1, 3))).magnitude()
    assert angular_steps.max() <= .6 * .02 + 1e-12


@pytest.mark.parametrize('gain', [float('nan'), float('inf'), .9, 2.1])
def test_reject_invalid_gain(gain):
    with pytest.raises(ValueError):
        amplify_windows(np.zeros((2, 10, 21, 3)), 'left', gain)
