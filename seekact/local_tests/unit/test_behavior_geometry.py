import unittest

import numpy as np

from services.behavior_ap.geometry import project_pixel


class BehaviorGeometryTests(unittest.TestCase):
    def setUp(self):
        self.K = np.array([[2., 0., 1.], [0., 2., 1.], [0., 0., 1.]])
        self.depth = np.full((3, 3), 2., dtype=np.float32)
        self.poses = dict(camera_position=[0., 0., 0.], camera_quat=[0., 0., 0., 1.],
                          robot_base_position=[0., 0., 0.], robot_base_quat=[0., 0., 0., 1.])

    def test_camera_optical_uses_axial_not_radial_depth(self):
        result = project_pixel(self.depth, self.K, [2, 0], output_frame='camera_optical', radius=0)
        np.testing.assert_allclose(result['point'], [1., -1., 2.])
        assert result['sampled_depth_m'] == 2
        assert result['valid_depth_samples'] == 1
        assert np.linalg.norm(result['point']) > result['sampled_depth_m']
        assert result['frame'] == 'camera_optical'

    def test_identity_pose_flips_optical_y_and_z_for_usd(self):
        result = project_pixel(self.depth, self.K, [2, 0], radius=0, **self.poses)
        np.testing.assert_allclose(result['point'], [1., 1., -2.])
        assert result['frame'] == 'base'

    def test_nonidentity_world_camera_and_base_poses(self):
        sine = np.sqrt(.5)
        result = project_pixel(
            self.depth, self.K, [2, 0], radius=0,
            camera_position=[10., 1., 3.], camera_quat=[0., 0., sine, sine],
            robot_base_position=[8., 1., 0.], robot_base_quat=[0., 0., sine, sine])
        # optical [1,-1,2] -> USD [1,1,-2] -> world [9,2,1]
        # base origin [8,1,0], yaw +90deg: local [1,-1,1].
        np.testing.assert_allclose(result['point'], [1., -1., 1.], atol=1e-12)

    def test_scaled_xyzw_and_nonidentity_y_rotation(self):
        sine = np.sqrt(.5)
        poses = dict(self.poses, camera_quat=[0., 7*sine, 0., 7*sine])
        result = project_pixel(self.depth, self.K, [1, 1], radius=0, **poses)
        np.testing.assert_allclose(result['point'], [-2., 0., 0.], atol=1e-12)

    def test_median_rejects_nan_inf_zero_and_negative_depth(self):
        depth = np.array([[np.nan, 2., np.inf], [-1., 0., 4.], [6., 8., 10.]])
        result = project_pixel(depth, self.K, [1, 1], output_frame='camera', radius=1)
        assert result['sampled_depth_m'] == 6
        assert result['valid_depth_samples'] == 5
        np.testing.assert_allclose(result['point'], [0., 0., 6.])
        # Sampling is read-only.
        assert depth[1, 1] == 0 and np.isnan(depth[0, 0])

    def test_border_patch_clips_and_fractional_ray_is_preserved(self):
        depth = np.array([[1., 3., 90.], [5., 7., 90.], [90., 90., 90.]])
        result = project_pixel(depth, self.K, [.25, .25], output_frame='camera_optical', radius=1)
        assert result['sample_window_xyxy_exclusive'] == [0, 0, 2, 2]
        assert result['sample_center_xy'] == [0, 0]
        assert result['sampled_depth_m'] == 4
        assert result['valid_depth_samples'] == 4
        np.testing.assert_allclose(result['point'], [-1.5, -1.5, 4.])

    def test_depth_range_filter_is_explicit(self):
        depth = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
        result = project_pixel(depth, self.K, [1, 1], output_frame='camera', radius=1,
                               min_depth=3, max_depth=6)
        assert result['sampled_depth_m'] == 5
        assert result['valid_depth_samples'] == 3

    def test_pinhole_skew_is_supported(self):
        K = self.K.copy()
        K[0, 1] = .5
        result = project_pixel(self.depth, K, [2, 0], output_frame='camera', radius=0)
        np.testing.assert_allclose(result['point'], [1.25, -1., 2.])

    def test_invalid_or_out_of_bounds_pixels_fail(self):
        for pixel in ([-1, 0], [3, 0], [0, 3], [0, 2.1], [np.nan, 0], [np.inf, 1], [1]):
            with self.subTest(pixel=pixel), self.assertRaises(ValueError):
                project_pixel(self.depth, self.K, pixel, output_frame='camera')

    def test_bad_depth_and_empty_neighborhood_fail(self):
        for depth in (np.zeros((3, 3)), np.full((3, 3), np.inf),
                      np.full((3, 3), np.nan), np.full((3, 3), -2.),
                      np.ones((3, 3, 1)), np.ones((0, 3)), np.array([['bad']]),
                      np.ones((3, 3), dtype=bool)):
            with self.subTest(shape=depth.shape), self.assertRaises(ValueError):
                project_pixel(depth, self.K, [0, 0], output_frame='camera')

    def test_bad_intrinsics_poses_options_fail(self):
        bad_intrinsics = (np.eye(2), np.zeros((3, 3)), np.full((3, 3), np.nan),
                          np.array([[1., 1., 0.], [1., 1., 0.], [0., 0., 1.]]))
        for K in bad_intrinsics:
            with self.subTest(K=K), self.assertRaises(ValueError):
                project_pixel(self.depth, K, [1, 1], output_frame='camera')
        for option in ({'radius': -1}, {'radius': True}, {'radius': 1.5},
                       {'output_frame': 'world'}, {'min_depth': -1}, {'max_depth': np.inf},
                       {'min_depth': 2, 'max_depth': 1}, {'max_depth': True}):
            with self.subTest(option=option), self.assertRaises(ValueError):
                project_pixel(self.depth, self.K, [1, 1], **self.poses, **option)
        for option in ({'camera_quat': [0., 0., 0., 0.]}, {'camera_quat': [0., 0., 1.]},
                       {'robot_base_quat': [0., np.nan, 0., 1.]},
                       {'camera_position': [np.inf, 0., 0.]}, {'robot_base_position': None}):
            with self.subTest(option=option), self.assertRaises(ValueError):
                project_pixel(self.depth, self.K, [1, 1], **dict(self.poses, **option))


if __name__ == '__main__':
    unittest.main()
