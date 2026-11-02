"""Geometric regressions: units, projection, occlusion, missing data and mount gating."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vision.road_geometry import (Calibration, associate_depth, depth_metres, height_obstacles,
                                  is_fresh, nearest_stamp, rotation_matrix)


def camera(width, height=1, fx=1, d=()):
    return Calibration(width, height, np.diag([fx, 1, 1]), np.array(d), 'optical')


class RoadGeometryTests(unittest.TestCase):
    def test_depth_buffer_padding_endianness_and_units(self):
        values = np.array([[1000, 0], [2000, 3000]], dtype='>u2')
        raw = values[0].tobytes() + b'xx' + values[1].tobytes() + b'xx'
        image = SimpleNamespace(encoding='16UC1', is_bigendian=True, height=2, width=2, step=6, data=raw)
        np.testing.assert_allclose(depth_metres(image), [[1, 0], [2, 3]])

    def test_identity_projection_preserves_semantic_coordinates(self):
        result = associate_depth(np.array([[1, 2, 0]], np.float32), camera(3), camera(3), np.eye(4),
                                 np.array([[0, 4, 5]], np.uint8), np.ones((1, 3), np.float32))
        np.testing.assert_array_equal(result['labels'], [0, 4])
        np.testing.assert_allclose(result['xyz'], [[0, 0, 1], [2, 0, 2]])
        self.assertTrue(np.isnan(result['aligned_depth'][0, 2]))
        np.testing.assert_array_equal(result['candidate'], [[255, 0, 0]])

    def test_z_buffer_does_not_label_an_occluded_background_point(self):
        result = associate_depth(np.array([[1, 2]], np.float32), camera(2), camera(2, fx=0.25), np.eye(4),
                                 np.zeros((1, 2), np.uint8), np.ones((1, 2), np.float32))
        np.testing.assert_array_equal(result['labels'], [0, 255])
        self.assertEqual(result['aligned_depth'][0, 0], 1)

    def test_projection_uses_color_distortion(self):
        labels = np.full((1, 30), 4, np.uint8)
        labels[0, 28] = 5
        result = associate_depth(np.array([[0, 0, 1]], np.float32), camera(3), camera(30, fx=10, d=[0.1, 0, 0, 0, 0]),
                                 np.eye(4), labels, np.ones_like(labels, np.float32))
        self.assertEqual(result['uv'][0, 0], 28)
        self.assertEqual(result['labels'][0], 5)

    def test_rational_projection_matches_opencv_reference(self):
        dc = camera(25, 15, fx=20)
        cc = camera(35, 25, fx=18, d=[0.1, -0.04, 0.002, -0.003, 0.01, 0.02, -0.01, 0.001])
        depth = np.full((15, 25), 2, np.float32)
        result = associate_depth(depth, dc, cc, np.eye(4), np.zeros((25, 35), np.uint8), np.ones((25, 35), np.float32))
        pixels, _ = cv2.projectPoints(result['xyz'].astype(np.float64), np.zeros(3), np.zeros(3), cc.k, cc.d)
        np.testing.assert_array_equal(result['uv'], np.rint(pixels.reshape(-1, 2)).astype(np.int32))

    def test_invalid_depth_and_low_confidence_never_create_free_candidate(self):
        result = associate_depth(np.array([[0, np.nan, np.inf, 1]], np.float32), camera(4), camera(4), np.eye(4),
                                 np.zeros((1, 4), np.uint8), np.full((1, 4), 0.2, np.float32))
        self.assertEqual(len(result['xyz']), 1)
        self.assertEqual(result['labels'][0], 255)
        self.assertFalse(result['candidate'].any())

    def test_internal_extrinsics_millimetres_are_rejected(self):
        matrix = np.eye(4)
        matrix[0, 3] = 25
        with self.assertRaisesRegex(ValueError, 'units'):
            associate_depth(np.ones((1, 1)), camera(1), camera(1), matrix, np.zeros((1, 1), np.uint8), np.ones((1, 1)))

    def test_mount_cannot_assume_ground_height(self):
        with self.assertRaisesRegex(ValueError, 'ground'):
            height_obstacles(np.zeros((1, 3)), np.eye(4), None)
        base, obstacles, ground = height_obstacles(np.array([[0, 0, -0.5], [0, 0, -0.3], [0, 0, -0.7]]), np.eye(4), -0.5)
        np.testing.assert_array_equal(obstacles, [False, True, False])
        np.testing.assert_array_equal(ground, [True, False, False])

    def test_stamp_pairing_and_clock_edges(self):
        self.assertEqual(nearest_stamp([(10, 'old'), (10.03, 'near')], 10.04, 0.02)[1], 'near')
        self.assertIsNone(nearest_stamp([(10, 'old')], 10.3, 0.02))
        self.assertFalse(is_fresh(10, 8, 0.75))
        self.assertFalse(is_fresh(10, 11, 0.75))
        self.assertFalse(is_fresh(0, 0, 0.75))
        self.assertTrue(is_fresh(10, 9.9, 0.75))

    def test_quaternion_rotation_and_bad_calibration(self):
        np.testing.assert_allclose(rotation_matrix([0, 0, 1, 0]), np.diag([-1, -1, 1]))
        with self.assertRaises(ValueError):
            rotation_matrix([0, 0, 0, 0])
        with self.assertRaises(ValueError):
            camera(1, fx=0).validate()


if __name__ == '__main__':
    unittest.main()
