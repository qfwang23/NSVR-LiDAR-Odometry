import csv
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from evaluate_run import evaluate_positions, validate_and_time
from position_metrics import (align_positions_se3, robust_ground_truth_gap_model,
                              interpolate_ground_truth_positions)


class MetricTests(unittest.TestCase):
    def test_rigid_alignment_no_scale(self):
        p = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]], float)
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        g = p @ rotation.T + [20, 30, 40]
        aligned, r, _ = align_positions_se3(p, g)
        np.testing.assert_allclose(aligned, g, atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(r), 1.0)
        scaled, _, _ = align_positions_se3(p * 2, g)
        self.assertGreater(np.linalg.norm(scaled - g), 0.1)

    def test_gaps_not_bridged(self):
        t = np.array([0, 1, 2, 3, 30, 31, 32, 33], float)
        gaps = robust_ground_truth_gap_model(t)
        self.assertEqual(int(gaps['gap_mask'].sum()), 1)
        p = np.column_stack([t, t, t])
        result = interpolate_ground_truth_positions(t, p, [-1, 1.5, 10, 31, 40], gaps['gap_mask'])
        np.testing.assert_array_equal(result['estimate_indices'], [1, 3])
        self.assertEqual(result['outside_gt_range_count'], 2)
        self.assertEqual(result['across_gt_gap_count'], 1)

    def test_kitti_strict_frame_pairing(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            rows = np.array([[0, 0, 0, 0, 0, 0, 0, 1], [1, 1, 0, 0, 0, 0, 0, 1],
                             [2, 0, 1, 0, 0, 0, 0, 1]], float)
            np.savetxt(d / 'gt', rows)
            np.savetxt(d / 'est', rows)
            metric, _, _ = evaluate_positions('kitti', d / 'gt', d / 'est')
            self.assertLess(metric['ape_position_m']['rmse'], 1e-12)
            rows[:, 0] += 1
            np.savetxt(d / 'est', rows)
            with self.assertRaises(ValueError):
                evaluate_positions('kitti', d / 'gt', d / 'est')
            np.savetxt(d / 'est', rows[:2])
            with self.assertRaises(ValueError):
                evaluate_positions('kitti', d / 'gt', d / 'est')

    def test_m2dgr_partial_gt_and_ecef_conditioning(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            gt_time = np.array([0, 1, 2, 3, 30, 31, 32, 33], float)
            estimate_time = np.array([0, 1, 2, 3, 10, 30, 31, 32, 33], float)
            # Synthetic GT deliberately has unusable orientation columns; only
            # positions enter APE. A long gap must not manufacture support.
            gt = np.zeros((len(gt_time), 8))
            gt[:, 0] = gt_time
            gt[:, 1:4] = np.column_stack([gt_time, 2 * gt_time, gt_time * 0])
            gt[:, 1:4] += [3000000, 4000000, 5000000]
            estimate = np.zeros((len(estimate_time), 8))
            estimate[:, 0] = estimate_time
            estimate[:, 1:4] = np.column_stack([estimate_time, 2 * estimate_time, estimate_time * 0])
            estimate[:, 7] = 1
            np.savetxt(d / 'gt', gt)
            np.savetxt(d / 'est', estimate)
            metric, _, _ = evaluate_positions('m2dgr', d / 'gt', d / 'est')
            self.assertEqual(metric['associated_poses'], 8)
            self.assertEqual(metric['across_gt_gap_count'], 1)
            self.assertAlmostEqual(metric['gt_coverage_fraction'], 8 / 9)
            self.assertEqual(metric['coverage_status'], 'partial_diagnostic_only')
            self.assertEqual(metric['gt_conditioning'], 'subtract_first_gt_position_for_numerical_conditioning')
            self.assertLess(metric['ape_position_m']['rmse'], 1e-10)

    def test_timing_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            poses = np.zeros((4, 8))
            poses[:, 0] = np.arange(4)
            poses[:, 7] = 1
            np.savetxt(d / 'trajectory.tum', poses)
            rows = []
            for i in range(4):
                rows.append(dict(frame=i+1, timestamp=i, f2f_enabled=1, vertical_enabled=1,
                    f2f_executed=int(i >= 2), global_voxels=0, wall_elapsed_s=i+.01,
                    bag_elapsed_s=i, registration_ms=5, f2f_ms=2 if i >= 2 else 0,
                    f2m_ms=3, pipeline_ms=6, algorithm_ms=6.5, processing_ms=7, peak_rss_mib=40,
                    deadline_lateness_ms=0))
            def write():
                with (d / 'timing.csv').open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            write()
            metric = validate_and_time(d / 'trajectory.tum', d / 'timing.csv', 4, 'full')
            self.assertEqual(metric['skip_first_2']['timed_frames'], 2)
            self.assertEqual(metric['skip_first_2']['registration_ms']['throughput_hz'], 200)
            self.assertAlmostEqual(metric['skip_first_2']['algorithm_ms']['throughput_hz'], 1000 / 6.5)
            self.assertEqual(metric['skip_first_2']['algorithm_ms']['sum'], 13)
            rows[-1]['algorithm_ms'] = 5
            write()
            with self.assertRaisesRegex(ValueError, 'nested'):
                validate_and_time(d / 'trajectory.tum', d / 'timing.csv', 4, 'full')
            rows[-1]['algorithm_ms'] = 8
            write()
            with self.assertRaisesRegex(ValueError, 'nested'):
                validate_and_time(d / 'trajectory.tum', d / 'timing.csv', 4, 'full')
            rows[-1]['algorithm_ms'] = 6.5
            for row in rows:
                row.update(vertical_surface_ms=1, vs_source_patches=6, vs_reference_patches=6,
                           vs_matched_patches=6, vs_fit_rejected=0, vs_rank_rejected=0,
                           vs_validation_rejected=0, vs_applied=1, vs_before=5, vs_after=4,
                           vs_delta_z=.01, vs_rotation_rad=.001)
            write()
            metric = validate_and_time(d / 'trajectory.tum', d / 'timing.csv', 4, 'full')
            self.assertEqual(metric['vertical_surface_diagnostics']['applied'], 4)
            for row in rows:
                row.update(vs_anchor_age_frames=2,vs_anchor_replaced=0,vs_anchor_footprint_motion_m=.5)
            write()
            metric=validate_and_time(d / 'trajectory.tum',d / 'timing.csv',4,'full')
            self.assertEqual(metric['vertical_surface_diagnostics']['anchor_max_age_frames'],2)
            rows[-1]['vs_anchor_age_frames']=-1
            write()
            with self.assertRaisesRegex(ValueError,'anchor'):
                validate_and_time(d / 'trajectory.tum',d / 'timing.csv',4,'full')
            rows[-1]['vs_anchor_age_frames']=2
            rows[-1]['vs_after'] = 5
            write()
            with self.assertRaisesRegex(ValueError, 'did not improve'):
                validate_and_time(d / 'trajectory.tum', d / 'timing.csv', 4, 'full')
            rows[-1]['vs_after'] = 4
            for row in rows:
                row['vertical_enabled'] = 0
            write()
            with self.assertRaisesRegex(ValueError, 'disabled control'):
                validate_and_time(d / 'trajectory.tum', d / 'timing.csv', 4, 'no_vertical')
            for row in rows:
                row['vertical_enabled'] = 1
            rows[-1]['f2f_enabled'] = 0
            write()
            with self.assertRaises(ValueError):
                validate_and_time(d / 'trajectory.tum', d / 'timing.csv', 4, 'full')
            rows[-1]['f2f_enabled'] = 1
            rows[-1]['global_voxels'] = 1
            write()
            with self.assertRaises(ValueError):
                validate_and_time(d / 'trajectory.tum', d / 'timing.csv', 4, 'full')


if __name__ == '__main__':
    unittest.main()
