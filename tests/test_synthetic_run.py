"""End-to-end adapter/runner checks using a tiny temporary synthetic ROS bag."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import genpy
import rosbag
from sensor_msgs.point_cloud2 import create_cloud_xyz32
from std_msgs.msg import Header

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from evaluate_run import validate_and_time


class SyntheticRunTests(unittest.TestCase):
    def test_all_variants_on_temporary_pointcloud_bag(self):
        if not (ROOT / 'build/lo_bag').is_file():
            self.skipTest('build lo_bag first')
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            points = [(x * 0.3, y * 0.3, -1.5)
                      for x in range(-10, 11) for y in range(-10, 11)]
            with rosbag.Bag(str(directory / 'room_01.bag'), 'w') as bag:
                for frame in range(4):
                    stamp = genpy.Time.from_sec(1000.0 + frame * 0.01)
                    cloud = create_cloud_xyz32(Header(stamp=stamp, frame_id='lidar'), points)
                    bag.write('/velodyne_points', cloud, t=stamp)
            for variant in ('full', 'no_f2f', 'no_vertical', 'neither'):
                with self.subTest(variant=variant):
                    output = directory / variant
                    command = [sys.executable, '-B', str(ROOT / 'scripts/run_dataset.py'),
                               '--config', str(ROOT / 'config/m2dgr.yaml'),
                               '--data-root', str(directory), '--sequence', 'room_01',
                               '--output', str(output), '--variant', variant,
                               '--threads', '2', '--max-frames', '4']
                    process = subprocess.run(command, capture_output=True, text=True, timeout=15)
                    self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
                    metadata = json.loads((output / 'run.json').read_text())
                    self.assertEqual(metadata['status'], 'complete')
                    self.assertEqual(metadata['role'], 'smoke')
                    self.assertEqual(metadata['jobs'][0]['processed_frames'], 4)
                    self.assertEqual(metadata['config']['voxel_size'], 0.6)
                    sequence = output / 'room_01'
                    metric = validate_and_time(sequence / 'trajectory.tum',
                                               sequence / 'timing.csv', 4, variant)
                    self.assertEqual(metric['skip_first_2']['timed_frames'], 2)
                    self.assertGreater(metric['skip_first_2']['algorithm_ms']['mean'], 0)
                    self.assertEqual({path.name for path in sequence.iterdir()},
                                     {'trajectory.tum', 'timing.csv', 'run.log'})
                    if variant == 'full':
                        ground_truth = directory / 'synthetic_gt.txt'
                        ground_truth.write_text(''.join(
                            '%.9f 0 0 0 0 0 0 1\n' % (1000.0 + frame * 0.01)
                            for frame in range(4)))
                        evaluation = [sys.executable, '-B', str(ROOT / 'scripts/evaluate_run.py'),
                                      '--dataset', 'm2dgr', '--sequence', 'room_01',
                                      '--variant', variant, '--run', str(sequence),
                                      '--ground-truth', str(ground_truth), '--expected-frames', '4']
                        evaluated = subprocess.run(evaluation, capture_output=True, text=True, timeout=15)
                        self.assertEqual(evaluated.returncode, 0, evaluated.stdout + evaluated.stderr)
                        report = json.loads((sequence / 'metrics.json').read_text())
                        self.assertEqual(report['role'], 'smoke')
                        self.assertEqual(report['accuracy']['associated_poses'], 4)
                        self.assertTrue((sequence / 'position_errors.csv').is_file())


if __name__ == '__main__':
    unittest.main()
