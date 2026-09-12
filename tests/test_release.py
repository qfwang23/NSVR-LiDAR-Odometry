import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('run_dataset', ROOT / 'scripts/run_dataset.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ReleaseTests(unittest.TestCase):
    def test_frozen_core_and_adapter_hashes(self):
        manifest = json.loads((ROOT / 'provenance/frozen_source.json').read_text())
        self.assertEqual(manifest['frozen_core_commit'], 'def06cff6b0aedf1207dad3887aa705f8a29d638')
        self.assertEqual(len(manifest['core_sha256']), 13)
        for path, expected in manifest['core_sha256'].items():
            self.assertEqual(runner.sha256(ROOT / path), expected, path)
        self.assertEqual(runner.sha256(ROOT / 'app/lo_bag.cpp'),
                         manifest['adapter']['sha256'])

    def test_selected_datasets(self):
        kitti = runner.load_config(ROOT / 'config/kitti.yaml')
        m2dgr = runner.load_config(ROOT / 'config/m2dgr.yaml')
        self.assertEqual(kitti['sequences'], ['%02d' % n for n in range(11)])
        self.assertEqual(m2dgr['sequences'], ['room_01', 'hall_01', 'door_01', 'lift_01', 'street_03', 'street_04', 'street_05'])
        manifest = json.loads((ROOT / 'provenance/frozen_source.json').read_text())
        for name, config, minimum, voxel in [('kitti', kitti, 5.0, 1.0), ('m2dgr', m2dgr, 1.0, 0.6)]:
            self.assertEqual(config['min_range'], minimum)
            self.assertEqual(config['voxel_size'], voxel)
            self.assertEqual(config['max_range'], 100.0)
            self.assertEqual(config['max_points_per_voxel'], 15)
            self.assertEqual(config['initial_threshold'], 2.0)
            self.assertEqual(config['min_motion_th'], 0.1)
            self.assertEqual(config['replay_rate'], 1.0)
            self.assertEqual(config['topic'], '/velodyne_points')
            self.assertEqual(runner.sha256(ROOT / 'config' / (name + '.yaml')),
                             manifest['config_sha256']['config/' + name + '.yaml'])

    def test_reject_bad_configs(self):
        valid = runner.load_config(ROOT / 'config/kitti.yaml')
        for replacement in [{'replay_rate': 2}, {'unused': 1}, {'min_range': -1},
                            {'max_points_per_voxel': 1.5}, {'sequences': ['../00']},
                            {'voxel_size': float('nan')}, {'sequences': ['00', '00']}]:
            with self.subTest(replacement=replacement), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'config.yaml'
                path.write_text(yaml.safe_dump(dict(valid, **replacement)))
                with self.assertRaises(ValueError):
                    runner.load_config(path)

    def adapter(self, *extra):
        binary = ROOT / 'build/lo_bag'
        if not binary.exists():
            self.skipTest('build lo_bag first')
        return subprocess.run([str(binary), '--bag', 'missing.bag',
                               '--trajectory', 'unused.tum', '--timing', 'unused.csv',
                               '--expected-frames', '1', *extra], capture_output=True, text=True)

    def test_adapter_rejects_bad_numbers_and_variants(self):
        for args in [('--replay-rate', '2'), ('--threads', '-1'), ('--max-frames', '-1'),
                     ('--max-points-per-voxel', '2147483648'), ('--variant', 'typo')]:
            with self.subTest(args=args):
                self.assertEqual(self.adapter(*args).returncode, 2)

    def test_adapter_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trajectory.tum'
            path.write_text('must survive\n')
            result = self.adapter('--trajectory', str(path))
            self.assertEqual(result.returncode, 1)
            self.assertIn('must not already exist', result.stderr)
            self.assertEqual(path.read_text(), 'must survive\n')


if __name__ == '__main__':
    unittest.main()
