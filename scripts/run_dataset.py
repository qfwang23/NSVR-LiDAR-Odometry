#!/usr/bin/env python3
"""Serial, recorded-time 1x playback with small text outputs only."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import rosbag
import yaml

ROOT = Path(__file__).resolve().parents[1]
PARAMETERS = {
    'min_range': '--min-range', 'max_range': '--max-range',
    'voxel_size': '--voxel-size', 'max_points_per_voxel': '--max-points-per-voxel',
    'initial_threshold': '--initial-threshold', 'min_motion_th': '--min-motion-th',
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def git(*args):
    result = subprocess.run(['git', '-C', str(ROOT), *args],
                            capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def load_config(path):
    config = yaml.safe_load(path.read_text())
    expected = set(PARAMETERS) | {'dataset', 'topic', 'sequences', 'replay_rate'}
    if not isinstance(config, dict) or set(config) != expected:
        raise ValueError('config must contain exactly: ' + ', '.join(sorted(expected)))
    if config['dataset'] not in ('kitti', 'm2dgr') or config['replay_rate'] != 1.0:
        raise ValueError('supported datasets: kitti/m2dgr; playback must be 1.0x')
    for key in PARAMETERS:
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('invalid numeric parameter: ' + key)
    if not 0 <= config['min_range'] < config['max_range']:
        raise ValueError('invalid range mask')
    if config['voxel_size'] <= 0 or config['initial_threshold'] <= 0 or config['min_motion_th'] < 0:
        raise ValueError('invalid voxel or adaptive-scale parameter')
    if type(config['max_points_per_voxel']) is not int or not 0 < config['max_points_per_voxel'] <= 2147483647:
        raise ValueError('invalid voxel capacity')
    if not isinstance(config['sequences'], list) or not config['sequences']:
        raise ValueError('sequences must be a nonempty list')
    for sequence in config['sequences']:
        if not isinstance(sequence, str) or not sequence or not sequence.replace('_', '').isalnum():
            raise ValueError('sequence names must be quoted strings without path components')
    if len(config['sequences']) != len(set(config['sequences'])):
        raise ValueError('duplicate sequence')
    if not isinstance(config['topic'], str) or not config['topic'].startswith('/'):
        raise ValueError('invalid ROS topic')
    return config


def progress(timing):
    if not timing.exists():
        return {'processed_frames': 0}
    last = None
    with timing.open() as stream:
        for row in csv.DictReader(stream):
            if row.get('global_buckets') is not None:
                last = row
    return {'processed_frames': int(last['frame']),
            'rss_mib': float(last['rss_mib']),
            'deadline_lateness_ms': float(last['deadline_lateness_ms'])} if last else {'processed_frames': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sequence', help='one sequence from the selected YAML; otherwise all')
    parser.add_argument('--variant', choices=['full', 'no_f2f', 'no_vertical', 'neither'], default='full')
    parser.add_argument('--threads', type=int, default=0)
    parser.add_argument('--max-frames', type=int, default=0, help='1..120: smoke test; 0: formal full run')
    args = parser.parse_args()
    if args.threads < 0 or not 0 <= args.max_frames <= 120:
        parser.error('threads must be nonnegative; smoke tests permit 1..120 frames')
    config_path = args.config.resolve()
    config = load_config(config_path)
    sequences = [args.sequence] if args.sequence else config['sequences']
    if any(sequence not in config['sequences'] for sequence in sequences):
        parser.error('sequence is not in the selected dataset YAML')
    binary = ROOT / 'build' / 'lo_bag'
    if not binary.is_file():
        parser.error('build/lo_bag not found; build the clean repository first')
    commit = git('rev-parse', 'HEAD')
    dirty = git('status', '--porcelain')
    if not args.max_frames and (not commit or dirty != ''):
        parser.error('full runs require a committed, clean Git checkout')
    provenance = json.loads((ROOT / 'provenance/frozen_source.json').read_text())
    for path, expected in provenance['core_sha256'].items():
        if sha256(ROOT / path) != expected:
            parser.error('recorded algorithm source has changed: ' + path)
    data_root = args.data_root.resolve()
    jobs = []
    for sequence in sequences:
        bag_path = data_root / (sequence + '.bag')
        with rosbag.Bag(str(bag_path)) as bag:
            topic_info = bag.get_type_and_topic_info().topics.get(config['topic'])
            if topic_info is None or topic_info.msg_type != 'sensor_msgs/PointCloud2':
                raise ValueError('missing PointCloud2 topic in ' + str(bag_path))
            jobs.append({'sequence': sequence, 'bag': str(bag_path),
                         'expected_frames': topic_info.message_count,
                         'bag_bytes': bag_path.stat().st_size,
                         'bag_mtime_ns': bag_path.stat().st_mtime_ns})
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    metadata = {'status': 'running', 'role': 'smoke' if args.max_frames else 'full_sequence',
                'method_revision': provenance['method_revision'],
                'frozen_core_commit': provenance['frozen_core_commit'],
                'started_unix_s': time.time(), 'config': config, 'config_sha256': sha256(config_path),
                'git_commit': commit, 'git_dirty': dirty, 'binary_sha256': sha256(binary),
                'core_sha256': provenance['core_sha256'], 'variant': args.variant,
                'threads': args.threads, 'max_frames': args.max_frames,
                'jobs': jobs}
    save_json(output / 'run.json', metadata)
    for job in jobs:
        directory = output / job['sequence']
        directory.mkdir()
        timing = directory / 'timing.csv'
        command = [str(binary), '--bag', job['bag'], '--topic', config['topic'],
                   '--expected-frames', str(job['expected_frames']),
                   '--trajectory', str(directory / 'trajectory.tum'), '--timing', str(timing),
                   '--replay-rate', '1.0', '--variant', args.variant,
                   '--threads', str(args.threads), '--max-frames', str(args.max_frames)]
        for key, option in PARAMETERS.items():
            command += [option, str(config[key])]
        job['command'] = command
        job['status'] = 'running'
        save_json(output / 'run.json', metadata)
        print('Starting ' + job['sequence'] + ' at 1.0x', flush=True)
        with (directory / 'run.log').open('x') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            try:
                while True:
                    try:
                        returncode = process.wait(timeout=600)
                        break
                    except subprocess.TimeoutExpired:
                        job.update(progress(timing))
                        job['checked_unix_s'] = time.time()
                        save_json(output / 'run.json', metadata)
                        print(json.dumps(job, ensure_ascii=False), flush=True)
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                metadata['status'] = job['status'] = 'interrupted'
                save_json(output / 'run.json', metadata)
                raise
        job.update(progress(timing))
        job['exit_code'] = returncode
        expected = min(args.max_frames, job['expected_frames']) if args.max_frames else job['expected_frames']
        job['status'] = 'complete' if returncode == 0 and job['processed_frames'] == expected else 'failed'
        print(job['sequence'] + ': ' + job['status'], flush=True)
        if job['status'] == 'failed':
            metadata['status'] = 'failed'
            save_json(output / 'run.json', metadata)
            return 1
        save_json(output / 'run.json', metadata)
    metadata['status'] = 'complete'
    metadata['finished_unix_s'] = time.time()
    save_json(output / 'run.json', metadata)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError, rosbag.ROSBagException) as error:
        sys.exit(str(error))
