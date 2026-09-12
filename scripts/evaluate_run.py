#!/usr/bin/env python3
"""Validate complete trajectories and report position-only APE and timing."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from position_metrics import (load_tum_positions, robust_ground_truth_gap_model,
                              interpolate_ground_truth_positions, align_positions_se3,
                              position_error_stats)
from run_dataset import save_json, sha256


def evaluate_positions(dataset, gt_path, trajectory):
    et, ep = load_tum_positions(trajectory)
    gt, gp = load_tum_positions(gt_path)
    if len(et) < 3 or len(gt) < 3:
        raise ValueError('at least three poses required')
    conditioning = 'none'
    if dataset == 'kitti':
        if len(et) != len(gt):
            raise ValueError('KITTI requires one estimate for every GT row')
        difference = float(np.max(np.abs(et - gt)))
        if difference > 1e-4:
            raise ValueError('KITTI timestamps disagree with GT (no implicit clock correction)')
        indices = np.arange(len(et))
        associated_gt = gp
        association_info = {'pairing': 'row_index_verified_timestamps',
                            'max_timestamp_difference_s': difference}
    elif dataset == 'm2dgr':
        # Only a translation for ECEF numerical conditioning; never rotate scans.
        if np.max(np.abs(gp)) > 1e5:
            gp = gp - gp[0]
            conditioning = 'subtract_first_gt_position_for_numerical_conditioning'
        gap = robust_ground_truth_gap_model(gt)
        association = interpolate_ground_truth_positions(gt, gp, et, gap['gap_mask'])
        indices = association['estimate_indices']
        associated_gt = association['positions']
        association_info = {
            'pairing': 'linear_gt_position_interpolation_excluding_gt_gaps',
            'gt_gap_count': int(np.count_nonzero(gap['gap_mask'])),
            'gt_gap_threshold_s': gap['threshold_s'],
            'outside_gt_range_count': association['outside_gt_range_count'],
            'across_gt_gap_count': association['across_gt_gap_count'],
        }
    else:
        raise ValueError('unknown dataset')
    aligned, rotation, translation = align_positions_se3(ep[indices], associated_gt)
    residual = aligned - associated_gt
    errors = np.linalg.norm(residual, axis=1)
    coverage = len(indices) / len(et)
    singular_values = np.linalg.svd(ep[indices] - ep[indices].mean(axis=0), compute_uv=False)
    result = {
        'trajectory_frames': len(et), 'associated_poses': len(indices),
        'gt_coverage_fraction': coverage,
        'coverage_status': 'at_least_90_percent' if coverage >= .9 else 'partial_diagnostic_only',
        'ape_position_m': position_error_stats(errors),
        'alignment': 'SE3_Kabsch_positions_only_no_scale', 'alignment_scale': 1.0,
        'alignment_rotation': rotation.tolist(), 'alignment_translation': translation.tolist(),
        'estimate_position_singular_values': singular_values.tolist(),
        'gt_conditioning': conditioning, 'ground_truth_sha256': sha256(gt_path),
        'trajectory_sha256': sha256(trajectory), **association_info,
    }
    if dataset == 'kitti':
        result['aligned_gt_z_rmse_m'] = float(np.sqrt(np.mean(residual[:, 2] ** 2)))
        result['aligned_gt_xy_rmse_m'] = float(np.sqrt(np.mean(np.sum(residual[:, :2] ** 2, axis=1))))
    return result, et[indices], errors


def validate_and_time(trajectory, timing, expected_frames, variant):
    if expected_frames < 3:
        raise ValueError('at least three frames required for fixed warmup exclusion')
    if variant not in ('full', 'no_f2f', 'no_vertical', 'neither'):
        raise ValueError('unknown variant')
    poses = np.loadtxt(trajectory, ndmin=2)
    if poses.shape != (expected_frames, 8) or not np.all(np.isfinite(poses)):
        raise ValueError('trajectory must contain exactly N finite TUM poses')
    if not np.allclose(np.linalg.norm(poses[:, 4:8], axis=1), 1.0, atol=1e-6, rtol=0):
        raise ValueError('invalid estimate quaternion')
    if np.any(np.diff(poses[:, 0]) <= 0):
        raise ValueError('trajectory timestamps must be strictly increasing')
    with timing.open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != expected_frames or [int(r['frame']) for r in rows] != list(range(1, expected_frames + 1)):
        raise ValueError('timing frame counts/indices do not match expected input')
    columns = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    if any(not np.all(np.isfinite(v)) for v in columns.values()):
        raise ValueError('nonfinite timing data')
    if np.max(np.abs(columns['timestamp'] - poses[:, 0])) > 1e-8:
        raise ValueError('timing and trajectory timestamps disagree')
    f2f = int(variant in ('full', 'no_vertical'))
    vertical = int(variant in ('full', 'no_f2f'))
    if not np.all(columns['f2f_enabled'] == f2f) or not np.all(columns['vertical_enabled'] == vertical):
        raise ValueError('recorded ablation flags do not match requested variant')
    executed = (np.arange(expected_frames) >= 2).astype(int) * f2f
    if not np.array_equal(columns['f2f_executed'], executed):
        raise ValueError('unexpected F2F execution schedule')
    if np.any(columns['global_voxels'] != 0):
        raise ValueError('unexpected global visualization-map accumulation')
    if np.any(columns['wall_elapsed_s'] + 1e-6 < columns['bag_elapsed_s']):
        raise ValueError('processing advanced ahead of recorded-time 1x schedule')
    scopes = ['algorithm_ms', 'registration_ms', 'f2f_ms', 'f2m_ms', 'pipeline_ms', 'processing_ms']
    if 'vertical_surface_ms' in columns:
        scopes.append('vertical_surface_ms')
    if any(np.any(columns[k] < 0) for k in scopes):
        raise ValueError('negative timings')
    if (np.any(columns['pipeline_ms'] > columns['algorithm_ms'] + 1e-6)
            or np.any(columns['algorithm_ms'] > columns['processing_ms'] + 1e-6)):
        raise ValueError('nested pipeline/algorithm/processing timings disagree')
    result = {}
    for warmup in (0, 2):
        if expected_frames <= warmup:
            raise ValueError('not enough frames for fixed warmup exclusion')
        record = {'warmup_frames': warmup, 'timed_frames': expected_frames - warmup}
        for scope in scopes:
            values = columns[scope][warmup:]
            mean = float(values.mean())
            record[scope] = {'mean': mean, 'median': float(np.median(values)),
                             'p95': float(np.percentile(values, 95)), 'sum': float(values.sum()),
                             'throughput_hz': 1000.0 / mean if mean > 0 else None}
        result['all_frames' if warmup == 0 else 'skip_first_2'] = record
    result.update({'peak_rss_mib': float(columns['peak_rss_mib'].max()),
                   'final_lateness_ms': float(columns['deadline_lateness_ms'][-1]),
                   'max_lateness_ms': float(columns['deadline_lateness_ms'].max()),
                   'wall_elapsed_s': float(columns['wall_elapsed_s'][-1]),
                   'bag_elapsed_s': float(columns['bag_elapsed_s'][-1])})
    diagnostics = {}
    for stage in ('ff', 'lm'):
        if stage + '_vertical_gates' not in columns:
            continue
        fields = ['iterations', 'vertical_gates', 'zero_steps', 'clipped_steps', 'full_fallbacks', 'stage_clipped']
        fields += [key for key in ['correspondences', 'geometry_matches'] if stage + '_' + key in columns]
        d = {key: int(columns[stage + '_' + key].sum()) for key in fields}
        if d['vertical_gates'] != d['zero_steps'] + d['clipped_steps'] + d['full_fallbacks']:
            raise ValueError('vertical diagnostic counts disagree')
        if not vertical and (d['vertical_gates'] or d['stage_clipped']):
            raise ValueError('vertical safeguard executed in disabled control')
        diagnostics[stage] = d
        if not vertical and d.get('geometry_matches', 0):
            raise ValueError('vertical geometry executed in disabled control')
    result['vertical_diagnostics'] = diagnostics
    if 'vs_applied' in columns:
        fields=['source_patches','reference_patches','matched_patches','fit_rejected',
                'rank_rejected','validation_rejected','applied']
        surface={key:int(columns['vs_'+key].sum()) for key in fields}
        surface['max_abs_delta_z_m']=float(np.max(np.abs(columns['vs_delta_z'])))
        surface['max_rotation_rad']=float(np.max(columns['vs_rotation_rad']))
        if 'vs_range_bias' in columns:
            surface['max_abs_range_bias']=float(np.max(np.abs(columns['vs_range_bias'])))
        if 'vs_anchor_age_frames' in columns:
            ages=columns['vs_anchor_age_frames']
            replaced=columns['vs_anchor_replaced']
            motion=columns['vs_anchor_footprint_motion_m']
            if (np.any(ages<0) or np.any(ages!=np.floor(ages)) or np.any(motion<0)
                    or not np.all(np.isin(replaced,[0,1]))):
                raise ValueError('invalid temporal anchor diagnostics')
            surface.update(anchor_replacements=int(replaced.sum()),
                           anchor_max_age_frames=int(ages.max()),
                           anchor_mean_age_frames=float(ages.mean()),
                           anchor_max_footprint_motion_m=float(motion.max()))
        if not vertical and any(surface.values()):
            raise ValueError('surface refinement executed in disabled control')
        if not vertical and np.any(columns['vertical_surface_ms'] != 0):
            raise ValueError('surface refinement timed in disabled control')
        if not np.all(np.isin(columns['vs_applied'], [0, 1])):
            raise ValueError('invalid surface acceptance flag')
        applied=columns['vs_applied']>0
        if np.any(columns['vs_after'][applied]>=columns['vs_before'][applied]):
            raise ValueError('accepted surface correction did not improve validation objective')
        result['vertical_surface_diagnostics']=surface
    return result


def evaluate_job(dataset, sequence, variant, directory, gt_path, expected_frames):
    trajectory = directory / 'trajectory.tum'
    timing = validate_and_time(trajectory, directory / 'timing.csv', expected_frames, variant)
    accuracy, timestamps, errors = evaluate_positions(dataset, gt_path, trajectory)
    metadata_path = directory.parent / 'run.json'
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    result = {'dataset': dataset, 'sequence': sequence, 'variant': variant,
              'status': 'complete', 'method_revision': metadata.get('method_revision', 'unrecorded'),
              'git_commit': metadata.get('git_commit'), 'binary_sha256': metadata.get('binary_sha256'),
              'role': metadata.get('role', 'unrecorded'),
              'frozen_core_commit': metadata.get('frozen_core_commit'),
              'accuracy': accuracy, 'timing': timing}
    save_json(directory / 'metrics.json', result)
    with (directory / 'position_errors.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['timestamp', 'ape_position_m'])
        writer.writerows(zip(timestamps, errors))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=['kitti', 'm2dgr'])
    parser.add_argument('--sequence', required=True)
    parser.add_argument('--variant', required=True, choices=['full', 'no_f2f', 'no_vertical', 'neither'])
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--ground-truth', required=True, type=Path)
    parser.add_argument('--expected-frames', required=True, type=int)
    args = parser.parse_args()
    print(json.dumps(evaluate_job(args.dataset, args.sequence, args.variant, args.run,
                                 args.ground_truth, args.expected_frames), indent=2))
