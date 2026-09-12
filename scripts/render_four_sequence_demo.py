#!/usr/bin/env python3
"""Render a 2x2 mapping demo from existing TUM poses and indexed ROS1 bags.

This is offline visualization, not an odometry benchmark. Selected point clouds
are transformed by their recorded poses without GT, smoothing, or calibration.
Only the video, a poster, and small provenance JSON are written to disk.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import rosbag


WIDTH, HEIGHT = 1920, 1080
PANEL_W, PANEL_H = 936, 446
PANEL_POS = [(18, 100), (966, 100), (18, 558), (966, 558)]
BG = (8, 15, 21)
INK = (226, 237, 241)
MUTED = (136, 163, 177)
CYAN = (80, 227, 224)
ORANGE = (255, 166, 72)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quaternion_rotation(q):
    x, y, z, w = q
    if abs(float(q @ q) - 1.0) > 1e-6:
        raise ValueError('Non-unit recorded quaternion')
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def native_xyz(message):
    if message.is_bigendian:
        raise ValueError('Only little-endian input is supported')
    fields = {f.name: f for f in message.fields}
    if any(fields[k].datatype != 7 or fields[k].count != 1 for k in ('x', 'y', 'z')):
        raise ValueError('XYZ fields must be FLOAT32')
    dtype = np.dtype({'names': ['x', 'y', 'z'], 'formats': ['<f4'] * 3,
                      'offsets': [fields[k].offset for k in ('x', 'y', 'z')],
                      'itemsize': message.point_step})
    cloud = np.ndarray((message.height, message.width), dtype=dtype,
                       buffer=message.data, strides=(message.row_step, message.point_step))
    return np.column_stack([cloud[k].ravel() for k in ('x', 'y', 'z')])


def load_panel(spec, scans, max_points):
    trajectory_path, bag_path = Path(spec['trajectory']), Path(spec['bag'])
    poses = np.loadtxt(trajectory_path)
    if (poses.ndim != 2 or poses.shape[1] != 8 or len(poses) < 3
            or not np.isfinite(poses).all() or np.any(np.diff(poses[:, 0]) <= 0)):
        raise ValueError('Invalid complete TUM trajectory: ' + spec['label'])
    timing = np.genfromtxt(spec['timing'], delimiter=',', names=True)
    if (len(timing) != len(poses) or not np.array_equal(timing['timestamp'], poses[:, 0])
            or not np.array_equal(timing['frame'], np.arange(1, len(poses)+1))):
        raise ValueError('Trajectory and timing frame association mismatch')
    if not np.all(timing['vertical_enabled'] == 1) or not np.all(timing['f2f_enabled'] == 1):
        raise ValueError('The demo requires the full NSVR method')
    selected = np.unique(np.linspace(0, len(poses)-1, min(scans, len(poses))).round().astype(int))
    selected_set = set(selected.tolist())
    chunks, elevations, sampled = [], [], []
    with rosbag.Bag(str(bag_path)) as bag:
        info = bag.get_type_and_topic_info().topics[spec['topic']]
        if info.message_count != len(poses):
            raise ValueError('The trajectory must cover all input scans')
        connections = list(bag._get_connections(topics=[spec['topic']]))
        # Noetic's indexed reader avoids deserializing tens of GB of unused data.
        entries = list(bag._get_entries(connections))
        if len(entries) != len(poses):
            raise ValueError('Unexpected bag index count')
        for i in sorted(selected_set):
            entry = entries[i]
            _, message, _ = bag._read_message((entry.chunk_pos, entry.offset))
            if abs(message.header.stamp.to_sec() - poses[i, 0]) > 1e-6:
                raise ValueError('Native header and recorded pose timestamp differ')
            xyz = native_xyz(message)
            ranges = np.linalg.norm(xyz, axis=1)
            keep = (np.isfinite(xyz).all(axis=1) & (ranges > spec['min_range'])
                    & (ranges < spec['max_range']))
            xyz = xyz[keep]
            if len(xyz) > max_points:
                xyz = xyz[np.linspace(0, len(xyz)-1, max_points).astype(int)]
            world = xyz @ quaternion_rotation(poses[i, 4:]).T + poses[i, 1:4]
            chunks.append(world.astype(np.float32))
            elevations.append(xyz[:, 2].copy())
            sampled.append(i)
            if len(sampled) % 60 == 0:
                print('%s: sampled %d / %d clouds' % (spec['label'], len(sampled), len(selected)), flush=True)
    points = np.concatenate(chunks)
    # A fixed orthographic camera is chosen per sequence, independent of GT.
    cov = np.cov(poses[:, 1:3].T)
    _, eig = np.linalg.eigh(cov)
    major = eig[:, -1]
    if major[0] < 0:
        major = -major
    minor = np.array([-major[1], major[0]])
    projection = np.array([[*major, 0.0], [*(0.90*minor), 0.435]])
    projected = points @ projection.T
    projected_path = poses[:, 1:4] @ projection.T
    low = np.minimum(projected.min(axis=0), projected_path.min(axis=0))
    high = np.maximum(projected.max(axis=0), projected_path.max(axis=0))
    margin = spec.get('view_margin_m')
    if margin is not None:
        if not np.isfinite(margin) or margin <= 0:
            raise ValueError('Camera margin must be positive')
        # Indoor camera framing suppresses distant returns outside the viewport,
        # never trajectory samples. It does not alter the cloud or pose estimate.
        low = projected_path.min(axis=0) - margin
        high = projected_path.max(axis=0) + margin
    center = (low + high) * 0.5
    scale = min((PANEL_W-76) / max(high[0]-low[0], 1.0),
                (PANEL_H-128) / max(high[1]-low[1], 1.0))

    def pixel_xy(p):
        p = (p-center) * [scale, -scale] + [PANEL_W/2, (PANEL_H+60)/2]
        return np.rint(p).astype(np.int32)

    projected_chunks = np.split(pixel_xy(projected), np.cumsum([len(c) for c in chunks])[:-1])
    colors = []
    for z in elevations:
        a = np.clip((z + 1.8) / 5.0, 0, 1)[:, None]
        c = (1-a) * np.array([29, 97, 125]) + a * np.array([147, 207, 205])
        colors.append(c.astype(np.uint8))
    provenance = {'label': spec['label'], 'bag': bag_path.name,
                  'bag_bytes': bag_path.stat().st_size, 'trajectory_sha256': digest(trajectory_path),
                  'timing_sha256': digest(spec['timing']), 'complete_pose_count': len(poses),
                  'sampled_clouds': len(selected), 'visualized_points': len(points),
                  'header_duration_s': float(poses[-1, 0]-poses[0, 0]),
                  'all_sampled_header_stamps_match': True,
                  'camera_margin_m': margin,
                  'range_mask_m': [spec['min_range'], spec['max_range']]}
    print(spec['label'] + ': indexing, timestamps and projection checked', flush=True)
    return {'poses': poses, 'xy': projected_chunks, 'colors': colors,
            'indices': np.asarray(sampled), 'path': pixel_xy(projected_path),
            'canvas': np.full((PANEL_H, PANEL_W, 3), BG, np.uint8),
            'cursor': 0, 'scale': scale, 'provenance': provenance}


def font(size, bold=False, mono=False):
    family = 'DejaVuSansMono' if mono else 'DejaVuSans'
    suffix = '-Bold' if bold else ''
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/' + family + suffix + '.ttf', size)


def render_panel(panel, fraction):
    poses = panel['poses']
    index = min(int(round(fraction*(len(poses)-1))), len(poses)-1)
    while panel['cursor'] < len(panel['indices']) and panel['indices'][panel['cursor']] <= index:
        k = panel['cursor']
        xy = panel['xy'][k]
        visible = (xy[:, 0] >= 1) & (xy[:, 0] < PANEL_W-1) & (xy[:, 1] >= 63) & (xy[:, 1] < PANEL_H-23)
        p, c = xy[visible], panel['colors'][k][visible]
        np.maximum.at(panel['canvas'], (p[:, 1], p[:, 0]), c)
        panel['cursor'] += 1
    frame = panel['canvas'].copy()
    cv2.polylines(frame, [panel['path'][:index+1]], False, ORANGE, 2, cv2.LINE_AA)
    position = tuple(panel['path'][index])
    cv2.circle(frame, position, 9, (19, 72, 83), -1, cv2.LINE_AA)
    cv2.circle(frame, position, 6, CYAN, -1, cv2.LINE_AA)
    cv2.circle(frame, position, 9, (204, 250, 252), 1, cv2.LINE_AA)
    canvas = Image.fromarray(frame)
    d = ImageDraw.Draw(canvas)
    d.rectangle((0, 0, PANEL_W-1, 58), fill=(13, 24, 33))
    d.text((20, 10), panel['provenance']['label'], font=font(25, bold=True), fill=INK)
    elapsed = poses[index, 0]-poses[0, 0]
    d.text((20, 40), 'NSVR  |  %s / %s poses  |  scan time %.1f s' %
           (format(index+1, ','), format(len(poses), ','), elapsed), font=font(14, mono=True), fill=MUTED)
    d.text((PANEL_W-183, 17), '%5.1f%% complete' % (100*fraction), font=font(16, mono=True), fill=CYAN)
    target_m = 110.0 / panel['scale']
    power = 10.0 ** np.floor(np.log10(target_m))
    length_m = min((power, 2*power, 5*power, 10*power), key=lambda x: abs(x-target_m))
    length_px = int(round(length_m * panel['scale']))
    y = PANEL_H-27
    d.line((20, y, 20+length_px, y), fill=INK, width=2)
    d.line((20, y-3, 20, y+3), fill=INK, width=2)
    d.line((20+length_px, y-3, 20+length_px, y+3), fill=INK, width=2)
    d.text((28+length_px, y-8), '%g m' % length_m, font=font(14, mono=True), fill=INK)
    d.rectangle((1, PANEL_H-6, PANEL_W-2, PANEL_H-2), fill=(25, 46, 58))
    d.rectangle((1, PANEL_H-6, max(1, int((PANEL_W-2)*fraction)), PANEL_H-2), fill=CYAN)
    d.rectangle((0, 0, PANEL_W-1, PANEL_H-1), outline=(51, 73, 86), width=1)
    return canvas


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True, help='JSON list of exactly four input specifications')
    p.add_argument('--output', type=Path, required=True, help='New output directory')
    p.add_argument('--seconds', type=float, default=28)
    p.add_argument('--fps', type=int, default=24)
    p.add_argument('--sampled-scans', type=int, default=300)
    p.add_argument('--points-per-scan', type=int, default=8000)
    args = p.parse_args()
    specs = json.loads(args.manifest.read_text())
    if len(specs) != 4 or not 6 <= args.seconds <= 60 or not 12 <= args.fps <= 30:
        p.error('Four sequences, 6..60 seconds and 12..30 fps required')
    if not 3 <= args.sampled_scans <= 500 or not 100 <= args.points_per_scan <= 15000:
        p.error('Visualization sampling limits exceeded')
    args.output.mkdir(parents=True, exist_ok=False)
    panels = [load_panel(s, args.sampled_scans, args.points_per_scan) for s in specs]
    video = args.output / 'nsvr_four_sequence_demo.mp4'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'avc1'), args.fps, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError('An OpenCV/FFmpeg H.264 encoder is required')
    count = int(round(args.seconds * args.fps))
    active = count - 2*args.fps
    try:
        for i in range(count):
            fraction = min(i / max(active-1, 1), 1.0)
            frame = Image.new('RGB', (WIDTH, HEIGHT), BG)
            draw = ImageDraw.Draw(frame)
            draw.text((22, 14), 'NSVR', font=font(35, bold=True), fill=CYAN)
            draw.text((152, 18), 'LiDAR Odometry', font=font(29, bold=True), fill=INK)
            draw.text((22, 61), 'FOUR-SEQUENCE MAPPING REPLAY', font=font(18, mono=True), fill=MUTED)
            draw.text((1155, 26), 'Recorded NSVR poses + original LiDAR scans', font=font(20), fill=INK)
            for panel, pos in zip(panels, PANEL_POS):
                frame.paste(render_panel(panel, fraction), pos)
            draw.text((23, 1020), 'Orange: estimated trajectory     Cyan marker: current pose', font=font(18), fill=INK)
            draw.text((23, 1050), 'Accelerated visualization | Normalized sequence progress | Display-only point-cloud subsampling', font=font(15), fill=MUTED)
            draw.text((1517, 1020), 'Original evaluation: 1.0x', font=font(18), fill=CYAN)
            writer.write(cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))
            if i == count-1:
                frame.save(args.output / 'nsvr_four_sequence_poster.png')
            if i % args.fps == 0:
                print('Encoded %d / %d frames' % (i, count), flush=True)
    finally:
        writer.release()
    capture = cv2.VideoCapture(str(video))
    actual = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    codec = int(capture.get(cv2.CAP_PROP_FOURCC))
    capture.release()
    if actual != count or abs(fps-args.fps) > 1e-6:
        raise RuntimeError('Encoded video failed frame/fps verification')
    record = {'method': 'NSVR', 'source_core_commit': 'def06cff6b0aedf1207dad3887aa705f8a29d638',
              'width': WIDTH, 'height': HEIGHT, 'fps': fps, 'frames': actual,
              'seconds': actual/fps, 'codec': ''.join(chr((codec >> (8*k)) & 255) for k in range(4)),
              'video_sha256': digest(video), 'video_bytes': video.stat().st_size,
              'visualization_only': True, 'sequence_progress_normalized': True,
              'ground_truth_used': False, 'new_odometry_runs': False,
              'projection': 'Fixed per-sequence orthographic camera; no pose smoothing, GT alignment or sensor correction.',
              'sampling': 'Uniform selected scans and point rows for display only; source trajectories cover every input scan.',
              'sequences': [v['provenance'] for v in panels]}
    (args.output / 'nsvr_four_sequence_demo.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    main()
