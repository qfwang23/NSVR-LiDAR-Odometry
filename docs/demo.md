# Four-sequence mapping demo

The 1920 × 1080, 24 fps, 28-second video combines **KITTI 00**, **KITTI 02**,
**M2DGR Hall01**, and **M2DGR Room01** in four panels. Its input trajectories
are the complete NSVR runs used for the manuscript, not another method's poses
or a new experiment. The trajectories contain 4,541, 4,661, 3,511, and 728 poses.

Each selected original scan is placed with its recorded native-LiDAR-to-world
pose. Header timestamps are checked against the corresponding trajectory and
timing rows. Bag recording times and ROS frame-name strings are not used to
rotate or associate the point clouds. No ground truth, additional calibration,
pose smoothing, or re-estimation is used.

## How to read the video

- Blue/cyan points: accumulated observations from selected original scans.
- Orange line: recorded estimated trajectory through the currently shown pose.
- Cyan marker: current recorded pose.
- Each panel has a fixed orthographic view and a projected-distance scale bar.
- Indoor views focus on the trajectory and nearby structure; distant point
  returns outside the viewport are clipped, but the complete trajectory is shown.
- The four sequences advance according to **normalized sequence progress**.
  Their physical durations differ, so the panels are not simultaneous physical
  recordings. The final two seconds hold the completed maps.

The video is an accelerated visualization, not a real-time performance
measurement. The original odometry evaluations used 1.0x recorded-speed replay
and processed all input scans. Rendering selects 300 scans per sequence and at
most 8,000 points per selected scan for display only. This visualization
subsampling does not indicate frame dropping by the odometry.

## Reproduce the rendering

Use the ROS Noetic environment already required by this repository. The
optional renderer additionally requires NumPy, Pillow, OpenCV with an FFmpeg
H.264 encoder, and the DejaVu fonts. On the reference Ubuntu environment the
additional packages are `python3-opencv`, `python3-pil`, and `fonts-dejavu-core`.

Prepare a JSON list of four input specifications, following
[demo_inputs.example.json](demo_inputs.example.json). Replace placeholder paths
with your input bags and complete NSVR run outputs. Run from the repository root:

```bash
/usr/bin/python3 -B scripts/render_four_sequence_demo.py \
  --manifest /path/to/demo_inputs.json \
  --output /path/to/new-demo-output
```

The output directory must not exist. Rendering writes only an H.264 MP4, a PNG
poster, and a small JSON provenance record. Point clouds are read by indexed
access and retained only in memory for the visualization; no PCD files, raw
frame images, or accumulating cloud files are written. The renderer uses the
ROS Noetic indexed bag reader and has been checked on the supplied bag layouts,
including the 22-byte PointCloud2 records in M2DGR.

Video and input-trajectory hashes, frame counts, sampled-scan counts, camera
settings, and the recorded source revision are in
[media/nsvr_four_sequence_demo.json](media/nsvr_four_sequence_demo.json).
The source datasets and raw trajectories are not distributed here.
