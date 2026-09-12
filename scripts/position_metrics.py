"""Frozen position-only APE primitives: unit-scale alignment and gap-aware pairing."""
import math
import numpy as np


def load_tum_positions(path):
    """Load strictly time-ordered timestamp/position columns from a TUM file."""
    timestamps = []
    positions = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            columns = stripped.split()
            if len(columns) < 4:
                raise ValueError(
                    f"{path}:{line_number}: expected timestamp and xyz columns"
                )
            values = np.asarray([float(value) for value in columns[:4]], dtype=float)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{path}:{line_number}: non-finite timestamp/position")
            timestamps.append(values[0])
            positions.append(values[1:4])

    if not timestamps:
        raise ValueError(f"{path}: no trajectory rows")
    timestamps = np.asarray(timestamps, dtype=float)
    positions = np.asarray(positions, dtype=float)
    if np.any(np.diff(timestamps) <= 0.0):
        raise ValueError(f"{path}: timestamps must be strictly increasing")
    return timestamps, positions


def robust_ground_truth_gap_model(timestamps):
    """Identify anomalously long GT intervals using only that GT's cadence.

    The one-sided fence is median(dt) plus six robust standard deviations.  Its
    scale is the larger of Gaussian-consistent MAD and normalized IQR, with a
    floating-point resolution floor.  Thus no sequence class or hand-selected
    time-gap threshold enters the decision.
    """
    timestamps = np.asarray(timestamps, dtype=float)
    intervals = np.diff(timestamps)
    if intervals.size == 0:
        return {
            "intervals": intervals,
            "gap_mask": np.zeros(0, dtype=bool),
            "median_s": math.nan,
            "mad_s": math.nan,
            "iqr_s": math.nan,
            "robust_scale_s": math.nan,
            "threshold_s": math.inf,
        }
    if np.any(~np.isfinite(intervals)) or np.any(intervals <= 0.0):
        raise ValueError("ground-truth timestamps must have positive finite intervals")

    median = float(np.median(intervals))
    mad = float(np.median(np.abs(intervals - median)))
    q1, q3 = np.quantile(intervals, [0.25, 0.75])
    iqr = float(q3 - q1)
    # 1.4826*MAD and IQR/1.349 are robust estimates of Gaussian sigma.
    timestamp_magnitude = max(1.0, float(np.max(np.abs(timestamps))))
    resolution_floor = 32.0 * float(np.spacing(timestamp_magnitude))
    robust_scale = max(1.4826 * mad, iqr / 1.349, resolution_floor)
    threshold = median + 6.0 * robust_scale
    return {
        "intervals": intervals,
        "gap_mask": intervals > threshold,
        "median_s": median,
        "mad_s": mad,
        "iqr_s": iqr,
        "robust_scale_s": robust_scale,
        "threshold_s": threshold,
    }


def interpolate_ground_truth_positions(gt_timestamps, gt_positions,
                                       estimate_timestamps, gap_mask):
    """Interpolate only within GT bounds and non-gap bracketing intervals."""
    gt_timestamps = np.asarray(gt_timestamps, dtype=float)
    gt_positions = np.asarray(gt_positions, dtype=float)
    estimate_timestamps = np.asarray(estimate_timestamps, dtype=float)
    gap_mask = np.asarray(gap_mask, dtype=bool)
    if gt_positions.shape != (gt_timestamps.size, 3):
        raise ValueError("ground-truth positions must have shape (N, 3)")
    if gap_mask.shape != (max(0, gt_timestamps.size - 1),):
        raise ValueError("gap mask must describe every consecutive GT interval")

    valid_estimate_indices = []
    interpolated_positions = []
    left_indices = []
    right_indices = []
    alphas = []
    outside_count = 0
    gap_count = 0

    for estimate_index, timestamp in enumerate(estimate_timestamps):
        if not math.isfinite(float(timestamp)):
            outside_count += 1
            continue
        right = int(np.searchsorted(gt_timestamps, timestamp, side="left"))
        if right < gt_timestamps.size and gt_timestamps[right] == timestamp:
            valid_estimate_indices.append(estimate_index)
            interpolated_positions.append(gt_positions[right])
            left_indices.append(right)
            right_indices.append(right)
            alphas.append(0.0)
            continue
        if right == 0 or right == gt_timestamps.size:
            outside_count += 1
            continue

        left = right - 1
        if gap_mask[left]:
            gap_count += 1
            continue
        interval = gt_timestamps[right] - gt_timestamps[left]
        alpha = float((timestamp - gt_timestamps[left]) / interval)
        valid_estimate_indices.append(estimate_index)
        interpolated_positions.append(
            (1.0 - alpha) * gt_positions[left] + alpha * gt_positions[right]
        )
        left_indices.append(left)
        right_indices.append(right)
        alphas.append(alpha)

    return {
        "estimate_indices": np.asarray(valid_estimate_indices, dtype=np.int64),
        "positions": np.asarray(interpolated_positions, dtype=float).reshape(-1, 3),
        "left_gt_indices": np.asarray(left_indices, dtype=np.int64),
        "right_gt_indices": np.asarray(right_indices, dtype=np.int64),
        "alpha": np.asarray(alphas, dtype=float),
        "outside_gt_range_count": outside_count,
        "across_gt_gap_count": gap_count,
    }


def align_positions_se3(estimate_positions, ground_truth_positions):
    """Unit-scale, orientation-preserving Kabsch alignment estimate -> GT."""
    estimate_positions = np.asarray(estimate_positions, dtype=float)
    ground_truth_positions = np.asarray(ground_truth_positions, dtype=float)
    if estimate_positions.shape != ground_truth_positions.shape:
        raise ValueError("estimate and ground truth must have identical shapes")
    if estimate_positions.ndim != 2 or estimate_positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if estimate_positions.shape[0] < 3:
        raise ValueError("at least three associated positions are required")
    if not (np.all(np.isfinite(estimate_positions)) and
            np.all(np.isfinite(ground_truth_positions))):
        raise ValueError("positions must be finite")

    estimate_center = np.mean(estimate_positions, axis=0)
    ground_truth_center = np.mean(ground_truth_positions, axis=0)
    centered_estimate = estimate_positions - estimate_center
    centered_ground_truth = ground_truth_positions - ground_truth_center
    covariance = centered_estimate.T @ centered_ground_truth
    left_vectors, _, right_vectors_transposed = np.linalg.svd(covariance)
    rotation = right_vectors_transposed.T @ left_vectors.T
    if np.linalg.det(rotation) < 0.0:
        right_vectors_transposed[-1, :] *= -1.0
        rotation = right_vectors_transposed.T @ left_vectors.T
    translation = ground_truth_center - rotation @ estimate_center
    aligned = (rotation @ estimate_positions.T).T + translation
    return aligned, rotation, translation


def position_error_stats(errors):
    errors = np.asarray(errors, dtype=float)
    if errors.size == 0:
        return {}
    return {
        "rmse": float(np.sqrt(np.mean(errors * errors))),
        "mean": float(np.mean(errors)),
        "std": float(np.std(errors)),
        "median": float(np.median(errors)),
        "min": float(np.min(errors)),
        "max": float(np.max(errors)),
        "sse": float(np.sum(errors * errors)),
    }
