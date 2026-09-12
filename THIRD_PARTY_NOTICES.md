# Third-party notices and source provenance

The authors' contributions to this release are available under the [MIT License](LICENSE).
This grant does not replace third-party copyright notices or change the licenses
of inherited code, external dependencies, or datasets. Keep this file and the
applicable texts in `LICENSES/` with copies or substantial portions of the software.

## Inherited KISS-ICP code

Parts of the base registration, voxel map, adaptive-threshold, and motion-prediction
implementation descend from KISS-ICP through the earlier `qfwang23/LO` codebase.
The direct predecessor recorded for the frozen implementation is
[`qfwang23/LO` at `17825a3`](https://github.com/qfwang23/LO/tree/17825a3).
This release preserves the frozen algorithm files; the upstream notices are
collected here and in [LICENSES/KISS-ICP-MIT.txt](LICENSES/KISS-ICP-MIT.txt).

KISS-ICP copyright:

> Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill Stachniss.

The following mapping documents retained or historically inherited portions; it
does not claim that each current file is an unchanged upstream copy.

| Release file(s) | Inherited portions and reference implementation |
| --- | --- |
| `cpp/LO/pipeline/LO.cpp` | Motion prediction, motion-activation test, preprocessing/downsampling, and adaptive-threshold structure; [KissICP.cpp](https://github.com/PRBonn/kiss-icp/blob/v0.2.0/cpp/kiss_icp/pipeline/KissICP.cpp) and [Threshold.cpp](https://github.com/PRBonn/kiss-icp/blob/v0.2.0/cpp/kiss_icp/core/Threshold.cpp). |
| `cpp/LO/pipeline/LO.hpp` | Adaptive-threshold state, constructor, and model-deviation update; [Threshold.hpp](https://github.com/PRBonn/kiss-icp/blob/v0.2.0/cpp/kiss_icp/core/Threshold.hpp). |
| `cpp/LO/core/VoxelHashMap.cpp`, `cpp/LO/core/VoxelHashMap.hpp` | Voxel-map representation, update routines, and historical correspondence-search structure, subsequently modified; [VoxelHashMap.cpp](https://github.com/PRBonn/kiss-icp/blob/v0.2.0/cpp/kiss_icp/core/VoxelHashMap.cpp) and [VoxelHashMap.hpp](https://github.com/PRBonn/kiss-icp/blob/v0.2.0/cpp/kiss_icp/core/VoxelHashMap.hpp). |
| `cpp/LO/core/Registration.cpp`, `cpp/LO/core/Registration.hpp` | Historical point-registration scaffolding and interface, subsequently substantially reworked; [Registration.cpp](https://github.com/PRBonn/kiss-icp/blob/v0.2.0/cpp/kiss_icp/core/Registration.cpp) and [Registration.hpp](https://github.com/PRBonn/kiss-icp/blob/v0.2.0/cpp/kiss_icp/core/Registration.hpp). |

The `v0.2.0` links identify an official upstream version containing the matched
implementation. The exact original import commit has not been established; the
links are not a claim that this was the only source version. Later NSVR surface
refinement and reference-management changes do not remove the inherited notices.
The authoritative license text for the cited version is the
[KISS-ICP MIT license](https://github.com/PRBonn/kiss-icp/blob/v0.2.0/LICENSE).

## Build-time dependencies

The repository does not vendor dependency source trees or distribute compiled
binaries. CMake obtains the following header dependencies; their upstream
licenses are preserved for convenience and remain applicable to included code.

| Dependency | Configured version | License and retained notice |
| --- | --- | --- |
| Sophus | `1.22.11`, `nachovizzo/Sophus` | MIT; Copyright 2011–2017 Hauke Strasdat and 2012–2017 Steven Lovegrove. [Local text](LICENSES/Sophus-MIT.txt); [versioned upstream text](https://github.com/nachovizzo/Sophus/blob/1.22.11/LICENSE.txt). |
| robin-map | `v1.2.1`, `Tessil/robin-map` | MIT; Copyright (c) 2017 Thibaut Goetghebuer-Planchon. [Local text](LICENSES/robin-map-MIT.txt); [versioned upstream text](https://github.com/Tessil/robin-map/blob/v1.2.1/LICENSE). |

Eigen, TBB, ROS, Python, and the Python packages used by the scripts are supplied
by the user's environment, not relicensed by this repository. A distributor of
binaries, containers, or vendored dependencies must include the applicable
notices and meet the obligations for the actual dependency versions and files
distributed. This source-release notice is not a complete binary-package license
manifest.

## Datasets and paper artifacts

KITTI and M2DGR inputs, raw trajectories, and dataset ground-truth files are not
distributed with this source release. Obtain datasets separately under their
respective terms. The software license does not grant rights to those datasets.
The paper's citations and the visualizations in the README do not replace the
software notices above.
