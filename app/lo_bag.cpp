#include <sophus/se3.hpp>

#include "LO/pipeline/LO.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ros/time.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <limits>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <sys/resource.h>
#include <unistd.h>
#include <tbb/global_control.h>

namespace {

struct Arguments {
    std::string bag;
    std::string topic = "/velodyne_points";
    std::string trajectory;
    std::string timing;
    std::string variant = "full";
    double min_range = 1.0;
    double max_range = 100.0;
    double voxel_size = 0.4;
    int max_points_per_voxel = 15;
    double initial_threshold = 2.0;
    double min_motion_threshold = 0.1;
    double replay_rate = 1.0;
    std::size_t expected_frames = 0;
    std::size_t max_frames = 0;
    std::size_t threads = 0;
};

[[noreturn]] void Usage(const char* program, const std::string& error = {}) {
    if (!error.empty()) std::cerr << "ERROR: " << error << "\n\n";
    std::cerr
        << "Usage: " << program
        << " --bag FILE --trajectory FILE --timing FILE --expected-frames N [options]\n"
        << "  --topic NAME                 PointCloud2 topic\n"
        << "  --min-range M               Dataset-wide near range\n"
        << "  --max-range M               Dataset-wide far range\n"
        << "  --voxel-size M              Dataset-wide map voxel edge\n"
        << "  --max-points-per-voxel N    Dataset-wide voxel capacity\n"
        << "  --initial-threshold M       Original adaptive-threshold initializer\n"
        << "  --min-motion-th M           Original adaptive-threshold motion gate\n"
        << "  --replay-rate 1.0           Fixed recorded-time playback rate\n"
        << "  --variant NAME              full, no_f2f, no_vertical, neither\n"
        << "  --threads N                 TBB concurrency limit (0: library default)\n"
        << "  --max-frames N              Smoke test only (0: entire sequence)\n";
    std::exit(error.empty() ? 0 : 2);
}

std::string RequireValue(int argc, char** argv, int& index) {
    if (index + 1 >= argc) Usage(argv[0], std::string("missing value for ") + argv[index]);
    return argv[++index];
}

double StrictDouble(const std::string& value, const std::string& option) {
    std::size_t consumed = 0;
    double parsed = 0.0;
    try {
        parsed = std::stod(value, &consumed);
    } catch (const std::exception&) {
        Usage("lo_bag", "invalid value for " + option + ": " + value);
    }
    if (consumed != value.size() || !std::isfinite(parsed)) {
        Usage("lo_bag", "invalid value for " + option + ": " + value);
    }
    return parsed;
}

std::size_t StrictSize(const std::string& value, const std::string& option) {
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos) {
        Usage("lo_bag", "invalid unsigned integer for " + option);
    }
    std::size_t consumed = 0;
    unsigned long long parsed = 0;
    try {
        parsed = std::stoull(value, &consumed);
    } catch (const std::exception&) {
        Usage("lo_bag", "invalid value for " + option + ": " + value);
    }
    if (consumed != value.size() || parsed > std::numeric_limits<std::size_t>::max()) {
        Usage("lo_bag", "invalid value for " + option + ": " + value);
    }
    return static_cast<std::size_t>(parsed);
}

Arguments ParseArguments(int argc, char** argv) {
    Arguments args;
    for (int index = 1; index < argc; ++index) {
        const std::string option(argv[index]);
        if (option == "--bag") {
            args.bag = RequireValue(argc, argv, index);
        } else if (option == "--topic") {
            args.topic = RequireValue(argc, argv, index);
        } else if (option == "--trajectory") {
            args.trajectory = RequireValue(argc, argv, index);
        } else if (option == "--timing") {
            args.timing = RequireValue(argc, argv, index);
        } else if (option == "--min-range") {
            args.min_range = StrictDouble(RequireValue(argc, argv, index), option);
        } else if (option == "--max-range") {
            args.max_range = StrictDouble(RequireValue(argc, argv, index), option);
        } else if (option == "--voxel-size") {
            args.voxel_size = StrictDouble(RequireValue(argc, argv, index), option);
        } else if (option == "--max-points-per-voxel") {
            const auto capacity = StrictSize(RequireValue(argc, argv, index), option);
            if (capacity > static_cast<std::size_t>(std::numeric_limits<int>::max()))
                Usage(argv[0], "voxel capacity exceeds int range");
            args.max_points_per_voxel = static_cast<int>(capacity);
        } else if (option == "--initial-threshold") {
            args.initial_threshold = StrictDouble(RequireValue(argc, argv, index), option);
        } else if (option == "--min-motion-th") {
            args.min_motion_threshold = StrictDouble(RequireValue(argc, argv, index), option);
        } else if (option == "--replay-rate") {
            args.replay_rate = StrictDouble(RequireValue(argc, argv, index), option);
        } else if (option == "--expected-frames") {
            args.expected_frames = StrictSize(RequireValue(argc, argv, index), option);
        } else if (option == "--max-frames") {
            args.max_frames = StrictSize(RequireValue(argc, argv, index), option);
        } else if (option == "--variant") {
            args.variant = RequireValue(argc, argv, index);
        } else if (option == "--threads") {
            args.threads = StrictSize(RequireValue(argc, argv, index), option);
        } else if (option == "-h" || option == "--help") {
            Usage(argv[0]);
        } else {
            Usage(argv[0], "unknown option: " + option);
        }
    }
    if (args.bag.empty()) Usage(argv[0], "--bag is required");
    if (args.trajectory.empty()) Usage(argv[0], "--trajectory is required");
    if (args.timing.empty()) Usage(argv[0], "--timing is required");
    if (args.expected_frames == 0) Usage(argv[0], "--expected-frames must be positive");
    if (!(args.min_range >= 0.0 && args.max_range > args.min_range)) {
        Usage(argv[0], "range limits must satisfy 0 <= min < max");
    }
    if (!(args.voxel_size > 0.0) || args.max_points_per_voxel <= 0 ||
        !(args.initial_threshold > 0.0) || !(args.min_motion_threshold >= 0.0)) {
        Usage(argv[0], "voxel, capacity, threshold, and motion parameters are invalid");
    }
    if (args.variant != "full" && args.variant != "no_f2f" &&
        args.variant != "no_vertical" && args.variant != "neither") {
        Usage(argv[0], "unknown ablation variant");
    }
    if (args.replay_rate != 1.0) {
        Usage(argv[0], "recorded-time playback is fixed at exactly 1.0x");
    }
    return args;
}

std::vector<Eigen::Vector3d> ConvertPointCloud(const sensor_msgs::PointCloud2& message) {
    if (message.is_bigendian || message.point_step == 0 ||
        message.row_step != static_cast<std::size_t>(message.width) * message.point_step ||
        message.data.size() != static_cast<std::size_t>(message.row_step) * message.height) {
        throw std::runtime_error("requires packed little-endian PointCloud2 data");
    }
    for (const std::string name : {"x", "y", "z"}) {
        const auto field = std::find_if(message.fields.begin(), message.fields.end(),
            [&](const auto& f) { return f.name == name; });
        if (field == message.fields.end() || field->datatype != sensor_msgs::PointField::FLOAT32 ||
            field->count != 1 || static_cast<std::size_t>(field->offset) + sizeof(float) > message.point_step) {
            throw std::runtime_error("requires valid FLOAT32 XYZ fields");
        }
    }
    const std::size_t count = static_cast<std::size_t>(message.width) * message.height;
    std::vector<Eigen::Vector3d> points;
    points.reserve(count);
    sensor_msgs::PointCloud2ConstIterator<float> x(message, "x");
    sensor_msgs::PointCloud2ConstIterator<float> y(message, "y");
    sensor_msgs::PointCloud2ConstIterator<float> z(message, "z");
    for (std::size_t index = 0; index < count; ++index, ++x, ++y, ++z) {
        if (std::isfinite(*x) && std::isfinite(*y) && std::isfinite(*z)) {
            // Preserve native XYZ. No fixed sensor-frame rotation is applied.
            points.emplace_back(*x, *y, *z);
        }
    }
    if (points.empty()) throw std::runtime_error("point cloud contains no finite XYZ point");
    return points;
}

void WritePose(std::ostream& stream, double stamp, const Sophus::SE3d& pose) {
    const Eigen::Vector3d& translation = pose.translation();
    const Eigen::Quaterniond quaternion = pose.unit_quaternion();
    if (!std::isfinite(stamp) || !translation.allFinite() ||
        !quaternion.coeffs().allFinite() ||
        std::abs(quaternion.norm() - 1.0) > 1e-6) {
        throw std::runtime_error("non-finite or invalid pose output");
    }
    stream << std::fixed << std::setprecision(9) << stamp << ' '
           << std::setprecision(12) << translation.x() << ' ' << translation.y() << ' '
           << translation.z() << ' ' << quaternion.x() << ' ' << quaternion.y() << ' '
           << quaternion.z() << ' ' << quaternion.w() << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Arguments args = ParseArguments(argc, argv);
        // Fail with bad_alloc before exhausting host RAM/swap. This does not prune
        // registration data or silently drop scans. Partial text outputs survive.
        struct rlimit address_limit;
        if (getrlimit(RLIMIT_AS, &address_limit) != 0) {
            throw std::runtime_error("cannot inspect address-space resource limit");
        }
        address_limit.rlim_cur = std::min(address_limit.rlim_cur, rlim_t(8ULL << 30));
        if (setrlimit(RLIMIT_AS, &address_limit) != 0) {
            throw std::runtime_error("cannot install 8-GiB address-space safety limit");
        }
        tbb::global_control concurrency(tbb::global_control::max_allowed_parallelism,
            args.threads ? args.threads : tbb::global_control::active_value(
                tbb::global_control::max_allowed_parallelism));
        ros::Time::init();

        if (std::filesystem::absolute(args.trajectory).lexically_normal() ==
                std::filesystem::absolute(args.timing).lexically_normal() ||
            std::filesystem::exists(args.trajectory) || std::filesystem::exists(args.timing)) {
            throw std::runtime_error("output files must be distinct and must not already exist");
        }
        std::ofstream trajectory(args.trajectory, std::ios::trunc);
        trajectory.exceptions(std::ios::badbit | std::ios::failbit);
        std::ofstream timing(args.timing, std::ios::trunc);
        timing.exceptions(std::ios::badbit | std::ios::failbit);
        if (!timing) throw std::runtime_error("cannot open timing file: " + args.timing);
        timing
            << "frame,timestamp,algorithm_ms,points,registration_ms,f2f_ms,f2m_ms,processing_ms,"
               "pipeline_ms,f2f_enabled,vertical_enabled,f2f_executed,"
               "deadline_lateness_ms,wall_elapsed_s,bag_elapsed_s,"
               "rss_mib,peak_rss_mib,local_voxels,local_buckets,global_voxels,global_buckets,"
               "ff_iterations,ff_vertical_gates,ff_zero_steps,ff_clipped_steps,ff_full_fallbacks,ff_stage_clipped,"
               "lm_iterations,lm_vertical_gates,lm_zero_steps,lm_clipped_steps,lm_full_fallbacks,lm_stage_clipped,"
               "ff_correspondences,ff_geometry_matches,lm_correspondences,lm_geometry_matches,"
               "vertical_surface_ms,vs_source_patches,vs_reference_patches,vs_matched_patches,"
               "vs_fit_rejected,vs_rank_rejected,vs_validation_rejected,vs_applied,"
               "vs_before,vs_after,vs_delta_z,vs_rotation_rad,vs_range_bias,"
               "vs_anchor_age_frames,vs_anchor_footprint_motion_m,vs_anchor_replaced\n";
        timing << std::fixed << std::setprecision(9);

        lo::pipeline::LOConfig config;
        config.min_range = args.min_range;
        config.max_range = args.max_range;
        config.voxel_size = args.voxel_size;
        config.max_points_per_voxel = args.max_points_per_voxel;
        config.initial_threshold = args.initial_threshold;
        config.min_motion_th = args.min_motion_threshold;
        config.deskew = false;
        config.enable_f2f = args.variant == "full" || args.variant == "no_vertical";
        config.enable_vertical_constraint = args.variant == "full" || args.variant == "no_f2f";
        lo::pipeline::LO odometry(config);

        rosbag::Bag bag;
        bag.open(args.bag, rosbag::bagmode::Read);
        rosbag::View view(bag, rosbag::TopicQuery({args.topic}));
        if (view.size() == 0) throw std::runtime_error("selected PointCloud2 topic is empty");
        if (view.size() != args.expected_frames) {
            throw std::runtime_error("bag topic count changed after protocol discovery");
        }

        double previous_stamp = -1.0;
        std::size_t processed = 0;
        bool pacing_initialized = false;
        double first_bag_time = 0.0;
        std::chrono::steady_clock::time_point first_wall_time;

        for (const rosbag::MessageInstance& instance : view) {
            const double bag_time = instance.getTime().toSec();
            if (!pacing_initialized) {
                first_bag_time = bag_time;
                first_wall_time = std::chrono::steady_clock::now();
                pacing_initialized = true;
            }
            const double bag_elapsed_seconds = std::max(0.0, bag_time - first_bag_time);
            const auto deadline =
                first_wall_time +
                std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                    std::chrono::duration<double>(bag_elapsed_seconds));
            std::this_thread::sleep_until(deadline);
            const auto processing_start = std::chrono::steady_clock::now();
            const double deadline_lateness_ms = std::max(
                0.0,
                std::chrono::duration<double, std::milli>(processing_start - deadline).count());

            const auto message = instance.instantiate<sensor_msgs::PointCloud2>();
            if (!message) throw std::runtime_error("selected topic is not PointCloud2");
            const double stamp = message->header.stamp.toSec();
            if (!std::isfinite(stamp) ||
                (processed && stamp <= previous_stamp)) {
                throw std::runtime_error("PointCloud2 header timestamps are not strictly increasing");
            }
            const auto algorithm_start = std::chrono::steady_clock::now();
            std::vector<Eigen::Vector3d> points = ConvertPointCloud(*message);
            const auto registration_start = std::chrono::steady_clock::now();
            odometry.RegisterFrame(points, false);
            const auto processing_stop = std::chrono::steady_clock::now();

            const lo::pipeline::RegistrationMetrics& metrics =
                odometry.LastRegistrationMetrics();
            const double adapter_registration_ms =
                std::chrono::duration<double, std::milli>(
                    processing_stop - registration_start).count();
            const double end_to_end_ms =
                std::chrono::duration<double, std::milli>(
                    processing_stop - processing_start).count();
            if (!std::isfinite(metrics.registration_ms) || metrics.registration_ms < 0.0 ||
                !std::isfinite(metrics.f2f_ms) || metrics.f2f_ms < 0.0 ||
                !std::isfinite(metrics.f2m_ms) || metrics.f2m_ms < 0.0 ||
                !std::isfinite(adapter_registration_ms) || adapter_registration_ms < 0.0 ||
                !std::isfinite(end_to_end_ms) || end_to_end_ms < 0.0 ||
                !std::isfinite(deadline_lateness_ms) || deadline_lateness_ms < 0.0) {
                throw std::runtime_error("non-finite registration timing");
            }

            ++processed;
            previous_stamp = stamp;
            WritePose(trajectory, stamp, odometry.CurrentPose());
            trajectory.flush();
            struct rusage usage;
            getrusage(RUSAGE_SELF, &usage);
            std::ifstream statm("/proc/self/statm");
            unsigned long virtual_pages = 0, resident_pages = 0;
            statm >> virtual_pages >> resident_pages;
            const double rss_mib = resident_pages * double(sysconf(_SC_PAGESIZE)) / (1024 * 1024);
            const double wall_elapsed_seconds =
                std::chrono::duration<double>(processing_stop - first_wall_time).count();
            const double algorithm_ms = std::chrono::duration<double, std::milli>(processing_stop - algorithm_start).count();
            timing << processed << ',' << stamp << ',' << algorithm_ms << ',' << points.size() << ','
                   << metrics.registration_ms << ',' << metrics.f2f_ms << ','
                   << metrics.f2m_ms << ',' << end_to_end_ms
                   << ',' << adapter_registration_ms << ',' << config.enable_f2f
                   << ',' << config.enable_vertical_constraint << ',' << metrics.f2f_executed << ','
                   << deadline_lateness_ms << ',' << wall_elapsed_seconds << ','
                   << bag_elapsed_seconds << ',' << rss_mib << ',' << usage.ru_maxrss / 1024.0
                   << ',' << odometry.LocalVoxelCount() << ',' << odometry.LocalBucketCount()
                   << ',' << odometry.GlobalVoxelCount() << ',' << odometry.GlobalBucketCount();
            for (const auto& d : {metrics.ff_vertical, metrics.lm_vertical}) {
                timing << ',' << d.iterations << ',' << d.gates << ',' << d.zero_steps
                       << ',' << d.clipped_steps << ',' << d.full_fallbacks << ',' << d.stage_clipped;
            }
            for (const auto& d : {metrics.ff_vertical, metrics.lm_vertical})
                timing << ',' << d.correspondences << ',' << d.geometry_matches;
            const auto& vs=metrics.vertical_surface;
            timing << ',' << metrics.vertical_surface_ms << ',' << vs.source_patches
                   << ',' << vs.reference_patches << ',' << vs.matched_patches
                   << ',' << vs.fit_rejected << ',' << vs.rank_rejected
                   << ',' << vs.validation_rejected << ',' << vs.applied
                   << ',' << vs.before << ',' << vs.after << ',' << vs.delta_z
                   << ',' << vs.rotation_rad << ',' << vs.range_bias;
            timing << ',' << metrics.surface_anchor_age_frames
                   << ',' << metrics.surface_anchor_footprint_motion_m
                   << ',' << metrics.surface_anchor_replaced << '\n';
            timing.flush();
            if (processed == 1 || processed % 100 == 0) {
                std::cout << "Processed " << processed << '/' << args.expected_frames
                          << " scans, RSS=" << rss_mib << " MiB" << std::endl;
            }
            if (args.max_frames && processed >= args.max_frames) break;
        }
        bag.close();

        if (processed != (args.max_frames ? std::min(args.max_frames, args.expected_frames) : args.expected_frames)) {
            throw std::runtime_error("processed frame count differs from expected frame count");
        }
        trajectory.flush();
        if (!trajectory) throw std::runtime_error("failed while writing trajectory");

        std::cout << "Original-core reproduction completed " << processed
                  << " scans at fixed 1.0x recorded-time playback; native XYZ was unchanged\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
