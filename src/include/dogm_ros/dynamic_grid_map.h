#ifndef DYNAMIC_GRID_MAP_H
#define DYNAMIC_GRID_MAP_H

#include <vector>
#include <memory>
#include <string>
#include <random>
#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <nav_msgs/OccupancyGrid.h>
#include <visualization_msgs/MarkerArray.h>
#include <pcl/point_cloud.h>

#include "dogm_ros/structures.h"
#include "dogm_ros/particle_filter.h"
#include "dogm_ros/ego_calibration.h"
#include "dogm_ros/FilteredSigma.h"

// [Dual Radar] Radar Data Structure
struct RadarDataPacket {
    pcl::PointCloud<mmWaveCloudType>::ConstPtr cloud;
    double sensor_x;
    double sensor_y;
};

class DynamicGridMap {
public:
    DynamicGridMap(double grid_size, double resolution, int num_particles,
                   double process_noise_pos, double process_noise_vel,
                   int radar_buffer_size, int min_radar_points,
                   int radar_hint_search_radius,
                   bool use_fsd, int fsd_T_static, int fsd_T_free,
                   bool use_mc,
                   bool use_radar,
                   int lidar_hit_point,
                   // double lidar_noise_stddev, // Removed
                   // [Changed] Separated Thresholds
                   double particle_vector_vel_thresh, 
                   double particle_vector_ang_thresh,
                   double particle_static_vel_thresh,
                   double radar_static_vel_thresh,
                   bool cluster_mode);
    ~DynamicGridMap() = default;

    void generateMeasurementGrid(const sensor_msgs::LaserScan::ConstPtr& scan,
                                 const std::vector<RadarDataPacket>& radar_packets,
                                 const dogm_ros::FilteredSigma& sigmas);

    void updateOccupancy(double birth_prob);
    void shiftGrid(double dx, double dy);
    
    // ... (rest of the file is the same)
private:
    // ...
    bool use_mc_;
    bool use_radar_;
    int lidar_hit_point_;
    // double lidar_noise_stddev_; // Removed
    
    // [Changed] New Threshold Variables
    double particle_vector_vel_thresh_; 
    double particle_vector_ang_thresh_; 
    // ... (rest of the file is the same)
};

#endif // DYNAMIC_GRID_MAP_H