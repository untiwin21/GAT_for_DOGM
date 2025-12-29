#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Odometry.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <string>
#include <memory>
#include <algorithm>
#include <cmath>
#include <vector>

#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <tf2_eigen/tf2_eigen.h>
#include <pcl/common/transforms.h>

#include "dogm_ros/dynamic_grid_map.h"
#include "dogm_ros/structures.h"
#include "dogm_ros/ego_calibration.h"
#include "dogm_ros/FilteredSigma.h"

class DogmNode
{
public:
    DogmNode() : nh_(), pnh_("~"), last_ego_vx_(0.0), last_ego_vy_(0.0), tf_listener_(tf_buffer_), has_sigmas_(false)
    {
        loadParams();
        
        // [Modified] Passing separated thresholds to constructor, removed lidar_noise_stddev
        grid_map_ = std::make_unique<DynamicGridMap>(
            grid_size_, grid_res_, num_particles_,
            process_noise_pos_, process_noise_vel_,
            radar_buffer_size_, min_radar_points_,
            radar_hint_search_radius_, 
            use_fsd_, fsd_T_static_, fsd_T_free_,
            use_mc_, use_radar_,
            lidar_hit_point_, 
            particle_vector_vel_thresh_, // [New]
            particle_vector_ang_thresh_, // [New]
            particle_static_vel_thresh_, radar_static_vel_thresh_,
            cluster_mode_ 
        );
        
        grid_pub_   = nh_.advertise<nav_msgs::OccupancyGrid>(grid_topic_, 1);
        marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(marker_topic_, 1);
        scan_sub_ = nh_.subscribe<sensor_msgs::LaserScan>(lidar_topic_, 1, &DogmNode::scanCb, this);
        sigma_sub_ = nh_.subscribe<dogm_ros::FilteredSigma>("/filtered_sigma", 1, &DogmNode::sigmaCb, this);

        if (use_radar_) {
            radar_sub_1_ = nh_.subscribe<sensor_msgs::PointCloud2>(radar_topic_1_, 1, &DogmNode::radar1Cb, this);
            radar_sub_2_ = nh_.subscribe<sensor_msgs::PointCloud2>(radar_topic_2_, 1, &DogmNode::radar2Cb, this);
            ROS_INFO("Dual Radar fusion is ENABLED.");
        }
        if (use_ego_comp_) {
            odom_sub_ = nh_.subscribe<nav_msgs::Odometry>(odom_topic_, 1, &DogmNode::odomCb, this);
        }
        if (show_ego_velocity_arrow_) {
            ego_vel_pub_ = nh_.advertise<visualization_msgs::Marker>(ego_vel_marker_topic_, 1);
        }

        // Default sigma values
        latest_sigmas_.lidar_position_sigma = 0.1; 
        latest_sigmas_.radar_velocity_sigma = 0.25;

        const double hz = (filter_hz_ > 0.5 ? filter_hz_ : 10.0);
        timer_ = nh_.createTimer(ros::Duration(1.0 / hz), &DogmNode::updateLoop, this);
    }

private:
    void scanCb(const sensor_msgs::LaserScan::ConstPtr& msg) {
        last_scan_ = msg;
        has_scan_  = true;
    }

    void odomCb(const nav_msgs::Odometry::ConstPtr& msg) {
        if (!msg) return;
        ego_calib_.update(msg->twist.twist.linear.x, msg->twist.twist.linear.y);
    }

    void sigmaCb(const dogm_ros::FilteredSigma::ConstPtr& msg) {
        latest_sigmas_ = *msg;
        has_sigmas_ = true;
    }

    void radar1Cb(const sensor_msgs::PointCloud2::ConstPtr& msg) {
        if (!use_radar_) return;
        last_radar_cloud_1_.reset(new pcl::PointCloud<mmWaveCloudType>());
        pcl::fromROSMsg(*msg, *last_radar_cloud_1_);
        has_radar_1_ = !last_radar_cloud_1_->points.empty();
    }

    void radar2Cb(const sensor_msgs::PointCloud2::ConstPtr& msg) {
        if (!use_radar_) return;
        last_radar_cloud_2_.reset(new pcl::PointCloud<mmWaveCloudType>());
        pcl::fromROSMsg(*msg, *last_radar_cloud_2_);
        has_radar_2_ = !last_radar_cloud_2_->points.empty();
    }

    bool processRadarData(const pcl::PointCloud<mmWaveCloudType>::Ptr& cloud, 
                          const std::string& target_frame, 
                          const std::string& sensor_frame,
                          RadarDataPacket& packet) 
    {
        if (!cloud || cloud->points.empty()) return false;

        try {
            geometry_msgs::TransformStamped transform = tf_buffer_.lookupTransform(target_frame, sensor_frame, ros::Time(0));
            packet.sensor_x = transform.transform.translation.x;
            packet.sensor_y = transform.transform.translation.y;

             Eigen::Affine3d tf_eigen;
             tf_eigen = tf2::transformToEigen(transform);
             
             pcl::PointCloud<mmWaveCloudType>::Ptr transformed_cloud(new pcl::PointCloud<mmWaveCloudType>());
             pcl::transformPointCloud(*cloud, *transformed_cloud, tf_eigen);
             packet.cloud = transformed_cloud;
             return true;

        } catch (tf2::TransformException &ex) {
            ROS_WARN_THROTTLE(1.0, "Radar TF Lookup Failed: %s", ex.what());
            return false;
        }
    }

    void updateLoop(const ros::TimerEvent&) {
        if (!has_scan_ || !has_sigmas_) {
            ROS_INFO_THROTTLE(1.0, "Waiting for sensor data and learned sigmas...");
            return;
        }
        ros::Time now = ros::Time::now();
        double dt = (now - last_update_time_).toSec();
        if (dt <= 0.0) return;
        last_update_time_ = now;

        double dx, dy; int sx, sy;
        ego_calib_.getGridShift(dt, grid_res_, sx, sy, dx, dy);
        grid_map_->shiftGrid(dx, dy);

        std::vector<RadarDataPacket> radar_packets;
        if (use_radar_) {
            RadarDataPacket p1, p2;
            if (has_radar_1_ && processRadarData(last_radar_cloud_1_, base_frame_, radar_frame_1_, p1)) {
                radar_packets.push_back(p1);
            }
            if (has_radar_2_ && processRadarData(last_radar_cloud_2_, base_frame_, radar_frame_2_, p2)) {
                radar_packets.push_back(p2);
            }
        }
        
        grid_map_->generateMeasurementGrid(last_scan_, radar_packets, latest_sigmas_);
        
        has_radar_1_ = false; 
        has_radar_2_ = false;

        double curr_vx = ego_calib_.getVx();
        double curr_vy = ego_calib_.getVy();
        double d_vx = curr_vx - last_ego_vx_;
        double d_vy = curr_vy - last_ego_vy_;
        last_ego_vx_ = curr_vx;
        last_ego_vy_ = curr_vy;

        auto& pf = grid_map_->getParticleFilter();
        pf.predict(dt, persistence_prob_, velocity_damping_threshold_, 
                   velocity_damping_factor_, max_velocity_, d_vx, d_vy);

        pf.sortParticlesByGridCell(*grid_map_);
        pf.updateWeights(grid_map_->getMeasurementGrid(), grid_map_->getGrid(), *grid_map_, latest_sigmas_);
        
        grid_map_->updateOccupancy(birth_prob_);
        
        grid_map_->calculateVelocityStatistics(max_vel_for_scaling_, use_ego_comp_, ego_calib_);
        
        auto new_borns = grid_map_->generateNewParticles(
            newborn_vel_stddev_, min_dynamic_birth_ratio_, max_dynamic_birth_ratio_,
            max_radar_speed_for_scaling_, dynamic_newborn_vel_stddev_, ego_calib_
        );
        
        pf.resample(new_borns);

        nav_msgs::OccupancyGrid grid_msg;
        grid_map_->toOccupancyGridMsg(grid_msg, base_frame_);
        grid_pub_.publish(grid_msg);

        visualization_msgs::MarkerArray marker_msg;
        grid_map_->toMarkerArrayMsg(marker_msg, base_frame_, show_velocity_arrows_, ego_calib_);
        marker_pub_.publish(marker_msg);

        has_scan_ = false;
    }

    void loadParams() {
        pnh_.param("use_radar", use_radar_, true);
        pnh_.param("lidar_topic", lidar_topic_, std::string("/scan"));
        pnh_.param("radar_topic_1", radar_topic_1_, std::string("/ti_mmwave/radar_scan_pcl_0"));
        pnh_.param("radar_topic_2", radar_topic_2_, std::string("/ti_mmwave/radar_scan_pcl_1"));
        pnh_.param("radar_frame_1", radar_frame_1_, std::string("radar_1"));
        pnh_.param("radar_frame_2", radar_frame_2_, std::string("radar_2"));
        
        pnh_.param("grid_topic", grid_topic_, std::string("/dogm/grid"));
        pnh_.param("marker_topic", marker_topic_, std::string("/dogm/markers"));
        pnh_.param("base_frame", base_frame_, std::string("base_link"));
        pnh_.param("grid_size", grid_size_, 3.0);
        pnh_.param("grid_resolution", grid_res_, 0.15);
        pnh_.param("filter_update_rate", filter_hz_, 10.0);
        pnh_.param("num_particles", num_particles_, 20000);
        pnh_.param("process_noise_pos", process_noise_pos_, 0.03);
        pnh_.param("process_noise_vel", process_noise_vel_, 0.1);
        pnh_.param("max_velocity", max_velocity_, 1.5);
        pnh_.param("persistence_prob", persistence_prob_, 0.99);
        pnh_.param("birth_prob", birth_prob_, 0.02);
        pnh_.param("newborn_vel_stddev", newborn_vel_stddev_, 0.3);
        pnh_.param("dynamic_newborn_vel_stddev", dynamic_newborn_vel_stddev_, 1.5);
        pnh_.param("velocity_damping_threshold", velocity_damping_threshold_, 0.6);
        pnh_.param("velocity_damping_factor", velocity_damping_factor_, 0.2);
        pnh_.param("particle_static_velocity_threshold", particle_static_vel_thresh_, 0.3);
        pnh_.param("radar_static_velocity_threshold", radar_static_vel_thresh_, 0.5);
        
        // [Modified] New Thresholds for Sector Gating
        pnh_.param("particle_vector_Velthresh", particle_vector_vel_thresh_, 0.3); 
        pnh_.param("particle_vector_Angthresh", particle_vector_ang_thresh_, 30.0);

        pnh_.param("radar_buffer_size", radar_buffer_size_, 5);
        pnh_.param("min_radar_points", min_radar_points_, 2);
        pnh_.param("radar_hint_search_radius", radar_hint_search_radius_, 2); 
        pnh_.param("min_dynamic_birth_ratio", min_dynamic_birth_ratio_, 0.1);
        pnh_.param("max_dynamic_birth_ratio", max_dynamic_birth_ratio_, 0.9);
        pnh_.param("max_radar_speed_for_scaling", max_radar_speed_for_scaling_, 1.0);
        pnh_.param("use_false_static_detection", use_fsd_, true);
        pnh_.param("fsd_T_static", fsd_T_static_, 4);
        pnh_.param("fsd_T_free", fsd_T_free_, 10);
        pnh_.param("use_measurement_correction", use_mc_, true);
        pnh_.param("max_velocity_for_scaling", max_vel_for_scaling_, 2.0);
        pnh_.param("show_velocity_arrows", show_velocity_arrows_, true);
        pnh_.param("use_ego_comp", use_ego_comp_, true);
        pnh_.param("odom_topic", odom_topic_, std::string("/odom"));
        pnh_.param("show_ego_velocity_arrow", show_ego_velocity_arrow_, true);
        pnh_.param("ego_vel_marker_topic", ego_vel_marker_topic_, std::string("/dogm/ego_velocity_arrow"));
        pnh_.param("lidar_hit_point", lidar_hit_point_, 5);

        pnh_.param("cluster_mode", cluster_mode_, false);
    }

    ros::NodeHandle nh_, pnh_;
    ros::Subscriber scan_sub_, odom_sub_;
    ros::Subscriber radar_sub_1_, radar_sub_2_;
    ros::Subscriber sigma_sub_;
    ros::Publisher  grid_pub_, marker_pub_, ego_vel_pub_;
    ros::Timer      timer_;
    std::unique_ptr<DynamicGridMap> grid_map_;
    EgoCalibration ego_calib_;
    
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    
    double last_ego_vx_;
    double last_ego_vy_;

    // Parameters
    bool use_radar_;
    std::string lidar_topic_, grid_topic_, marker_topic_, base_frame_, odom_topic_;
    std::string radar_topic_1_, radar_topic_2_;
    std::string radar_frame_1_, radar_frame_2_;

    double grid_size_, grid_res_, filter_hz_, persistence_prob_, birth_prob_;
    int num_particles_;
    double process_noise_pos_, process_noise_vel_, max_velocity_, newborn_vel_stddev_;
    double min_dynamic_birth_ratio_, max_dynamic_birth_ratio_;
    double max_radar_speed_for_scaling_;
    double dynamic_newborn_vel_stddev_;
    double velocity_damping_threshold_, velocity_damping_factor_;
    double max_vel_for_scaling_;
    
    // [Modified] Replaced old thresh with new ones
    double particle_vector_vel_thresh_;
    double particle_vector_ang_thresh_;

    bool show_velocity_arrows_, use_ego_comp_;
    int radar_buffer_size_, min_radar_points_, radar_hint_search_radius_;
    bool use_fsd_, use_mc_;
    int fsd_T_static_, fsd_T_free_;
    std::string ego_vel_marker_topic_;
    bool show_ego_velocity_arrow_;
    int lidar_hit_point_;
    double particle_static_vel_thresh_;
    double radar_static_vel_thresh_;
    
    bool cluster_mode_;

    sensor_msgs::LaserScan::ConstPtr last_scan_;
    pcl::PointCloud<mmWaveCloudType>::Ptr last_radar_cloud_1_;
    pcl::PointCloud<mmWaveCloudType>::Ptr last_radar_cloud_2_;
    dogm_ros::FilteredSigma latest_sigmas_;
    bool has_scan_{false}, has_radar_1_{false}, has_radar_2_{false}, has_sigmas_{false};
    ros::Time last_update_time_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "dogm_node");
    DogmNode node;
    ros::spin();
    return 0;
}