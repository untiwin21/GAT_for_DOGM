#ifndef RADAR_VELOCITY_VIZ_H
#define RADAR_VELOCITY_VIZ_H

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
#include <string>
#include <deque>
#include <limits>
#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <geometry_msgs/TransformStamped.h>

#include "dogm_ros/structures.h"

// Class for the Radar Velocity Visualization Node
class RadarVizNode
{
public:
    // Constructor
    RadarVizNode();

private:
    // Callback functions for incoming radar point clouds (Dual Radar)
    void radar1Cb(const sensor_msgs::PointCloud2::ConstPtr& msg);
    void radar2Cb(const sensor_msgs::PointCloud2::ConstPtr& msg);

    // Common processing function to transform and buffer data
    void processRadar(const sensor_msgs::PointCloud2::ConstPtr& msg, const std::string& sensor_frame);

    // Loads parameters from the parameter server
    void loadParams();

    // Helper function to map velocity to color
    void interpolateColor(double velocity, double& r, double& g, double& b);

    // ROS NodeHandles
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;

    // Subscribers (Dual Radar)
    ros::Subscriber radar_sub_1_;
    ros::Subscriber radar_sub_2_;
    
    // Publisher
    ros::Publisher radar_viz_pub_;

    // TF Buffer & Listener for coordinate transformation
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    // Parameters
    bool use_radar_;
    std::string radar_topic_1_;       
    std::string radar_topic_2_;
    std::string radar_frame_1_;
    std::string radar_frame_2_;
    
    std::string radar_viz_topic_;     // Topic to publish visualization markers to
    std::string base_frame_;          // Target frame ID (e.g., base_link)
    double radar_viz_lifetime_;       // How long markers should persist in RViz
    double radar_viz_color_max_vel_;  // Speed for max color intensity
    int viz_buffer_size_;             // Number of scans to buffer

    // Buffer to store TRANSFORMED point clouds
    std::deque<pcl::PointCloud<mmWaveCloudType>::Ptr> cloud_buffer_;
};

#endif // RADAR_VELOCITY_VIZ_H