#ifndef DOGM_ROS_RADAR_MERGE_H
#define DOGM_ROS_RADAR_MERGE_H

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <string>

// DOGM 데이터 구조체 포함
#include "dogm_ros/structures.h"

class RadarMerge {
public:
    RadarMerge();
    ~RadarMerge() = default;

    /**
     * @brief 두 개의 Radar ROS 메시지를 받아서 base_frame 기준으로 변환 후 병합하여 반환
     * * @param cloud1_msg 첫 번째 레이더 메시지 (ROS)
     * @param cloud2_msg 두 번째 레이더 메시지 (ROS)
     * @param target_frame 변환할 기준 좌표계 (예: base_link)
     * @param merged_cloud [Output] 병합된 PCL 클라우드
     */
    void process(const sensor_msgs::PointCloud2::ConstPtr& cloud1_msg,
                 const sensor_msgs::PointCloud2::ConstPtr& cloud2_msg,
                 const std::string& target_frame,
                 pcl::PointCloud<mmWaveCloudType>& merged_cloud);

private:
    // TF Buffer & Listener
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
};

#endif // DOGM_ROS_RADAR_MERGE_H