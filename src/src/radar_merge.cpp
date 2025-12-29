#include "dogm_ros/radar_merge.h"
// [핵심] PCL 변환 함수 사용을 위한 헤더 추가 (이게 없으면 에러 발생)
#include <pcl/common/transforms.h> 

RadarMerge::RadarMerge() : tf_listener_(tf_buffer_) {
}

void RadarMerge::process(const sensor_msgs::PointCloud2::ConstPtr& cloud1_msg,
                         const sensor_msgs::PointCloud2::ConstPtr& cloud2_msg,
                         const std::string& target_frame,
                         pcl::PointCloud<mmWaveCloudType>& merged_cloud) 
{
    merged_cloud.clear();

    // 데이터를 변환하고 합치는 람다 함수
    auto transformAndAdd = [&](const sensor_msgs::PointCloud2::ConstPtr& msg) {
        if (!msg) return;

        // 1. ROS 메시지를 PCL 포맷으로 변환
        pcl::PointCloud<mmWaveCloudType> pcl_temp;
        pcl::fromROSMsg(*msg, pcl_temp);

        if (pcl_temp.empty()) return;

        // 2. 좌표계가 다르면 TF를 이용해 변환
        if (msg->header.frame_id != target_frame) {
            try {
                // TF 조회 (최신 시간 기준)
                geometry_msgs::TransformStamped transform_stamped = 
                    tf_buffer_.lookupTransform(target_frame, msg->header.frame_id, 
                                               ros::Time(0), ros::Duration(0.1));

                // [수정 포인트] Geometry Transform -> Eigen Matrix 변환
                Eigen::Matrix4f mat;
                pcl_ros::transformAsMatrix(transform_stamped.transform, mat);

                pcl::PointCloud<mmWaveCloudType> pcl_transformed;
                
                // [수정 포인트] PCL 라이브러리의 원본 함수 사용 (링크 에러 해결)
                pcl::transformPointCloud(pcl_temp, pcl_transformed, mat);
                
                merged_cloud += pcl_transformed;

            } catch (tf2::TransformException &ex) {
                ROS_WARN_THROTTLE(1.0, "[RadarMerge] TF Error: %s", ex.what());
            }
        } else {
            // 좌표계가 이미 같으면 그냥 합침
            merged_cloud += pcl_temp;
        }
    };

    // 3. 두 센서 데이터를 각각 처리하여 merged_cloud에 누적
    transformAndAdd(cloud1_msg);
    transformAndAdd(cloud2_msg);

    // 4. 합쳐진 데이터의 헤더 설정
    merged_cloud.header.frame_id = target_frame;
    pcl_conversions::toPCL(ros::Time::now(), merged_cloud.header.stamp);
}