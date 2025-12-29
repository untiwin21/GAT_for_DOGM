#include "dogm_ros/radar_velocity_viz.h"
#include <pcl/common/transforms.h> // [수정] PCL 원본 변환 함수 헤더 추가

// Function to interpolate color based on velocity magnitude using a rainbow spectrum
// Violet (low speed) -> Blue -> Green -> Yellow -> Orange -> Red (high speed)
void RadarVizNode::interpolateColor(double velocity, double& r, double& g, double& b)
{
    // Use absolute velocity (speed) for color mapping
    velocity = std::abs(velocity);

    const double min_vel = 0.0;
    const double max_vel = radar_viz_color_max_vel_;
    velocity = std::max(min_vel, std::min(velocity, max_vel)); 

    // Normalize velocity to 0.0 - 1.0 within the range
    double normalized_vel = (max_vel - min_vel > 1e-3) ? ((velocity - min_vel) / (max_vel - min_vel)) : 0.0;

    // Map normalized velocity (0-1) to HSV color space (Hue: 240 (blue/violet) to 0 (red))
    double hue = 240.0 * (1.0 - normalized_vel);
    double saturation = 1.0; 
    double value = 1.0;      

    // Convert HSV to RGB
    int i = static_cast<int>(hue / 60.0) % 6;
    double f = (hue / 60.0) - i;
    double p = value * (1.0 - saturation);
    double q = value * (1.0 - f * saturation);
    double t = value * (1.0 - (1.0 - f) * saturation);

    switch(i) {
        case 0: r = value; g = t; b = p; break; 
        case 1: r = q; g = value; b = p; break; 
        case 2: r = p; g = value; b = t; break; 
        case 3: r = p; g = q; b = value; break; 
        case 4: r = t; g = p; b = value; break; 
        case 5: r = value; g = p; b = q; break; 
        default: r = g = b = 1.0; break; 
    }
}


// Constructor
RadarVizNode::RadarVizNode() : nh_(), pnh_("~"), tf_listener_(tf_buffer_)
{
    loadParams(); 

    radar_viz_pub_ = nh_.advertise<visualization_msgs::Marker>(radar_viz_topic_, 1);

    if (use_radar_) {
        // Subscribe to both radar topics
        radar_sub_1_ = nh_.subscribe<sensor_msgs::PointCloud2>(radar_topic_1_, 1, &RadarVizNode::radar1Cb, this);
        radar_sub_2_ = nh_.subscribe<sensor_msgs::PointCloud2>(radar_topic_2_, 1, &RadarVizNode::radar2Cb, this);

        ROS_INFO("Radar Velocity Visualization Node started.");
        ROS_INFO("Subscribing to Radar 1: %s (Frame: %s)", radar_topic_1_.c_str(), radar_frame_1_.c_str());
        ROS_INFO("Subscribing to Radar 2: %s (Frame: %s)", radar_topic_2_.c_str(), radar_frame_2_.c_str());
    } else {
        ROS_WARN("Radar visualization is disabled by param 'use_radar'.");
    }
}

// Loads parameters from the parameter server
void RadarVizNode::loadParams()
{
    pnh_.param("use_radar", use_radar_, true);
    
    // Updated params to match dogm_node naming convention
    pnh_.param("radar_topic_1", radar_topic_1_, std::string("/ti_mmwave/radar_scan_pcl_0"));
    pnh_.param("radar_topic_2", radar_topic_2_, std::string("/ti_mmwave/radar_scan_pcl_1"));
    pnh_.param("radar_frame_1", radar_frame_1_, std::string("radar_1"));
    pnh_.param("radar_frame_2", radar_frame_2_, std::string("radar_2"));

    pnh_.param("radar_viz_topic", radar_viz_topic_, std::string("/dogm/radar_velocity_viz")); 
    pnh_.param("base_frame", base_frame_, std::string("base_link")); 
    pnh_.param("radar_viz_lifetime", radar_viz_lifetime_, 0.2); // Short lifetime for dynamic updates
    pnh_.param("radar_viz_color_max_vel", radar_viz_color_max_vel_, 1.5); 
    pnh_.param("viz_buffer_size", viz_buffer_size_, 5); // Increased buffer slightly for dual stream
}

// Callback for Radar 1
void RadarVizNode::radar1Cb(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
    processRadar(msg, radar_frame_1_);
}

// Callback for Radar 2
void RadarVizNode::radar2Cb(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
    processRadar(msg, radar_frame_2_);
}

// Common processing function
void RadarVizNode::processRadar(const sensor_msgs::PointCloud2::ConstPtr& msg, const std::string& sensor_frame)
{
    if (msg->data.empty()) return;

    pcl::PointCloud<mmWaveCloudType>::Ptr cloud_transformed(new pcl::PointCloud<mmWaveCloudType>());
    pcl::PointCloud<mmWaveCloudType>::Ptr cloud_raw(new pcl::PointCloud<mmWaveCloudType>());
    
    pcl::fromROSMsg(*msg, *cloud_raw);

    // --- TF Transformation ---
    try {
        // Lookup transform from sensor_frame to base_frame
        geometry_msgs::TransformStamped transform_stamped;
        // Use a timeout to ensure TF is available
        transform_stamped = tf_buffer_.lookupTransform(base_frame_, sensor_frame, ros::Time(0), ros::Duration(0.1));

        // [수정] pcl_ros::transformPointCloud 대신 Eigen Matrix 변환 후 PCL Native 함수 사용
        // 이유: 커스텀 포인트 타입(mmWaveCloudType)에 대한 pcl_ros 템플릿 인스턴스가 라이브러리에 없기 때문
        Eigen::Matrix4f mat;
        pcl_ros::transformAsMatrix(transform_stamped.transform, mat);

        pcl::transformPointCloud(*cloud_raw, *cloud_transformed, mat);
    }
    catch (tf2::TransformException &ex) {
        ROS_WARN_THROTTLE(1.0, "[RadarViz] TF Exception: %s. Skipping this scan.", ex.what());
        return;
    }

    // --- Buffer Management ---
    // Store the TRANSFORMED cloud
    cloud_buffer_.push_back(cloud_transformed);

    // Remove old scans
    while (cloud_buffer_.size() > static_cast<size_t>(viz_buffer_size_))
    {
        cloud_buffer_.pop_front();
    }

    // --- Visualization ---
    if (cloud_buffer_.empty()) return;

    visualization_msgs::Marker points_viz;
    points_viz.header.stamp = ros::Time::now(); // Use current time for viz
    points_viz.header.frame_id = base_frame_;   // Important: Markers are now in base_frame
    points_viz.ns = "radar_velocity_status";
    points_viz.id = 0;
    points_viz.type = visualization_msgs::Marker::POINTS;
    points_viz.action = visualization_msgs::Marker::ADD;
    points_viz.pose.orientation.w = 1.0; 
    points_viz.scale.x = 0.04; // Slightly larger points for visibility
    points_viz.scale.y = 0.04;
    points_viz.lifetime = ros::Duration(radar_viz_lifetime_); 

    for (const auto& cloud : cloud_buffer_)
    {
        for (const auto& pt : cloud->points) {
            geometry_msgs::Point p;
            p.x = pt.x; 
            p.y = pt.y; 
            p.z = pt.z; // Use transformed Z
            points_viz.points.push_back(p);

            // Add color based on velocity
            std_msgs::ColorRGBA color;
            color.a = 1.0;
            double r, g, b;
            interpolateColor(pt.velocity, r, g, b); 
            color.r = r; color.g = g; color.b = b;
            points_viz.colors.push_back(color);
        }
    }
    
    radar_viz_pub_.publish(points_viz);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "radar_velocity_viz_node"); 
    RadarVizNode node;                                 
    ros::spin();                                       
    return 0;
}