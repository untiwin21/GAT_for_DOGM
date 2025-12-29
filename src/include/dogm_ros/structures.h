#ifndef DOGM_ROS_STRUCTURES_H
#define DOGM_ROS_STRUCTURES_H

#include <cstdint>
#include <vector>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

/**
 * @brief Custom point type for mmWave Radar
 */
struct mmWaveCloudType
{
    PCL_ADD_POINT4D;
    union
    {
        struct
        {
            float intensity; 
            float velocity;  
        };
        float data_c[4];
    };
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

POINT_CLOUD_REGISTER_POINT_STRUCT(mmWaveCloudType,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (float, velocity, velocity)
)

/**
 * @brief Particle state representation
 */
struct Particle {
    double x{0.0}, y{0.0};
    double vx{0.0}, vy{0.0};
    double weight{0.0};
    int    grid_cell_idx{-1};
    int    age{0}; 
};

// [수정됨] RadarPoint를 GridCell 밖으로 꺼내서 전역으로 정의
/**
 * @brief Individual Radar point feature
 */
struct RadarPoint
{
    double radial_velocity; 
    double x;               
    double y;               
    int age;                
};

/**
 * @brief Refined radar hint for fusion
 */
struct RadarHint {
    double vr;          // Mean radial velocity
    double sensor_x;    // Sensor origin X for azimuth calculation
    double sensor_y;    // Sensor origin Y for azimuth calculation
    bool valid{false};

    // [Optimization] Pre-computed trigonometric values
    double obs_cos_theta{0.0};
    double obs_sin_theta{0.0};
    
    // Default constructor explicitly
    RadarHint() : vr(0.0), sensor_x(0.0), sensor_y(0.0), valid(false),
                  obs_cos_theta(0.0), obs_sin_theta(0.0) {}
};

/**
 * @brief Main Grid Cell structure
 */
struct GridCell {
    double m_occ{0.0};
    double m_free{0.0};
    double rho_b{0.0};
    double rho_p{0.0};

    double mean_vx{0.0}, mean_vy{0.0};
    double var_vx{0.0},  var_vy{0.0}, covar_vxy{0.0};

    bool   is_dynamic{false};
    double dynamic_score{0.0};
    double mahalanobis_dist{0.0}; 

    std::uint8_t dyn_streak{0};
    std::uint8_t stat_streak{0};
    std::uint8_t free_streak{0}; 

    // Collection of hints from different sensors
    std::vector<RadarHint> radar_hints;

    // Buffer for temporal aggregation per sensor
    // [확인] 이제 RadarPoint가 전역이므로 에러 없이 인식됨
    std::vector<RadarPoint> radar_buffer_1; // Right sensor
    std::vector<RadarPoint> radar_buffer_2; // Left sensor

    // Solver results
    bool   has_solved_velocity{false};
    double solved_vx{0.0};
    double solved_vy{0.0};

    // Default Constructor
    GridCell() = default;
};

/**
 * @brief Cell for sensor measurement evidence
 */
struct MeasurementCell {
    double m_occ_z{0.0};  
    double m_free_z{0.0}; 

    bool   has_lidar_model{false}; 
    double mean_x{0.0};
    double mean_y{0.0};
    double inv_cov_xx{0.0};
    double inv_cov_xy{0.0};
    double inv_cov_yy{0.0};
};

#endif // DOGM_ROS_STRUCTURES_H