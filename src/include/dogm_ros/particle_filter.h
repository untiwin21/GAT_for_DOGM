#ifndef PARTICLE_FILTER_H
#define PARTICLE_FILTER_H

#include <vector>
#include <random>
#include <cmath>
#include "structures.h"
#include "dogm_ros/FilteredSigma.h"

// Ensure M_PI is defined
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class DynamicGridMap; // Forward declaration

class ParticleFilter {
public:
    ParticleFilter(int num_particles, double process_noise_pos, double process_noise_vel);
    ~ParticleFilter() = default;

    void predict(double dt, double survival_prob,
                 double damping_thresh, double damping_factor,
                 double max_vel,
                 double d_ego_vx = 0.0, double d_ego_vy = 0.0);

    void updateWeights(const std::vector<MeasurementCell>& measurement_grid,
                       const std::vector<GridCell>& grid,
                       const DynamicGridMap& grid_map,
                       const dogm_ros::FilteredSigma& sigmas);

    void sortParticlesByGridCell(const DynamicGridMap& grid_map);
    
    void resample(const std::vector<Particle>& new_born_particles);

    /**
     * @brief Checks if a candidate particle matches the winner within the sector gate.
     * * @param winner The reference particle (highest weight).
     * @param candidate The particle to check.
     * @param vel_thresh Allowed velocity magnitude difference [m/s].
     * @param ang_thresh_deg Allowed angle difference [degrees].
     * @return true If both magnitude and angle conditions are met.
     */
    bool checkSectorMatch(const Particle& winner, const Particle& candidate, 
                          double vel_thresh, double ang_thresh_deg);

    std::vector<Particle>& getParticles() { return particles_; }
    const std::vector<Particle>& getParticles() const { return particles_; }

private:
    int num_particles_;
    double process_noise_pos_;
    double process_noise_vel_;
    std::vector<Particle> particles_;
    
    std::mt19937 random_generator_; 
    
    std::normal_distribution<double> pos_noise_dist_;
    std::normal_distribution<double> vel_noise_dist_;

    // Helper functions
    double normalizeAngle(double angle);
    double angleDiff(double a1, double a2);
};

#endif // PARTICLE_FILTER_H