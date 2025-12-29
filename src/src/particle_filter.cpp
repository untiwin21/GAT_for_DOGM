#include "dogm_ros/particle_filter.h"
#include "dogm_ros/dynamic_grid_map.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <omp.h>

// [Robustness] Minimum probability to prevent particle death
static const double MIN_RADAR_PROB = 0.1;

ParticleFilter::ParticleFilter(int num_particles, double process_noise_pos, double process_noise_vel)
    : num_particles_(num_particles),
      process_noise_pos_(process_noise_pos),
      process_noise_vel_(process_noise_vel),
      random_generator_(std::mt19937(std::random_device()())),
      pos_noise_dist_(0.0, process_noise_pos_),
      vel_noise_dist_(0.0, process_noise_vel_)
{
    particles_.resize(num_particles_);
    double initial_weight = 1.0 / num_particles_;
    for (auto& p : particles_) { p.weight = initial_weight; }
}

void ParticleFilter::predict(double dt, double survival_prob,
                             double damping_thresh, double damping_factor,
                             double max_vel,
                             double d_ego_vx, double d_ego_vy) 
{
    const int GRACE_PERIOD = 3;
    #pragma omp parallel for
    for (int i = 0; i < particles_.size(); ++i) {
        auto& p = particles_[i];
        
        // Ego-motion compensation
        p.vx -= d_ego_vx; 
        p.vy -= d_ego_vy;

        double pos_noise_x, pos_noise_y, vel_noise_x, vel_noise_y;
        #pragma omp critical (random_gen) 
        {
            pos_noise_x = pos_noise_dist_(random_generator_);
            pos_noise_y = pos_noise_dist_(random_generator_);
            vel_noise_x = vel_noise_dist_(random_generator_);
            vel_noise_y = vel_noise_dist_(random_generator_);
        }
        
        // Motion model update
        p.x += p.vx * dt + pos_noise_x;
        p.y += p.vy * dt + pos_noise_y;
        p.vx += vel_noise_x; 
        p.vy += vel_noise_y;

        // Velocity clamping
        double speed = std::sqrt(p.vx * p.vx + p.vy * p.vy);
        if (speed > max_vel) {
            p.vx = (p.vx / speed) * max_vel;
            p.vy = (p.vy / speed) * max_vel;
        }

        // Damping for stationary particles
        if (p.age > GRACE_PERIOD) {
            if (speed < damping_thresh) {
                p.vx *= damping_factor;
                p.vy *= damping_factor;
            }
        }

        p.weight *= survival_prob;
        p.age++;
    }
}

void ParticleFilter::updateWeights(const std::vector<MeasurementCell>& measurement_grid,
                                   const std::vector<GridCell>& grid,
                                   const DynamicGridMap& grid_map,
                                   const dogm_ros::FilteredSigma& sigmas)
{
    double total_weight = 0.0;
    const double radar_variance = sigmas.radar_velocity_sigma * sigmas.radar_velocity_sigma;
    // Normalization factor for Gaussian probability
    const double radar_norm_factor = -0.5 / std::max(1e-9, radar_variance);

    const int search_radius = grid_map.getRadarHintSearchRadius();

    #pragma omp parallel for reduction(+:total_weight)
    for (int i = 0; i < particles_.size(); ++i)
    {
        auto& p = particles_[i];
        
        if (p.weight > 1e-9)
        {
            if (p.grid_cell_idx >= 0 && p.grid_cell_idx < static_cast<int>(measurement_grid.size()))
            {
                const auto& meas_cell = measurement_grid[p.grid_cell_idx];
                
                // 1. LiDAR Likelihood
                double lidar_likelihood = 1e-9;
                if (meas_cell.has_lidar_model) {
                    double dx = p.x - meas_cell.mean_x;
                    double dy = p.y - meas_cell.mean_y;
                    
                    double zx = dx * meas_cell.inv_cov_xx + dy * meas_cell.inv_cov_xy;
                    double zy = dx * meas_cell.inv_cov_xy + dy * meas_cell.inv_cov_yy;
                    double mahal_dist_sq = zx * dx + zy * dy;
                    
                    lidar_likelihood = std::exp(-0.5 * mahal_dist_sq);
                    // Squaring emphasizes the peak, making the filter more aggressive
                    // lidar_likelihood = lidar_likelihood * lidar_likelihood;
                }

                // 2. Radar Likelihood
                // Initialized to 1.0 (identity for multiplication).
                // If no valid radar hints are found, it remains 1.0 (neutral).
                double radar_likelihood = 1.0; 
                
                int p_gx, p_gy;
                grid_map.indexToGrid(p.grid_cell_idx, p_gx, p_gy);

                for (int dy = -search_radius; dy <= search_radius; ++dy) {
                    for (int dx = -search_radius; dx <= search_radius; ++dx) {
                        int nx = p_gx + dx;
                        int ny = p_gy + dy;
                        
                        if (!grid_map.isInside(nx, ny)) continue;
                        
                        const auto& neighbor_cell = grid[grid_map.gridToIndex(nx, ny)];
                        
                        if (!neighbor_cell.radar_hints.empty()) {
                            for (const auto& hint : neighbor_cell.radar_hints) {
                                // [Filtering] Ignore low radial velocity measurements.
                                // If abs(vr) < 0.1, it's likely lateral motion (Doppler ~ 0) or static clutter.
                                // We skip these to prevent them from killing dynamic particles.
                                if (std::abs(hint.vr) < 0.1) continue;

                                // B. Expected Radial Velocity Calculation
                                // Uses pre-computed trigonometric values (obs_cos_theta, obs_sin_theta)
                                // to avoid expensive 'atan2', 'cos', 'sin' calls inside the loop.
                                // Formula: vr_exp = vx * cos(theta) + vy * sin(theta)
                                double vr_expected = p.vx * hint.obs_cos_theta + p.vy * hint.obs_sin_theta;
                                
                                // C. Probability Calculation
                                double error = vr_expected - hint.vr;
                                double prob = std::exp(error * error * radar_norm_factor);
                                
                                // D. Update Likelihood (Product Rule)
                                // Use MIN_RADAR_PROB to prevent likelihood from becoming zero.
                                radar_likelihood *= std::max(prob, MIN_RADAR_PROB);
                            }
                        }
                    }
                }

                // 3. Final Weight Update
                // Combine LiDAR and Radar likelihoods.
                p.weight *= (lidar_likelihood * radar_likelihood);
            }
            else { 
                // Penalty for particles outside the valid grid area
                p.weight *= 0.01; 
            }
        }
        total_weight += p.weight;
    }

    // Normalization
    if (total_weight > 1e-9) {
        for (auto& p : particles_) p.weight /= total_weight;
    }
}


void ParticleFilter::sortParticlesByGridCell(const DynamicGridMap& grid_map)
{
    #pragma omp parallel for
    for (int i = 0; i < particles_.size(); ++i) {
        auto& p = particles_[i];
        int grid_x, grid_y;
        if (grid_map.worldToGrid(p.x, p.y, grid_x, grid_y)) {
            p.grid_cell_idx = grid_map.gridToIndex(grid_x, grid_y);
        } else {
            p.grid_cell_idx = -1;
        }
    }
    std::sort(particles_.begin(), particles_.end(),
              [](const Particle& a, const Particle& b) {
                  return a.grid_cell_idx < b.grid_cell_idx;
              });
}

void ParticleFilter::resample(const std::vector<Particle>& new_born_particles)
{
    std::vector<Particle> combined_pool;
    combined_pool.reserve(particles_.size() + new_born_particles.size());
    combined_pool.insert(combined_pool.end(), particles_.begin(), particles_.end());
    combined_pool.insert(combined_pool.end(), new_born_particles.begin(), new_born_particles.end());

    if (combined_pool.empty()) { particles_.clear(); return; }

    double total_weight = 0.0;
    for (const auto& p : combined_pool) total_weight += p.weight;

    if (total_weight < 1e-9) {
        for (auto& p : combined_pool) p.weight = 1.0 / combined_pool.size();
        total_weight = 1.0;
    } else {
        for (auto& p : combined_pool) p.weight /= total_weight;
    }

    std::vector<Particle> new_particle_set;
    new_particle_set.reserve(num_particles_);
    std::uniform_real_distribution<double> dist(0.0, 1.0 / num_particles_);
    double r = dist(random_generator_); 
    double c = combined_pool[0].weight;
    int i = 0;

    for (int m = 0; m < num_particles_; ++m) {
        double u = r + m * (1.0 / num_particles_);
        while (u > c) {
            i++;
            if (i >= static_cast<int>(combined_pool.size())) i = combined_pool.size() - 1;
            c += combined_pool[i].weight;
        }
        new_particle_set.push_back(combined_pool[i]);
    }
    particles_ = new_particle_set;
    
    if (!particles_.empty()) {
        double uniform_weight = 1.0 / particles_.size();
        for (auto& p : particles_) p.weight = uniform_weight;
    }
}

// ----------------------------------------------------------------
// [Implementation] Sector Gating Logic (Separated Mag & Angle)
// ----------------------------------------------------------------
bool ParticleFilter::checkSectorMatch(const Particle& winner, const Particle& candidate, 
                                      double vel_thresh, double ang_thresh_deg)
{
    // 1. Calculate Magnitudes (Speed)
    double winner_mag = std::sqrt(winner.vx * winner.vx + winner.vy * winner.vy);
    double cand_mag = std::sqrt(candidate.vx * candidate.vx + candidate.vy * candidate.vy);
    
    // 2. Calculate Angles (Direction) [Rad]
    double winner_ang = std::atan2(winner.vy, winner.vx);
    double cand_ang = std::atan2(candidate.vy, candidate.vx);

    // 3. Compute Differences
    double mag_diff = std::abs(winner_mag - cand_mag);
    double ang_diff_rad = angleDiff(winner_ang, cand_ang);
    double ang_diff_deg = std::abs(ang_diff_rad * 180.0 / M_PI);

    // 4. Check Thresholds (Sector Gating: AND condition)
    if (mag_diff <= vel_thresh && ang_diff_deg <= ang_thresh_deg) {
        return true;
    }
    return false;
}

double ParticleFilter::normalizeAngle(double angle) {
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}

double ParticleFilter::angleDiff(double a1, double a2) {
    double diff = a1 - a2;
    return normalizeAngle(diff);
}