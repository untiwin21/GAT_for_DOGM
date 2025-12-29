#include "dogm_ros/dynamic_grid_map.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include <random>
#include <queue> 
#include <map>

DynamicGridMap::DynamicGridMap(double grid_size, double resolution, int num_particles,
                               double process_noise_pos, double process_noise_vel,
                               int radar_buffer_size, int min_radar_points,
                               int radar_hint_search_radius,
                               bool use_fsd, int fsd_T_static, int fsd_T_free,
                               bool use_mc,
                               bool use_radar,
                               int lidar_hit_point,
                               // [Params]
                               double particle_vector_vel_thresh,
                               double particle_vector_ang_thresh,
                               double particle_static_vel_thresh,
                               double radar_static_vel_thresh,
                               bool cluster_mode)
    : grid_size_(grid_size),
      resolution_(resolution),
      radar_buffer_size_(radar_buffer_size),
      min_radar_points_(min_radar_points),
      radar_hint_search_radius_(radar_hint_search_radius),
      use_fsd_(use_fsd),
      fsd_T_static_(fsd_T_static),
      fsd_T_free_(fsd_T_free),
      use_mc_(use_mc),
      use_radar_(use_radar),
      lidar_hit_point_(lidar_hit_point),
      // [Thresholds]
      particle_vector_vel_thresh_(particle_vector_vel_thresh),
      particle_vector_ang_thresh_(particle_vector_ang_thresh),
      particle_static_vel_thresh_(particle_static_vel_thresh),
      radar_static_vel_thresh_(radar_static_vel_thresh),
      cluster_mode_(cluster_mode),
      random_generator_(std::mt19937(std::random_device()()))
{
    grid_width_  = static_cast<int>(std::round(grid_size_ / resolution_));
    grid_height_ = grid_width_;
    origin_x_ = 0.0;
    origin_y_ = -grid_size_ / 2.0;
    grid_.assign(grid_width_ * grid_height_, GridCell{});
    measurement_grid_.assign(grid_width_ * grid_height_, MeasurementCell{});
    particle_filter_ = std::make_unique<ParticleFilter>(
        num_particles, process_noise_pos, process_noise_vel
    );
}

// ... (Helper functions - 기존과 동일) ...
bool DynamicGridMap::isInside(int gx, int gy) const { return (gx >= 0 && gx < grid_width_ && gy >= 0 && gy < grid_height_); }
int DynamicGridMap::gridToIndex(int gx, int gy) const { return gy * grid_width_ + gx; }
void DynamicGridMap::indexToGrid(int idx, int& gx, int& gy) const { gy = idx / grid_width_; gx = idx % grid_width_; }
bool DynamicGridMap::worldToGrid(double wx, double wy, int& gx, int& gy) const {
    gx = static_cast<int>(std::floor((wx - origin_x_) / resolution_));
    gy = static_cast<int>(std::floor((wy - origin_y_) / resolution_));
    return isInside(gx, gy);
}
void DynamicGridMap::gridToWorld(int gx, int gy, double& wx, double& wy) const {
    wx = origin_x_ + (static_cast<double>(gx) + 0.5) * resolution_;
    wy = origin_y_ + (static_cast<double>(gy) + 0.5) * resolution_;
}

void DynamicGridMap::generateMeasurementGrid(const sensor_msgs::LaserScan::ConstPtr& scan, 
                                             const std::vector<RadarDataPacket>& radar_packets,
                                             const dogm_ros::FilteredSigma& sigmas) {
    for (auto& m : measurement_grid_) { m.m_free_z = 0.0; m.m_occ_z = 0.0; m.has_lidar_model = false; }
    for (auto& c : grid_) { c.radar_hints.clear(); }

    for (auto& c : grid_) {
        for (auto& rp : c.radar_buffer_1) { rp.age++; }
        c.radar_buffer_1.erase(std::remove_if(c.radar_buffer_1.begin(), c.radar_buffer_1.end(),
                           [this](const RadarPoint& rp) { return rp.age > this->radar_buffer_size_; }), c.radar_buffer_1.end());
        for (auto& rp : c.radar_buffer_2) { rp.age++; }
        c.radar_buffer_2.erase(std::remove_if(c.radar_buffer_2.begin(), c.radar_buffer_2.end(),
                           [this](const RadarPoint& rp) { return rp.age > this->radar_buffer_size_; }), c.radar_buffer_2.end());
    }

    if (use_radar_) {
        for (size_t i = 0; i < radar_packets.size(); ++i) {
            const auto& packet = radar_packets[i];
            if (!packet.cloud) continue;
            for (const auto& pt : packet.cloud->points) {
                int gx, gy;
                if (worldToGrid(pt.x, pt.y, gx, gy)) {
                    int idx = gridToIndex(gx, gy);
                    RadarPoint new_rp; 
                    new_rp.radial_velocity = pt.velocity; 
                    new_rp.x = pt.x; new_rp.y = pt.y; new_rp.age = 0;
                    if (i == 0) grid_[idx].radar_buffer_1.push_back(new_rp);
                    else if (i == 1) grid_[idx].radar_buffer_2.push_back(new_rp);
                }
            }
        }
    }

    auto processBuffer = [&](const std::vector<RadarPoint>& buffer, double sx, double sy, int min_pts) -> RadarHint 
    {
        RadarHint hint;
        if (buffer.size() < static_cast<size_t>(min_pts)) return hint;

        double sum_vr = 0.0, sum_x = 0.0, sum_y = 0.0;
        for (const auto& rp : buffer) { 
            sum_vr += rp.radial_velocity; 
            sum_x += rp.x; 
            sum_y += rp.y; 
        }

        // 1. 관측 데이터들의 평균(무게중심) 계산
        hint.vr = sum_vr / buffer.size(); 
        double mean_x = sum_x / buffer.size(); 
        double mean_y = sum_y / buffer.size();
        
        hint.sensor_x = sx; 
        hint.sensor_y = sy; 
        hint.valid = true;

        // 2. [핵심 수정] 관측 무게중심을 기준으로 '각도'를 확정 (파티클 위치 무관)
        double dx = mean_x - sx;
        double dy = mean_y - sy;
        double angle = std::atan2(dy, dx); // 센서 -> 관측점 절대 각도

        // 3. 파티클 필터용 삼각함수 값 미리 계산 (최적화 & 안정화)
        hint.obs_cos_theta = std::cos(angle);
        hint.obs_sin_theta = std::sin(angle);

        return hint;
    };

    for (int idx = 0; idx < grid_.size(); ++idx) {
        auto& c = grid_[idx];
        if (radar_packets.size() > 0) {
            RadarHint h1 = processBuffer(c.radar_buffer_1, radar_packets[0].sensor_x, radar_packets[0].sensor_y, min_radar_points_);
            if (h1.valid) c.radar_hints.push_back(h1);
        }
        if (radar_packets.size() > 1) {
            RadarHint h2 = processBuffer(c.radar_buffer_2, radar_packets[1].sensor_x, radar_packets[1].sensor_y, min_radar_points_);
            if (h2.valid) c.radar_hints.push_back(h2);
        }
    }
    
    // Lidar Processing (Simplified for brevity as it's unchanged)
    std::vector<std::vector<std::pair<double, double>>> cell_hits(grid_width_ * grid_height_);
    if (!scan) return;
    const double angle_min = static_cast<double>(scan->angle_min);
    const double angle_inc = static_cast<double>(scan->angle_increment);
    const double range_max = static_cast<double>(scan->range_max);
    
    for (size_t i = 0; i < scan->ranges.size(); ++i) {
        double r  = static_cast<double>(scan->ranges[i]);
        if (!std::isfinite(r) || r<0.01) { r = range_max; }
        const double th = angle_min + angle_inc * static_cast<double>(i);
        const double step  = resolution_ * 0.9;
        const double limit = std::min(r, range_max);
        const double P_FREE = 0.4; 

        for (double rr = 0.0; rr < limit; rr += step) {
            const double wx = rr * std::cos(th); const double wy = rr * std::sin(th);
            int gx, gy;
            if (!worldToGrid(wx, wy, gx, gy)) break;
            measurement_grid_[gridToIndex(gx, gy)].m_free_z = P_FREE;
        }
        if (r < range_max) {
            const double wx = r * std::cos(th); const double wy = r * std::sin(th);
            int gx, gy;
            if (worldToGrid(wx, wy, gx, gy)) cell_hits[gridToIndex(gx, gy)].push_back({wx, wy});
        }
    }
    
    const double lidar_variance_reg = sigmas.lidar_position_sigma * sigmas.lidar_position_sigma;
    for (int idx = 0; idx < static_cast<int>(grid_.size()); ++idx) {
        auto& meas_cell = measurement_grid_[idx];
        const auto& hits = cell_hits[idx];
        if (hits.size() < static_cast<size_t>(lidar_hit_point_)) continue; 
        meas_cell.m_occ_z = 0.7; meas_cell.m_free_z = 0.0; 
        double sum_x = 0.0, sum_y = 0.0;
        for (const auto& p : hits) { sum_x += p.first; sum_y += p.second; }
        meas_cell.mean_x = sum_x / hits.size(); meas_cell.mean_y = sum_y / hits.size();
        double sum_xx = 0.0, sum_xy = 0.0, sum_yy = 0.0;
        for (const auto& p : hits) {
            double dx = p.first - meas_cell.mean_x; double dy = p.second - meas_cell.mean_y;
            sum_xx += dx * dx; sum_xy += dx * dy; sum_yy += dy * dy;
        }
        double n_minus_1 = (hits.size() > 1) ? (hits.size() - 1) : 1.0;
        double cov_xx = (sum_xx / n_minus_1) + lidar_variance_reg;
        double cov_xy = (sum_xy / n_minus_1);
        double cov_yy = (sum_yy / n_minus_1) + lidar_variance_reg;
        double det = cov_xx * cov_yy - cov_xy * cov_xy;
        if (std::abs(det) < 1e-9) continue;
        double inv_det = 1.0 / det;
        meas_cell.inv_cov_xx = cov_yy * inv_det; meas_cell.inv_cov_xy = -cov_xy * inv_det; meas_cell.inv_cov_yy = cov_xx * inv_det;
        meas_cell.has_lidar_model = true;
    }
}

void DynamicGridMap::updateOccupancy(double birth_prob) {
    // ... (Occupancy Update logic unchanged) ...
    for (auto& c : grid_) {
        c.m_occ  = std::max(0.0, c.m_occ  * 0.55);
        c.m_free = std::max(0.0, c.m_free * 0.55);
        c.rho_b  = std::max(0.0, c.rho_b  * 0.55);
        c.rho_p  = std::max(0.0, c.rho_p  * 0.55);
    }
    for (int idx = 0; idx < static_cast<int>(grid_.size()); ++idx) {
        auto& cell = grid_[idx];
        const auto& meas = measurement_grid_[idx];
        double m_occ_z_proxy = meas.m_occ_z; 
        double m_occ_pred = std::min(1.0, std::max(0.0, cell.m_occ));
        double m_free_pred= std::min(1.0, std::max(0.0, cell.m_free));
        double m_unk_pred = std::max(0.0, 1.0 - m_occ_pred - m_free_pred);
        double K = m_occ_pred * meas.m_free_z + m_free_pred * m_occ_z_proxy;
        double norm = 1.0 / std::max(1e-9, (1.0 - K));
        double m_occ_upd = norm * (m_occ_pred * (1.0 - meas.m_free_z) + m_unk_pred * m_occ_z_proxy);
        double m_free_upd= norm * (m_free_pred * (1.0 - m_occ_z_proxy) + m_unk_pred * meas.m_free_z);
        m_occ_upd = std::min(1.0, std::max(0.0, m_occ_upd));
        m_free_upd= std::min(1.0, std::max(0.0, m_free_upd));
        double term = m_occ_pred + birth_prob * m_unk_pred;
        cell.rho_b = (term > 1e-9) ? (m_occ_upd * birth_prob * m_unk_pred) / term : 0.0;
        cell.rho_p = std::max(0.0, m_occ_upd - cell.rho_b);
        cell.m_occ  = m_occ_upd; cell.m_free = m_free_upd;
    }
}

void DynamicGridMap::calculateVelocityStatistics(double max_vel_for_scaling,
                                                 bool   use_ego_comp,
                                                 const EgoCalibration& ego_calib)
{
    // 1. Initialization
    for (auto& cell : grid_) {
        cell.is_dynamic = false; 
        cell.dyn_streak = 0; cell.stat_streak = 0;
        cell.dynamic_score *= 0.85; 
        if (cell.dynamic_score < 0.01) cell.dynamic_score = 0.0;
        if (cell.m_free > 0.8) cell.free_streak = std::min<std::uint8_t>(255, cell.free_streak + 1);
        else if (cell.m_occ > 0.6) cell.free_streak = 0;
    }

    auto& parts = particle_filter_->getParticles();
    const int need_on_frames = 2; const int need_off_frames = 4;

    auto flush_cell = [&](int cell_idx, int start, int end) {
        if (cell_idx < 0 || cell_idx >= static_cast<int>(grid_.size())) return;
        auto& c = grid_[cell_idx];

        if (end - start <= 2) {
            c.stat_streak = std::min<std::uint8_t>(255, c.stat_streak + 1);
            if (c.stat_streak >= need_off_frames) c.is_dynamic = false;
            c.dyn_streak = 0; return;
        }

        // --- [Cell-Level Winner Mode] ---
        // 1. Find Winner in Cell
        double max_weight = -1.0;
        int winner_idx = start;
        for (int j = start; j < end; ++j) {
            if (parts[j].weight > max_weight) {
                max_weight = parts[j].weight;
                winner_idx = j;
            }
        }
        
        // 2. Sector Gating (Vel 0.1, Ang 60.0)
        double mode_vx_sum = 0.0;
        double mode_vy_sum = 0.0;
        double mode_w_sum = 0.0;

        for (int j = start; j < end; ++j) {
            if (particle_filter_->checkSectorMatch(parts[winner_idx], parts[j], 
                                                   particle_vector_vel_thresh_, 
                                                   particle_vector_ang_thresh_)) 
            {
                mode_vx_sum += parts[j].vx * parts[j].weight;
                mode_vy_sum += parts[j].vy * parts[j].weight;
                mode_w_sum  += parts[j].weight;
            }
        }

        if (mode_w_sum <= 1e-9) {
            c.stat_streak = std::min<std::uint8_t>(255, c.stat_streak + 1);
            if (c.stat_streak >= need_off_frames) c.is_dynamic = false;
            c.dyn_streak = 0; return;
        }

        c.mean_vx = mode_vx_sum / mode_w_sum;
        c.mean_vy = mode_vy_sum / mode_w_sum;
        // --- [End Cell-Level] ---

        double speed_p = std::sqrt(c.mean_vx * c.mean_vx + c.mean_vy * c.mean_vy);
        if (use_ego_comp) { 
            speed_p = ego_calib.getAbsoluteSpeed(c.mean_vx, c.mean_vy);
        }

        double m_unk = std::max(0.0, 1.0 - c.m_occ - c.m_free);
        const bool is_occupied = (c.m_occ >= c.m_free && c.m_occ >= m_unk);
        const bool has_speed_from_particles = (speed_p > particle_static_vel_thresh_);

        double max_comp_speed = 0.0;
        int gx, gy; indexToGrid(cell_idx, gx, gy);

        if (use_radar_) {
             for (int dy = -radar_hint_search_radius_; dy <= radar_hint_search_radius_; ++dy) {
                for (int dx = -radar_hint_search_radius_; dx <= radar_hint_search_radius_; ++dx) {
                    int nx = gx + dx; int ny = gy + dy;
                    if (isInside(nx, ny)) {
                        const auto& neighbor = grid_[gridToIndex(nx, ny)];
                        double cell_wx, cell_wy;
                        gridToWorld(nx, ny, cell_wx, cell_wy);

                        for (const auto& hint : neighbor.radar_hints) {
                            double azimuth = std::atan2(cell_wy - hint.sensor_y, cell_wx - hint.sensor_x);
                            double abs_vr = std::abs(ego_calib.getAbsoluteRadialVelocity(hint.vr, azimuth));
                            if (abs_vr > max_comp_speed) {
                                max_comp_speed = abs_vr;
                            }
                        }
                    }
                }
            }
        }
        
        const bool has_speed_from_radar = (max_comp_speed > radar_static_vel_thresh_);
        bool dyn_candidate = false;
        if (use_radar_) {
            dyn_candidate = is_occupied && (has_speed_from_particles || has_speed_from_radar);
            bool is_currently_static = is_occupied && !dyn_candidate;
            if (use_fsd_ && is_currently_static && c.stat_streak >= fsd_T_static_ && c.free_streak >= fsd_T_free_) { 
                dyn_candidate = true; c.free_streak = 0; 
            }
        } else {
            dyn_candidate = is_occupied && has_speed_from_particles;
        }

        if (dyn_candidate) {
            int streak_increase = (has_speed_from_radar) ? 2 : 1;
            c.dyn_streak  = std::min<std::uint8_t>(255, c.dyn_streak + streak_increase);
            c.stat_streak = 0;
        } else {
            c.stat_streak = std::min<std::uint8_t>(255, c.stat_streak + 1);
            c.dyn_streak  = 0;
        }

        if (!c.is_dynamic && c.dyn_streak >= need_on_frames) c.is_dynamic = true;
        if ( c.is_dynamic && c.stat_streak >= need_off_frames) c.is_dynamic = false;

        const double target = c.is_dynamic ? std::min(1.0, speed_p / std::max(1e-6, max_vel_for_scaling)) : 0.0;
        const double alpha  = 0.6;
        c.dynamic_score = alpha * target + (1.0 - alpha) * c.dynamic_score;
    }; 

    int current_idx = -1; int first_i = 0;
    for (int i = 0; i <= static_cast<int>(parts.size()); ++i) {
        bool last = (i == static_cast<int>(parts.size()));
        int idx   = last ? -1 : parts[i].grid_cell_idx;
        if (last || idx != current_idx) {
            flush_cell(current_idx, first_i, i);
            if (last) break;
            current_idx = idx;
            first_i = i;
        }
    }

    // [New] Execute Cluster-based Processing if enabled
    if (cluster_mode_) {
        processDynamicClusters(ego_calib);
    }
}


// void DynamicGridMap::processDynamicClusters(const EgoCalibration& ego_calib) {
//     auto& parts = particle_filter_->getParticles();
    
//     // 1. Pre-compute particle ranges for each cell (Efficiency O(N))
//     // parts is sorted by grid_cell_idx. We build a map: cell_idx -> {start_idx, end_idx}
//     std::vector<std::pair<int, int>> cell_particle_ranges(grid_.size(), {-1, -1});
    
//     int current_cell = -1;
//     int start_p = 0;
//     for (int i = 0; i <= static_cast<int>(parts.size()); ++i) {
//         bool last = (i == static_cast<int>(parts.size()));
//         int c_idx = last ? -1 : parts[i].grid_cell_idx;
        
//         if (last || c_idx != current_cell) {
//             if (current_cell >= 0 && current_cell < static_cast<int>(grid_.size())) {
//                 cell_particle_ranges[current_cell] = {start_p, i};
//             }
//             current_cell = c_idx;
//             start_p = i;
//         }
//     }

//     // 2. BFS Clustering
//     std::vector<bool> visited(grid_.size(), false);
    
//     for (int i = 0; i < grid_.size(); ++i) {
//         // Only process dynamic cells that haven't been visited
//         if (visited[i] || !grid_[i].is_dynamic) continue;
        
//         // Start new cluster
//         std::vector<int> cluster_indices;
//         std::queue<int> q;
        
//         q.push(i);
//         visited[i] = true;
        
//         while (!q.empty()) {
//             int curr = q.front(); q.pop();
//             cluster_indices.push_back(curr);
            
//             int gx, gy; indexToGrid(curr, gx, gy);
            
//             // 8-Connectivity Search
//             for (int dy = -1; dy <= 1; ++dy) {
//                 for (int dx = -1; dx <= 1; ++dx) {
//                     if (dx == 0 && dy == 0) continue;
//                     int nx = gx + dx; int ny = gy + dy;
                    
//                     if (isInside(nx, ny)) {
//                         int nidx = gridToIndex(nx, ny);
//                         if (!visited[nidx] && grid_[nidx].is_dynamic) {
//                             visited[nidx] = true;
//                             q.push(nidx);
//                         }
//                     }
//                 }
//             }
//         }
        
//         // 3. Process the Cluster
//         if (cluster_indices.empty()) continue;

//         // A. Find "Global Winner" in the entire cluster
//         double global_max_weight = -1.0;
//         int global_winner_p_idx = -1;

//         // Iterate all cells in cluster to find winner particle
//         for (int c_idx : cluster_indices) {
//             std::pair<int, int> range = cell_particle_ranges[c_idx];
//             if (range.first == -1) continue; // No particles in this cell

//             for (int p_idx = range.first; p_idx < range.second; ++p_idx) {
//                 if (parts[p_idx].weight > global_max_weight) {
//                     global_max_weight = parts[p_idx].weight;
//                     global_winner_p_idx = p_idx;
//                 }
//             }
//         }

//         if (global_winner_p_idx == -1) continue; // Should not happen if dynamic

//         const Particle& global_winner = parts[global_winner_p_idx];

//         // B. Calculate Cluster Mean Velocity using Sector Gating
//         double cluster_vx_sum = 0.0;
//         double cluster_vy_sum = 0.0;
//         double cluster_w_sum = 0.0;

//         for (int c_idx : cluster_indices) {
//             std::pair<int, int> range = cell_particle_ranges[c_idx];
//             if (range.first == -1) continue;

//             for (int p_idx = range.first; p_idx < range.second; ++p_idx) {
//                 // Check Sector Match (0.1 m/s, 60.0 deg) against Global Winner
//                 if (particle_filter_->checkSectorMatch(global_winner, parts[p_idx], 
//                                                        particle_vector_vel_thresh_, 
//                                                        particle_vector_ang_thresh_)) 
//                 {
//                     cluster_vx_sum += parts[p_idx].vx * parts[p_idx].weight;
//                     cluster_vy_sum += parts[p_idx].vy * parts[p_idx].weight;
//                     cluster_w_sum  += parts[p_idx].weight;
//                 }
//             }
//         }

//         // C. Update Constituent Cells
//         if (cluster_w_sum > 1e-9) {
//             double cluster_mean_vx = cluster_vx_sum / cluster_w_sum;
//             double cluster_mean_vy = cluster_vy_sum / cluster_w_sum;

//             // Apply unified velocity to all cells in the cluster
//             for (int c_idx : cluster_indices) {
//                 grid_[c_idx].mean_vx = cluster_mean_vx;
//                 grid_[c_idx].mean_vy = cluster_mean_vy;
//             }
//         }
//     }
// }

void DynamicGridMap::processDynamicClusters(const EgoCalibration& ego_calib) {
    // 1. BFS Clustering (기존과 동일: 8-이웃 연결성으로 동적 셀 그룹화)
    std::vector<bool> visited(grid_.size(), false);
    
    for (int i = 0; i < grid_.size(); ++i) {
        // 이미 방문했거나 동적 셀이 아니면 건너뜀
        if (visited[i] || !grid_[i].is_dynamic) continue;
        
        std::vector<int> cluster_indices;
        std::queue<int> q;
        q.push(i);
        visited[i] = true;
        
        // BFS 탐색 시작
        while (!q.empty()) {
            int curr = q.front(); q.pop();
            cluster_indices.push_back(curr);
            int gx, gy; indexToGrid(curr, gx, gy);
            
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    int nx = gx + dx; int ny = gy + dy;
                    if (isInside(nx, ny)) {
                        int nidx = gridToIndex(nx, ny);
                        if (!visited[nidx] && grid_[nidx].is_dynamic) {
                            visited[nidx] = true;
                            q.push(nidx);
                        }
                    }
                }
            }
        }
        
        // 2. [수정됨] 클러스터 처리: 단순 평균 계산 (Simple Average of Cell Velocities)
        if (cluster_indices.empty()) continue;

        double cluster_vx_sum = 0.0;
        double cluster_vy_sum = 0.0;
        int count = 0;

        // 클러스터 내에 속한 각 셀의 속도(이미 개별적으로 계산된 값)를 모두 더함
        for (int c_idx : cluster_indices) {
             cluster_vx_sum += grid_[c_idx].mean_vx;
             cluster_vy_sum += grid_[c_idx].mean_vy;
             count++;
        }

        // 3. 평균 속도 적용
        if (count > 0) {
            double cluster_mean_vx = cluster_vx_sum / static_cast<double>(count);
            double cluster_mean_vy = cluster_vy_sum / static_cast<double>(count);

            // 계산된 평균 속도를 클러스터 내 모든 셀에 일괄 덮어쓰기
            for (int c_idx : cluster_indices) {
                grid_[c_idx].mean_vx = cluster_mean_vx;
                grid_[c_idx].mean_vy = cluster_mean_vy;
            }
        }
    }
}


std::vector<Particle> DynamicGridMap::generateNewParticles(double newborn_vel_stddev,
                                               double min_dynamic_birth_ratio,
                                               double max_dynamic_birth_ratio,
                                               double max_radar_speed_for_scaling,
                                               double dynamic_newborn_vel_stddev,
                                               const EgoCalibration& ego_calib)
{
    std::vector<Particle> new_particles;
    
    // 1. Ego Motion 보정용 정적 속도 가져오기
    double static_vx, static_vy;
    ego_calib.getStaticParticleVelocity(static_vx, static_vy);
    
    // 2. 속도 분포 설정 (기존 로직 유지)
    std::normal_distribution<double> static_vel_dist_x(static_vx, newborn_vel_stddev);
    std::normal_distribution<double> static_vel_dist_y(static_vy, newborn_vel_stddev);
    // 동적 파티클은 방향성 없이 랜덤하게 퍼짐 (Zero Mean + Large Stddev)
    std::normal_distribution<double> dynamic_fallback_dist(0.0, dynamic_newborn_vel_stddev);

    // [수정 핵심] 위치 랜덤 분포 추가 (Uniform Distribution)
    // 파티클 생성 위치를 셀 중심에서 셀 크기(resolution) 전체 영역으로 확장
    // 범위: [-resolution_/2.0, +resolution_/2.0]
    std::uniform_real_distribution<double> pos_dist(-resolution_ / 2.0, resolution_ / 2.0);

    for (int y = 0; y < grid_height_; ++y) {
        for (int x = 0; x < grid_width_; ++x) {
            int idx = gridToIndex(x, y);
            const auto& cell = grid_[idx];

            // 탄생(Birth) 조건: 확률 질량(rho_b)이 충분하고 점유된 셀일 때
            if (cell.rho_b > 0.5 && cell.m_occ > 0.6) {
                int num_to_birth = static_cast<int>(std::ceil(cell.rho_b * 4.0));
                
                // [Radar Hint 로직] - 여기서는 동적 파티클 비율 계산에만 사용 (기존 유지)
                double max_comp_speed = 0.0;
                if (use_radar_) {
                    for (int dy = -radar_hint_search_radius_; dy <= radar_hint_search_radius_; ++dy) {
                        for (int dx = -radar_hint_search_radius_; dx <= radar_hint_search_radius_; ++dx) {
                            int nx = x + dx; int ny = y + dy;
                            if (isInside(nx, ny)) {
                                const auto& neighbor = grid_[gridToIndex(nx, ny)];
                                double cell_wx, cell_wy;
                                gridToWorld(nx, ny, cell_wx, cell_wy);
                                for (const auto& hint : neighbor.radar_hints) {
                                    double azimuth = std::atan2(cell_wy - hint.sensor_y, cell_wx - hint.sensor_x);
                                    double abs_vr = std::abs(ego_calib.getAbsoluteRadialVelocity(hint.vr, azimuth));
                                    if (abs_vr > max_comp_speed) max_comp_speed = abs_vr;
                                }
                            }
                        }
                    }
                }

                // 동적 파티클 비율 결정
                double current_dynamic_ratio = min_dynamic_birth_ratio;
                if (max_comp_speed > 1e-6) {
                    double scale = std::min(1.0, max_comp_speed / std::max(1e-6, max_radar_speed_for_scaling));
                    current_dynamic_ratio = min_dynamic_birth_ratio + (max_dynamic_birth_ratio - min_dynamic_birth_ratio) * scale;
                }

                int num_dynamic = static_cast<int>(num_to_birth * current_dynamic_ratio);
                int num_static = num_to_birth - num_dynamic;

                // 3. 정적 파티클 생성 (Static Particles)
                for (int i = 0; i < num_static; ++i) {
                    Particle p;
                    gridToWorld(x, y, p.x, p.y); // 셀 중심 좌표 할당
                    
                    // [적용] 위치 랜덤 노이즈 추가 -> 셀 전체 영역에 분포
                    p.x += pos_dist(random_generator_);
                    p.y += pos_dist(random_generator_);
                    
                    p.vx = static_vel_dist_x(random_generator_);
                    p.vy = static_vel_dist_y(random_generator_);
                    p.weight = cell.rho_b / static_cast<double>(num_to_birth);
                    p.grid_cell_idx = idx; p.age = 0;
                    new_particles.push_back(p);
                }

                // 4. 동적 파티클 생성 (Dynamic Particles)
                for (int i = 0; i < num_dynamic; ++i) {
                    Particle p;
                    gridToWorld(x, y, p.x, p.y); // 셀 중심 좌표 할당
                    
                    // [적용] 위치 랜덤 노이즈 추가 -> 셀 전체 영역에 분포 (탈출 확률 확보)
                    p.x += pos_dist(random_generator_);
                    p.y += pos_dist(random_generator_);

                    // 속도는 기존과 동일하게 랜덤 분포 사용 (방향성 없음)
                    p.vx = dynamic_fallback_dist(random_generator_) + static_vx; 
                    p.vy = dynamic_fallback_dist(random_generator_) + static_vy; 
                    
                    p.weight = cell.rho_b / static_cast<double>(num_to_birth);
                    p.grid_cell_idx = idx; p.age = 0;
                    new_particles.push_back(p);
                }
            }
        }
    }
    return new_particles;
}

void DynamicGridMap::toOccupancyGridMsg(nav_msgs::OccupancyGrid& msg, const std::string& frame_id) const {
    msg.header.stamp = ros::Time::now();
    msg.header.frame_id = frame_id;
    msg.info.resolution = resolution_;
    msg.info.width  = grid_width_;
    msg.info.height = grid_height_;
    msg.info.origin.position.x = origin_x_;
    msg.info.origin.position.y = origin_y_;
    msg.info.origin.orientation.w = 1.0;
    msg.data.assign(grid_width_ * grid_height_, -1);
    for (size_t i = 0; i < grid_.size(); ++i) {
        const auto& c = grid_[i];
        double m_unk = std::max(0.0, 1.0 - c.m_occ - c.m_free);
        if (c.m_occ >= c.m_free && c.m_occ >= m_unk) {
            msg.data[i] = static_cast<int8_t>(std::min(100.0, c.m_occ * 100.0));
        } else if (c.m_free >= m_unk) {
            msg.data[i] = 0;
        } else {
            msg.data[i] = -1;
        }
    }
}

void DynamicGridMap::toMarkerArrayMsg(visualization_msgs::MarkerArray& arr,
                                      const std::string& frame_id,
                                      bool show_velocity_arrows,
                                      const EgoCalibration& ego_calib) const {
    arr.markers.clear();
    visualization_msgs::Marker cubes;
    cubes.header.stamp = ros::Time::now(); cubes.header.frame_id = frame_id; cubes.ns = "dogm_cells"; cubes.id = 0;
    cubes.type = visualization_msgs::Marker::CUBE_LIST; cubes.action = visualization_msgs::Marker::ADD;
    cubes.pose.orientation.w = 1.0; cubes.scale.x = resolution_; cubes.scale.y = resolution_; cubes.scale.z = 0.02;
    cubes.lifetime = ros::Duration(1);
    for (int y = 0; y < grid_height_; ++y) {
        for (int x = 0; x < grid_width_; ++x) {
            const auto& c = grid_[gridToIndex(x, y)];
            geometry_msgs::Point p; p.x = origin_x_ + (x + 0.5) * resolution_; p.y = origin_y_ + (y + 0.5) * resolution_; p.z = -0.02;
            std_msgs::ColorRGBA col; col.a = 0.2;
            double m_unk = std::max(0.0, 1.0 - c.m_occ - c.m_free);
            if (c.m_occ >= c.m_free && c.m_occ >= m_unk) {
                if (c.is_dynamic) { col.r = 1.0f; col.g = 0.0f; col.b = 0.0f; } 
                else              { col.r = 0.0f; col.g = 0.0f; col.b = 1.0f; } 
                col.a = 0.2 + 0.4 * std::min(1.0, c.m_occ); 
            } else if (c.m_free >= m_unk) { col.r = col.g = col.b = 1.0f; col.a = 0.5f; } 
            else { col.r = col.g = col.b = 0.5f; col.a = 0.5f; }
            cubes.points.push_back(p); cubes.colors.push_back(col);
        }
    }
    arr.markers.push_back(cubes);

    if (show_velocity_arrows) {
        visualization_msgs::Marker arrows;
        arrows.header.stamp = ros::Time::now(); arrows.header.frame_id = frame_id; arrows.ns = "dogm_vel"; arrows.id = 1;
        arrows.type = visualization_msgs::Marker::ARROW; arrows.action = visualization_msgs::Marker::ADD;
        arrows.scale.x = 0.02; arrows.scale.y = 0.04; arrows.scale.z = 0.04;
        arrows.color.r = 1.0; arrows.color.g = 0.0; arrows.color.b = 0.0; arrows.color.a = 1.0;
        arrows.lifetime = ros::Duration(0.2);
        int arrow_id = 10;
        for (int y = 0; y < grid_height_; ++y) {
            for (int x = 0; x < grid_width_; ++x) {
                const auto& c = grid_[gridToIndex(x, y)];
                if (!c.is_dynamic || c.m_occ < 0.6) continue;
                geometry_msgs::Point p0, p1;
                p0.x = origin_x_ + (x + 0.5) * resolution_; p0.y = origin_y_ + (y + 0.5) * resolution_; p0.z = 0.00;
                double abs_vx, abs_vy;
                ego_calib.getAbsoluteVelocity(c.mean_vx, c.mean_vy, abs_vx, abs_vy);
                const double scale = 0.4;
                p1.x = p0.x + scale * abs_vx; p1.y = p0.y + scale * abs_vy; p1.z = 0.00;
                visualization_msgs::Marker a = arrows;
                a.id = arrow_id++; a.points.clear(); a.points.push_back(p0); a.points.push_back(p1);
                arr.markers.push_back(a);
            }
        }
    }
}

void DynamicGridMap::shiftGrid(double dx, double dy) {
    int shift_x = static_cast<int>(std::round(dx / resolution_));
    int shift_y = static_cast<int>(std::round(dy / resolution_));
    if (shift_x == 0 && shift_y == 0) return;
    std::vector<GridCell> new_grid(grid_.size());
    std::vector<MeasurementCell> new_meas(measurement_grid_.size());
    for (int y = 0; y < grid_height_; ++y) {
        for (int x = 0; x < grid_width_; ++x) {
            int old_x = x + shift_x; int old_y = y + shift_y;
            if (isInside(old_x, old_y)) {
                int new_idx = gridToIndex(x, y); int old_idx = gridToIndex(old_x, old_y);
                new_grid[new_idx] = grid_[old_idx];
                new_meas[new_idx] = measurement_grid_[old_idx];
                for(auto& p : new_grid[new_idx].radar_buffer_1) { p.x -= dx; p.y -= dy; }
                for(auto& p : new_grid[new_idx].radar_buffer_2) { p.x -= dx; p.y -= dy; }
            }
        }
    }
    grid_ = new_grid;
    measurement_grid_ = new_meas;
}