#include "dogm_ros/ego_calibration.h"

EgoCalibration::EgoCalibration() 
    : ego_vx_(0.0), ego_vy_(0.0) {}

void EgoCalibration::update(double vx, double vy) {
    ego_vx_ = vx;
    ego_vy_ = vy;
}

void EgoCalibration::getGridShift(double dt, double resolution, int& shift_x, int& shift_y, double& dx, double& dy) const {
    dx = ego_vx_ * dt;
    dy = ego_vy_ * dt;
    
    // 로봇 이동량을 그리드 인덱스 단위로 변환
    shift_x = static_cast<int>(std::round(dx / resolution));
    shift_y = static_cast<int>(std::round(dy / resolution));
}

void EgoCalibration::getStaticParticleVelocity(double& vx, double& vy) const {
    // 로봇 내부(Relative)에서는 정적 물체가 로봇 반대로 움직임
    vx = -ego_vx_;
    vy = -ego_vy_;
}

double EgoCalibration::getAbsoluteSpeed(double rel_vx, double rel_vy) const {
    // 상대 속도 + 로봇 속도 = 절대 속도 (Vector Sum)
    double abs_vx = rel_vx + ego_vx_;
    double abs_vy = rel_vy + ego_vy_;
    return std::sqrt(abs_vx * abs_vx + abs_vy * abs_vy);
}

// [핵심] 레이다 속도 보정 구현
double EgoCalibration::getAbsoluteRadialVelocity(double raw_vr, double azimuth) const {
    // 로봇 속도를 물체 방향으로 투영 (내적)
    double ego_proj = ego_vx_ * std::cos(azimuth) + ego_vy_ * std::sin(azimuth);
    
    // 상대 속도 + 투영된 로봇 속도 = 절대 속도
    return raw_vr + ego_proj;
}

void EgoCalibration::getAbsoluteVelocity(double rel_vx, double rel_vy, double& abs_vx, double& abs_vy) const {
    abs_vx = rel_vx + ego_vx_;
    abs_vy = rel_vy + ego_vy_;
}

double EgoCalibration::getVx() const { return ego_vx_; }
double EgoCalibration::getVy() const { return ego_vy_; }