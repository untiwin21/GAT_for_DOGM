#ifndef DOGM_ROS_EGO_CALIBRATION_H
#define DOGM_ROS_EGO_CALIBRATION_H

#include <nav_msgs/Odometry.h>
#include <cmath>
#include <algorithm>

class EgoCalibration {
public:
    EgoCalibration();
    ~EgoCalibration() = default;

    // 오도메트리 데이터 업데이트
    void update(double vx, double vy);

    // 그리드 이동량 계산 (Shift)
    // 로봇이 dt동안 움직인 거리만큼 맵 인덱스를 얼마나 밀어야 하는지 계산
    void getGridShift(double dt, double resolution, int& shift_x, int& shift_y, double& dx, double& dy) const;

    // 정적 파티클 속도 반환 (Birth)
    // "정적 물체는 로봇 반대 방향(-ego)으로 움직인다"
    void getStaticParticleVelocity(double& vx, double& vy) const;

    // 절대 속도 복원 (Particle 판단용)
    // "상대 속도 + 로봇 속도 = 절대 속도"
    double getAbsoluteSpeed(double rel_vx, double rel_vy) const;

    // 절대 방사 속도 복원 (Radar 판단용) [중요!]
    // 레이다값(상대) + 로봇속도(투영) = 절대 방사 속도
    double getAbsoluteRadialVelocity(double raw_vr, double azimuth) const;

    // 시각화용 절대 속도 벡터 반환
    void getAbsoluteVelocity(double rel_vx, double rel_vy, double& abs_vx, double& abs_vy) const;

    // Getter
    double getVx() const;
    double getVy() const;

private:
    double ego_vx_;
    double ego_vy_;
};

#endif // DOGM_ROS_EGO_CALIBRATION_H