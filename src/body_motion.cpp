#include "body_motion.hpp"

Eigen::Matrix3d skew_symmetric(Eigen::Vector3d &v) {
    Eigen::Matrix3d ret;
    ret << 0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0;
    return ret;
}

Eigen::Matrix3d RotVec::to_rot_mat() {
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    double cos_theta = std::cos(angle);
    double sin_theta = std::sin(angle);
    Eigen::Matrix3d n_up = skew_symmetric(axis);
    Eigen::Matrix3d ret = cos_theta * I +
                          (1 - cos_theta) * axis * axis.transpose() +
                          sin_theta * n_up;
    return ret;
}

RotVec rotmat_to_rotvec(Eigen::Matrix3d &R) {
    Eigen::Vector3d axis{1.0, 0.0, 0.0};
    double angle{0.0};
    return RotVec(axis, angle);
}