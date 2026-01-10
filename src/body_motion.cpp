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
    // A = R - I
    Eigen::Matrix3d A = R - Eigen::Matrix3d::Identity();
    // SVD of A, get full V
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(A, Eigen::ComputeFullV);
    // the last col of V is the rotation axis
    Eigen::Vector3d axis = svd.matrixV().col(2);

    double trace = R.trace();
    double angle = std::acos((trace - 1.0) / 2.0);

    return RotVec(axis, angle);
}

Eigen::Matrix3d quaternion_to_rotmat(Eigen::Quaternionf &q) {
    double s = q.w();
    Eigen::Vector3d v(q.x(), q.y(), q.z());
    auto I = Eigen::Matrix3d::Identity();
    auto v_up = skew_symmetric(v);
    Eigen::Matrix3d R =
        v * v.transpose() + s * s * I + 2 * s * v_up + v_up * v_up;
    return R;
}

RotVec quaternion_to_rotvec(Eigen::Quaternionf &q) {
    double angle = 2 * std::acos(q.w());
    Eigen::Vector3d axis(q.x(), q.y(), q.z());
    axis /= std::sin(angle / 2.0);
    return RotVec(axis, angle);
}