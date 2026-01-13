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

Eigen::Quaterniond rotmat_to_quaternion(Eigen::Matrix3d &R) {
    Eigen::Quaterniond q;
    double trace = R.trace();
    if (trace > 0) {
        double s = 2.0 * std::sqrt(trace + 1.0);
        q.w() = 0.25 * s;
        q.x() = (R(2, 1) - R(1, 2)) / s;
        q.y() = (R(0, 2) - R(2, 0)) / s;
        q.z() = (R(1, 0) - R(0, 1)) / s;
    } else {
        if (R(0, 0) > R(1, 1) && R(0, 0) > R(2, 2)) {
            double s = 2.0 * std::sqrt(1.0 + R(0, 0) - R(1, 1) - R(2, 2));
            q.w() = (R(2, 1) - R(1, 2)) / s;
            q.x() = 0.25 * s;
            q.y() = (R(0, 1) + R(1, 0)) / s;
            q.z() = (R(0, 2) + R(2, 0)) / s;
        } else if (R(1, 1) > R(2, 2)) {
            double s = 2.0 * std::sqrt(1.0 + R(1, 1) - R(0, 0) - R(2, 2));
            q.w() = (R(0, 2) - R(2, 0)) / s;
            q.x() = (R(0, 1) + R(1, 0)) / s;
            q.y() = 0.25 * s;
            q.z() = (R(1, 2) + R(2, 1)) / s;
        } else {
            double s = 2.0 * std::sqrt(1.0 + R(2, 2) - R(0, 0) - R(1, 1));
            q.w() = (R(1, 0) - R(0, 1)) / s;
            q.x() = (R(0, 2) + R(2, 0)) / s;
            q.y() = (R(1, 2) + R(2, 1)) / s;
            q.z() = 0.25 * s;
        }
    }
    return q.normalized();
}

Eigen::Quaterniond rotvec_to_quaternion(RotVec &rot_vec) {
    double half_angle = rot_vec.angle / 2.0;
    double sin_half = std::sin(half_angle);
    Eigen::Quaterniond q;
    q.w() = std::cos(half_angle);
    q.x() = rot_vec.axis[0] * sin_half;
    q.y() = rot_vec.axis[1] * sin_half;
    q.z() = rot_vec.axis[2] * sin_half;
    return q;

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

Eigen::Vector3d so3_log(Eigen::Matrix3d &R) {
    RotVec rotVec = rotmat_to_rotvec(R);
    return rotVec.angle * rotVec.axis;
}

Eigen::Matrix3d so3_exp(Eigen::Vector3d &phi) {
    double theta = phi.norm();
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    if (theta < 1e-8) {
        return I;
    } else {
        Eigen::Vector3d a = phi / theta;
        Eigen::Matrix3d a_up = skew_symmetric(a);
        double cos_theta = std::cos(theta);
        double sin_theta = std::sin(theta);
        return cos_theta * I + (1 - cos_theta) * a * a.transpose() +
               sin_theta * a_up;
    }
}

/**
 * @brief Computes the exponential map for SO(3) and provides intermediate
 * values for SE(3) computation.
 *
 * @param[in] phi The rotation vector.
 * @param[out] R The resulting rotation matrix.
 * @param[out] theta The angle of rotation.
 * @param[out] cos_theta The cosine of the rotation angle.
 * @param[out] sin_theta The sine of the rotation angle.
 * @param[out] a The normalized rotation axis.
 * @param[out] a_up The skew-symmetric matrix of the rotation axis.
 * @return bool True if the norm of phi > 1e-8, false otherwise.
 */
bool so3_exp_for_se3(Eigen::Vector3d &phi, Eigen::Matrix3d &R, double &theta,
                     double &cos_theta, double &sin_theta, Eigen::Vector3d &a,
                     Eigen::Matrix3d &a_up) {
    theta = phi.norm();
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    if (theta < 1e-8) {
        R = I;
        return false;
    } else {
        a = phi / theta;
        a_up = skew_symmetric(a);
        cos_theta = std::cos(theta);
        sin_theta = std::sin(theta);
        R = cos_theta * I + (1 - cos_theta) * a * a.transpose() +
            sin_theta * a_up;
        return true;
    }
}

Eigen::Matrix4d se3_exp(Eigen::Vector<double, 6> &xi) {
    Eigen::Vector3d rho = xi.head<3>();
    Eigen::Vector3d phi = xi.tail<3>();
    Eigen::Matrix3d R;
    double theta;
    double cos_theta;
    double sin_theta;
    Eigen::Vector3d a;
    Eigen::Matrix3d a_up;
    bool norm_flag =
        so3_exp_for_se3(phi, R, theta, cos_theta, sin_theta, a, a_up);
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t;
    if (norm_flag) {
        double sin_theta_over_theta = sin_theta / theta;
        Eigen::Matrix3d J = sin_theta_over_theta * I +
                            (1 - sin_theta_over_theta) * a * a.transpose() +
                            ((1 - cos_theta) / theta) * a_up;
        t = J * rho;
    } else {
        t = rho;
    }
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = R;
    T.block<3, 1>(0, 3) = t;
    return T;
}

Eigen::Vector<double, 6> se3_log(Eigen::Matrix4d &T) {
    Eigen::Matrix3d R = T.block<3, 3>(0, 0);
    Eigen::Vector3d t = T.block<3, 1>(0, 3);
    Eigen::Vector3d phi = so3_log(R);
    Eigen::Vector3d rho;
    Eigen::Matrix3d R_;
    double theta;
    double cos_theta;
    double sin_theta;
    Eigen::Vector3d a;
    Eigen::Matrix3d a_up;
    bool norm_flag =
        so3_exp_for_se3(phi, R_, theta, cos_theta, sin_theta, a, a_up);
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    if (norm_flag) {
        double sin_theta_over_theta = sin_theta / theta;
        Eigen::Matrix3d J = sin_theta_over_theta * I +
                            (1 - sin_theta_over_theta) * a * a.transpose() +
                            ((1 - cos_theta) / theta) * a_up;
        rho = J.partialPivLu().solve(t); // find rho by solving J * rho = t
    } else {
        rho = t;
    }
    Eigen::Vector<double, 6> xi;
    xi.head<3>() = rho;
    xi.tail<3>() = phi;
    return xi;
}