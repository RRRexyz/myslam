#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Geometry>

Eigen::Matrix3d skew_symmetric(Eigen::Vector3d &v);

struct RotVec {
    Eigen::Vector3d axis; // Rotation axis
    double angle;         // Rotation angle in radians

    RotVec(Eigen::Vector3d &axis_, double angle_) : axis(axis_), angle(angle_) {
        auto norm = std::sqrt(axis[0] * axis[0] + axis[1] * axis[1] +
                              axis[2] * axis[2]);
        // Normalize the axis-vector
        if (norm > 1e-8) {
            axis[0] /= norm;
            axis[1] /= norm;
            axis[2] /= norm;
        } else {
            throw std::invalid_argument("Rotation axis norm is zero.");
        }
    };

    Eigen::Matrix3d to_rot_mat();
};

RotVec rotmat_to_rotvec(Eigen::Matrix3d &R);
Eigen::Matrix3d quaternion_to_rotmat(Eigen::Quaternionf &q);
RotVec quaternion_to_rotvec(Eigen::Quaternionf &q);