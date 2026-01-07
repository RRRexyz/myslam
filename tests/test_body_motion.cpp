#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "body_motion.hpp"
#include "doctest.h"

TEST_CASE("transfer to skew_symmetric matrix") {
    Eigen::Vector3d v = Eigen::Vector3d{1.0, 2.0, 3.0};
    Eigen::Matrix3d expected;
    expected << 0.0, -3.0, 2.0, 3.0, 0.0, -1.0, -2.0, 1.0, 0.0;

    Eigen::Matrix3d result = skew_symmetric(v);

    CHECK(result.isApprox(expected));
}

TEST_CASE("RotVec") {
    SUBCASE("test init") {
        Eigen::Vector3d axis = Eigen::Vector3d{1.0, 2.0, 3.0};
        double angle{M_PI / 4}; // 45 degrees

        RotVec rotVec(axis, angle);

        // Check if the axis is normalized
        double norm = std::sqrt(rotVec.axis[0] * rotVec.axis[0] +
                                rotVec.axis[1] * rotVec.axis[1] +
                                rotVec.axis[2] * rotVec.axis[2]);
        CHECK(std::abs(norm - 1.0) < 1e-8);

        // Check if the angle is set correctly
        CHECK(rotVec.angle == angle);
    }
    SUBCASE("test zero axis") {
        Eigen::Vector3d axis = Eigen::Vector3d{0.0, 0.0, 0.0};
        double angle{M_PI / 4}; // 45 degrees

        CHECK_THROWS_AS(RotVec rotVec(axis, angle), std::invalid_argument);
    }
    SUBCASE("to_rot_mat") {
        Eigen::Vector3d axis = Eigen::Vector3d{1.0, 0.0, 0.0};
        double angle{M_PI / 6}; // 30 degrees

        RotVec rot_vec(axis, angle);
        Eigen::Matrix3d rotMat = rot_vec.to_rot_mat();

        Eigen::Matrix3d expected;
        expected << 1.0, 0.0, 0.0, 0.0, std::cos(angle), -std::sin(angle), 0.0,
            std::sin(angle), std::cos(angle);

        CHECK(rotMat.isApprox(expected, 1e-8));
    }
}

TEST_CASE("rotmat_to_rotvec") {
    Eigen::Matrix3d rotMat;
    double angle = M_PI / 3; // 60 degrees
    rotMat << std::cos(angle), -std::sin(angle), 0.0, std::sin(angle),
        std::cos(angle), 0.0, 0.0, 0.0, 1.0;

    RotVec rotVec = rotmat_to_rotvec(rotMat);

    // Expected axis is z-axis
    Eigen::Vector3d expected_axis = Eigen::Vector3d{0.0, 0.0, 1.0};

    CHECK(rotVec.axis.isApprox(expected_axis, 1e-8));
    CHECK(std::abs(rotVec.angle - angle) < 1e-8);
}