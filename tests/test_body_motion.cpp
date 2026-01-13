#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "body_motion.hpp"
#include "doctest.h"
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

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

TEST_CASE("quaternion_to_rotmat") {
    SUBCASE("45 degrees around x-axis") {
        Eigen::Quaternionf q1(0.9238795f, 0.3826834f, 0.0f,
                              0.0f); // 45 degrees around x-axis
        Eigen::Matrix3d rotMat = quaternion_to_rotmat(q1);

        Eigen::Matrix3d expected;
        double angle = M_PI / 4; // 45 degrees
        expected << 1.0, 0.0, 0.0, 0.0, std::cos(angle), -std::sin(angle), 0.0,
            std::sin(angle), std::cos(angle);

        CHECK(rotMat.isApprox(expected, 1e-6));
    }
    SUBCASE("60 degrees around z-axis") {
        Eigen::Quaternionf q2(0.8660254f, 0.0f, 0.0f,
                              0.5000000f); // 60 degrees around z-axis
        Eigen::Matrix3d rotMat = quaternion_to_rotmat(q2);

        Eigen::Matrix3d expected;
        double angle{M_PI / 3}; // 60 degrees
        expected << std::cos(angle), -std::sin(angle), 0.0, std::sin(angle),
            std::cos(angle), 0.0, 0.0, 0.0, 1.0;

        CHECK(rotMat.isApprox(expected, 1e-6));
    }
}

TEST_CASE("quaternion_to_rotvec") {
    SUBCASE("45 degrees around x-axis") {
        Eigen::Quaternionf q1(0.9238795f, 0.3826834f, 0.0f,
                              0.0f); // 45 degrees around x-axis
        RotVec rotVec = quaternion_to_rotvec(q1);

        Eigen::Vector3d expected_axis = Eigen::Vector3d{1.0, 0.0, 0.0};
        double expected_angle = M_PI / 4; // 45 degrees

        CHECK(rotVec.axis.isApprox(expected_axis, 1e-6));
        CHECK(std::abs(rotVec.angle - expected_angle) < 1e-6);
    }
    SUBCASE("60 degrees around z-axis") {
        Eigen::Quaternionf q2(0.8660254f, 0.0f, 0.0f,
                              0.5000000f); // 60 degrees around z-axis
        RotVec rotVec = quaternion_to_rotvec(q2);

        Eigen::Vector3d expected_axis = Eigen::Vector3d{0.0, 0.0, 1.0};
        double expected_angle = M_PI / 3; // 60 degrees

        CHECK(rotVec.axis.isApprox(expected_axis, 1e-6));
        CHECK(std::abs(rotVec.angle - expected_angle) < 1e-6);
    }
}

TEST_CASE("so3_log") {
    Eigen::Matrix3d rotMat;
    double angle = M_PI / 4; // 45 degrees
    rotMat << std::cos(angle), -std::sin(angle), 0.0, std::sin(angle),
        std::cos(angle), 0.0, 0.0, 0.0, 1.0;
    Eigen::Vector3d log_vec = so3_log(rotMat);

    Sophus::SO3d sophus_rot(rotMat);
    Eigen::Vector3d expected = sophus_rot.log();

    CHECK(log_vec.isApprox(expected, 1e-8));
}

TEST_CASE("so3_exp") {
    SUBCASE("when phi's norm is too small") {
        Eigen::Vector3d phi = Eigen::Vector3d{1e-8, 1e-9, 1e-10};
        Eigen::Matrix3d exp_mat = so3_exp(phi);

        Eigen::Matrix3d expected = Eigen::Matrix3d::Identity();

        CHECK(exp_mat.isApprox(expected, 1e-8));
    }
    SUBCASE("when phi's norm is normal") {
        Eigen::Vector3d phi = Eigen::Vector3d{1.2, 2.3, 4.8};
        Eigen::Matrix3d exp_mat = so3_exp(phi);

        Sophus::SO3d sophus_rot = Sophus::SO3d::exp(phi);
        Eigen::Matrix3d expected = sophus_rot.matrix();

        CHECK(exp_mat.isApprox(expected, 1e-8));
    }
}

TEST_CASE("se3_exp") {
    SUBCASE("when phi's norm is too small") {
        Eigen::Vector<double, 6> xi;
        xi << 1.0, 2.0, 3.0, 1e-8, 1e-9, 1e-10;
        Eigen::Matrix4d T = se3_exp(xi);

        Sophus::SE3d sophus_T = Sophus::SE3d::exp(xi);
        Eigen::Matrix4d expected = sophus_T.matrix();

        CHECK(T.isApprox(expected, 1e-8));
    }
    SUBCASE("when phi's norm is normal") {
        Eigen::Vector<double, 6> xi;
        xi << 1.0, 2.0, 3.0, 0.5, -1.0, 2.0;
        Eigen::Matrix4d T = se3_exp(xi);

        Sophus::SE3d sophus_T = Sophus::SE3d::exp(xi);
        Eigen::Matrix4d expected = sophus_T.matrix();

        CHECK(T.isApprox(expected, 1e-8));
    }
}

TEST_CASE("se3_log") {
    Eigen::Matrix4d T;
    double angle = M_PI / 6; // 30 degrees
    T << std::cos(angle), -std::sin(angle), 0.0, 1.0, std::sin(angle),
        std::cos(angle), 0.0, 2.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 1.0;
    Eigen::Vector<double, 6> xi = se3_log(T);

    Sophus::SE3d sophus_T(T);
    Eigen::Vector<double, 6> expected = sophus_T.log();

    CHECK(xi.isApprox(expected, 1e-8));
}