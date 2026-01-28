#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include <opencv2/opencv.hpp>

TEST_CASE("Undistored images") {
    // 畸变参数
    double k1 = -0.28340811;
    double k2 = 0.07395907;
    double p1 = 0.00019359;
    double p2 = 1.76187114e-05;
    // 内参
    double fx = 458.654;
    double fy = 457.296;
    double cx = 367.215;
    double cy = 248.375;

    cv::Mat img = cv::imread("../../res/distorted.png", cv::IMREAD_GRAYSCALE);
    cv::Mat expected;
    cv::Mat cameraMatrix =
        (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    cv::Mat distCoeffs = (cv::Mat_<double>(1, 4) << k1, k2, p1, p2);
    cv::undistort(img, expected, cameraMatrix, distCoeffs);
    cv::imshow("expected", expected);
    cv::waitKey(0);
}