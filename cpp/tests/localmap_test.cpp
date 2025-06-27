#include <gtest/gtest.h>
#include "isaeslam/data/maps/localmap.h"
#include "isaeslam/data/sensors/Camera.h"
#include <Eigen/Dense>
#include <memory>

using namespace isae;

// Test LocalMap::computeRelativePose
TEST(LocalMapTest, ComputeRelativePose) {
    // Create a LocalMap instance
    LocalMap local_map(1, 10, 0);

	// Create sensors
	std::shared_ptr<ImageSensor> sensor1 = std::make_shared<Camera>(cv::Mat(), Eigen::Matrix3d::Identity());
	std::shared_ptr<ImageSensor> sensor2 = std::make_shared<Camera>(cv::Mat(), Eigen::Matrix3d::Identity());

    // Create mock frames
    Eigen::Affine3d transform1 = Eigen::Affine3d::Identity();
    Eigen::Affine3d transform2 = Eigen::Affine3d(Eigen::Translation3d(1.0, 0.0, 0.0));
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(6, 6);

    std::shared_ptr<Frame> frame1 = std::make_shared<Frame>();
	std::vector<std::shared_ptr<ImageSensor>> sensors1;
	sensors1.push_back(sensor1);
	frame1->init(sensors1, 0);
	frame1->setWorld2FrameTransform(transform1);
	frame1->setdTCov(cov);
    std::shared_ptr<Frame> frame2 = std::make_shared<Frame>();
	std::vector<std::shared_ptr<ImageSensor>> sensors2;
	sensors2.push_back(sensor2);
	frame2->init(sensors2, 1.0);
	frame2->setWorld2FrameTransform(transform2.inverse());
	frame2->setdTCov(cov);

    // Add frames to the local map
    local_map.addFrame(frame1);
    local_map.addFrame(frame2);

    // Variables to store results
    Eigen::Affine3d T_f1_f2;
    Eigen::MatrixXd computed_cov;

    // Test computeRelativePose
    bool result = local_map.computeRelativePose(frame1, frame2, T_f1_f2, computed_cov);

    // Verify the result
    EXPECT_TRUE(result);
    EXPECT_EQ(T_f1_f2.translation(), Eigen::Vector3d(1.0, 0.0, 0.0));
    EXPECT_EQ(computed_cov, cov);
}

TEST(LocalMapTest, ComputeRelativePoseInsufficientFrames) {
    // Create a LocalMap instance
    LocalMap local_map(1, 10, 0);

    // Create a mock frame
    Eigen::Affine3d transform = Eigen::Affine3d::Identity();
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(6, 6);

	std::shared_ptr<Frame> frame1 = std::make_shared<Frame>();
	std::shared_ptr<ImageSensor> sensor1 = std::make_shared<Camera>(cv::Mat(), Eigen::Matrix3d::Identity());
	std::vector<std::shared_ptr<ImageSensor>> sensors1;
	sensors1.push_back(sensor1);
	frame1->init(sensors1, 0);
	frame1->setWorld2FrameTransform(transform);
	frame1->setdTCov(cov);

    // Add only one frame to the local map
    local_map.addFrame(frame1);

    // Variables to store results
    Eigen::Affine3d T_f1_f2;
    Eigen::MatrixXd computed_cov;

    // Test computeRelativePose
    bool result = local_map.computeRelativePose(frame1, frame1, T_f1_f2, computed_cov);

    // Verify the result
    EXPECT_FALSE(result);
}

TEST(LocalMapTest, ComputeRelativePoseInBetweenTest) {
	// Create a LocalMap instance
    LocalMap local_map(1, 10, 0);

	// Create sensors
	std::shared_ptr<ImageSensor> sensor1 = std::make_shared<Camera>(cv::Mat(), Eigen::Matrix3d::Identity());
	std::shared_ptr<ImageSensor> sensor2 = std::make_shared<Camera>(cv::Mat(), Eigen::Matrix3d::Identity());
	std::shared_ptr<ImageSensor> sensor3 = std::make_shared<Camera>(cv::Mat(), Eigen::Matrix3d::Identity());

    // Create mock frames
    Eigen::Affine3d transform1 = Eigen::Affine3d::Identity();
    Eigen::Affine3d transform2 = Eigen::Affine3d(Eigen::Translation3d(1.0, 0.0, 0.0));
	Eigen::Affine3d transform3 = Eigen::Affine3d(Eigen::Translation3d(2.0, 0.0, 0.0));
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(6, 6);

    std::shared_ptr<Frame> frame1 = std::make_shared<Frame>();
	std::vector<std::shared_ptr<ImageSensor>> sensors1;
	sensors1.push_back(sensor1);
	frame1->init(sensors1, 0);
	frame1->setWorld2FrameTransform(transform1);
	frame1->setdTCov(cov);
    std::shared_ptr<Frame> frame2 = std::make_shared<Frame>();
	std::vector<std::shared_ptr<ImageSensor>> sensors2;
	sensors2.push_back(sensor2);
	frame2->init(sensors2, 1.0);
	frame2->setWorld2FrameTransform(transform2.inverse());
	frame2->setdTCov(cov);
	std::shared_ptr<Frame> frame3 = std::make_shared<Frame>();
	std::vector<std::shared_ptr<ImageSensor>> sensors3;
	sensors3.push_back(sensor3);
	frame3->init(sensors3, 2.0);
	frame3->setWorld2FrameTransform(transform3.inverse());
	frame3->setdTCov(cov);

    // Add frames to the local map
    local_map.addFrame(frame1);
    local_map.addFrame(frame2);
	local_map.addFrame(frame3);

    // Variables to store results
    Eigen::Affine3d T_f1_f3;
    Eigen::MatrixXd computed_cov;
    Eigen::MatrixXd cov_theory = Eigen::MatrixXd::Identity(6, 6);
    cov_theory << 2, 0, 0, 0, 0, 0,
                  0, 3, 0, 0, 0, 1,
                  0, 0, 3, 0, -1, 0,
                  0, 0, 0, 2, 0, 0,
                  0, 0, -1, 0, 2, 0,
                  0, 1, 0, 0, 0, 2;

    // Test computeRelativePose
    bool result = local_map.computeRelativePose(frame1, frame3, T_f1_f3, computed_cov);

    // Verify the result
    EXPECT_TRUE(result);
    EXPECT_EQ(T_f1_f3.translation(), Eigen::Vector3d(2.0, 0.0, 0.0));
    EXPECT_EQ((computed_cov-cov_theory).norm(), 0);
}