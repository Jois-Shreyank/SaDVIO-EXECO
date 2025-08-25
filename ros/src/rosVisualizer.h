#ifndef ROSVISUALIZER_H
#define ROSVISUALIZER_H

#include <cv_bridge/cv_bridge.h>
#include <opencv2/core.hpp>
#include <thread>

#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <unordered_map>
#include <vector>
#include <cstdlib>
#include <mutex>

#include "sensor_msgs/point_cloud2_iterator.hpp"
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <std_msgs/msg/header.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_broadcaster.h>
#include <visualization_msgs/msg/marker.hpp>

#include "isaeslam/data/mesh/mesh.h"
#include "isaeslam/data/mesh/global_mesh.h"
#include "isaeslam/slamCore.h"
#include "lvr2/io/ModelFactory.hpp"
#include "lvr2/types/MeshBuffer.hpp"

// ... (The convertToPointCloud2 function and the rest of the file up to runVisualizer remains the same) ...
sensor_msgs::msg::PointCloud2::SharedPtr convertToPointCloud2(const std::vector<Eigen::Vector3d> &points) {
    // Create a PointCloud2 message
    auto point_cloud_msg = std::make_shared<sensor_msgs::msg::PointCloud2>();

    // Set the header
    point_cloud_msg->header.frame_id = "world"; // Set the frame ID as needed
    point_cloud_msg->header.stamp    = rclcpp::Clock().now();

    // Set the point step and row step
    point_cloud_msg->point_step = sizeof(float) * 3;
    point_cloud_msg->row_step   = point_cloud_msg->point_step * point_cloud_msg->width;

    // Set the is_dense flag
    point_cloud_msg->is_dense = true;

    // Set the height and width of the point cloud
    point_cloud_msg->height = 1;
    point_cloud_msg->width  = points.size();

    // Set the fields of the point cloud
    sensor_msgs::PointCloud2Modifier modifier(*point_cloud_msg);
    modifier.setPointCloud2Fields(3,
                                  "x",
                                  1,
                                  sensor_msgs::msg::PointField::FLOAT32,
                                  "y",
                                  1,
                                  sensor_msgs::msg::PointField::FLOAT32,
                                  "z",
                                  1,
                                  sensor_msgs::msg::PointField::FLOAT32);
    // Resize the data vector
    point_cloud_msg->data.resize(points.size() * sizeof(float) * 3);

    // Copy the data from the Eigen vectors to the PointCloud2 message
    sensor_msgs::PointCloud2Iterator<float> iter_x(*point_cloud_msg, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(*point_cloud_msg, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(*point_cloud_msg, "z");

    for (const auto &point : points) {
        *iter_x = point.x();
        *iter_y = point.y();
        *iter_z = point.z();
        ++iter_x;
        ++iter_y;
        ++iter_z;
    }

    return point_cloud_msg;
}


class RosVisualizer : public rclcpp::Node {

  public:
    // ... (Constructor and other publish functions remain the same) ...
    RosVisualizer() : Node("slam_publisher") {
        std::cout << "\n Creation of ROS vizualizer" << std::endl;

        _pub_image_matches_in_time  = this->create_publisher<sensor_msgs::msg::Image>("image_matches_in_time", 1000);
        _pub_image_matches_in_frame = this->create_publisher<sensor_msgs::msg::Image>("image_matches_in_frame", 1000);
        _pub_image_kps              = this->create_publisher<sensor_msgs::msg::Image>("image_kps", 1000);
        _pub_vo_traj                = this->create_publisher<visualization_msgs::msg::Marker>("vo_traj", 1000);
        _pub_vo_pose                = this->create_publisher<geometry_msgs::msg::PoseStamped>("vo_pose", 1000);
        _pub_local_map_cloud        = this->create_publisher<visualization_msgs::msg::Marker>("map_local_cloud", 1000);
        _pub_local_map_cloud1       = this->create_publisher<visualization_msgs::msg::Marker>("map_local_cloud1", 1000);
        _pub_global_map_cloud       = this->create_publisher<visualization_msgs::msg::Marker>("map_global_cloud", 1000);
        _pub_local_map_lines        = this->create_publisher<visualization_msgs::msg::Marker>("map_local_lines", 1000);
        _pub_global_map_lines       = this->create_publisher<visualization_msgs::msg::Marker>("map_global_lines", 1000);
        _pub_marker                 = this->create_publisher<visualization_msgs::msg::Marker>("mesh", 1000);
        _pub_global_mesh            = this->create_publisher<visualization_msgs::msg::Marker>("global_mesh", 1000);
        _pub_cloud                  = this->create_publisher<sensor_msgs::msg::PointCloud2>("point_cloud", 1000);
        _tf_broadcaster             = std::make_shared<tf2_ros::TransformBroadcaster>(this);
        _pub_lvr_reconstructed_mesh = this->create_publisher<visualization_msgs::msg::Marker>("lvr_reconstructed_mesh", 1000);

        _vo_traj_msg.type    = visualization_msgs::msg::Marker::LINE_STRIP;
        _vo_traj_msg.color.a = 1.0;
        _vo_traj_msg.color.r = 0.0;
        _vo_traj_msg.color.g = 0.0;
        _vo_traj_msg.color.b = 1.0;
        _vo_traj_msg.scale.x = 0.05;

        // map points design
        _points_local.type    = visualization_msgs::msg::Marker::POINTS;
        _points_local.id      = 0;
        _points_local.scale.x = 0.05;
        _points_local.scale.y = 0.05;
        _points_local.color.a = 1.0;
        _points_local.color.r = 0.0;
        _points_local.color.g = 1.0;
        _points_local.color.b = 0.0;

        _points_local1.type    = visualization_msgs::msg::Marker::POINTS;
        _points_local1.id      = 0;
        _points_local1.scale.x = 0.05;
        _points_local1.scale.y = 0.05;
        _points_local1.color.a = 1.0;
        _points_local1.color.r = 1.0;
        _points_local1.color.g = 0.0;
        _points_local1.color.b = 0.0;

        _points_global.type    = visualization_msgs::msg::Marker::POINTS;
        _points_global.id      = 0;
        _points_global.scale.x = 0.05;
        _points_global.scale.y = 0.05;
        _points_global.color.a = 1.0;
        _points_global.color.r = 0.0;
        _points_global.color.g = 0.0;
        _points_global.color.b = 1.0;

        // map lines design
        _lines_local.type    = visualization_msgs::msg::Marker::LINE_LIST;
        _lines_local.id      = 4;
        _lines_local.scale.x = 0.05;
        _lines_local.scale.y = 0.05;
        _lines_local.color.a = 1.0;
        _lines_local.color.r = 1.0;
        _lines_local.color.g = 0.0;
        _lines_local.color.b = 0.0;

        _lines_global.type    = visualization_msgs::msg::Marker::LINE_LIST;
        _lines_global.id      = 4;
        _lines_global.scale.x = 0.05;
        _lines_global.scale.y = 0.05;
        _lines_global.color.a = 1.0;
        _lines_global.color.r = 0.5;
        _lines_global.color.g = 0.5;
        _lines_global.color.b = 0.5;

        // Mesh line design
        _mesh_line_list.type    = visualization_msgs::msg::Marker::TRIANGLE_LIST;
        _mesh_line_list.id      = 2;
        _mesh_line_list.color.a = 0.5;
        _mesh_line_list.scale.x = 1.0;
        _mesh_line_list.scale.y = 1.0;
        _mesh_line_list.scale.z = 1.0;

        // Design for the global mesh marker
        _global_mesh_line_list.type = visualization_msgs::msg::Marker::TRIANGLE_LIST;
        _global_mesh_line_list.id = 3;
        _global_mesh_line_list.color.a = 0.5;
        _global_mesh_line_list.color.r = 0.0;
        _global_mesh_line_list.color.g = 0.0;
        _global_mesh_line_list.color.b = 1.0;
        _global_mesh_line_list.scale.x = 1.0;
        _global_mesh_line_list.scale.y = 1.0;
        _global_mesh_line_list.scale.z = 1.0;
        
        // Design for the LVR2 reconstructed mesh
        _lvr_mesh_marker.type = visualization_msgs::msg::Marker::TRIANGLE_LIST;
        _lvr_mesh_marker.id = 5; // Use a unique ID
        _lvr_mesh_marker.header.frame_id = "world";
        _lvr_mesh_marker.pose.orientation.w = 1.0;
        _lvr_mesh_marker.scale.x = 1.0;
        _lvr_mesh_marker.scale.y = 1.0;
        _lvr_mesh_marker.scale.z = 1.0;
        _lvr_mesh_marker.color.a = 0.9;
        _lvr_mesh_marker.color.r = 0.2;
        _lvr_mesh_marker.color.g = 1.0;
        _lvr_mesh_marker.color.b = 0.2;
    }
    // ... (All publish functions, drawMatchesTopBottom, etc. are unchanged)
    void drawMatchesTopBottom(cv::Mat Itop,
                              std::vector<cv::KeyPoint> kp_top,
                              cv::Mat Ibottom,
                              std::vector<cv::KeyPoint> kp_bottom,
                              std::vector<cv::DMatch> m,
                              cv::Mat &resultImg) {

        uint H = Itop.rows;

        // rotate images 90°
        cv::rotate(Itop, Itop, cv::ROTATE_90_CLOCKWISE);
        cv::rotate(Ibottom, Ibottom, cv::ROTATE_90_CLOCKWISE);

        // change kp coords
        std::vector<cv::KeyPoint> kp_top2, kp_bottom2;
        for (auto &k : kp_top)
            kp_top2.push_back(cv::KeyPoint(H - k.pt.y, k.pt.x, 1));
        for (auto &k : kp_bottom)
            kp_bottom2.push_back(cv::KeyPoint(H - k.pt.y, k.pt.x, 1));

        drawMatches(Itop,
                    kp_top2,
                    Ibottom,
                    kp_bottom2,
                    m,
                    resultImg,
                    cv::Scalar::all(-1),
                    cv::Scalar::all(-1),
                    std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

        cv::rotate(resultImg, resultImg, cv::ROTATE_90_COUNTERCLOCKWISE);
    }

    void publishImage(const std::shared_ptr<isae::Frame> frame) {
        std_msgs::msg::Header header;
        header.frame_id = "world";
        header.stamp    = rclcpp::Node::now();
        // Display keypoints
        cv::Mat img_2_pub;
        cv::cvtColor(frame->getSensors().at(0)->getRawData(), img_2_pub, CV_GRAY2RGB);

        for (const auto &feat : frame->getSensors().at(0)->getFeatures()["pointxd"]) {
            cv::Scalar col;

            if (feat->getLandmark().lock() == nullptr) {
                col = cv::Scalar(0, 0, 255);
            } else if (feat->getLandmark().lock()->isResurected()) {
                col = cv::Scalar(0, 255, 0);
            } else {
                if (feat->getLandmark().lock()->isInitialized())
                    col = cv::Scalar(255, 0, 0);
            }
            Eigen::Vector2d pt2d = feat->getPoints().at(0);

            cv::circle(img_2_pub, cv::Point(pt2d.x(), pt2d.y()), 4, col, -1);
        }

        for (const auto &feat : frame->getSensors().at(0)->getFeatures()["edgeletxd"]) {
            cv::Scalar col;

            if (feat->getLandmark().lock() == nullptr) {
                col = cv::Scalar(0, 255, 0);
            } else {
                col = cv::Scalar(255, 255, 0);
            }
            Eigen::Vector2d pt2d  = feat->getPoints().at(0);
            Eigen::Vector2d pt2d2 = feat->getPoints().at(1);
            Eigen::Vector2d delta = 10 * (pt2d2 - pt2d);

            cv::circle(img_2_pub, cv::Point(pt2d.x(), pt2d.y()), 4, col, -1);
            cv::line(img_2_pub,
                     cv::Point(pt2d.x() - delta.x(), pt2d.y() - delta.y()),
                     cv::Point(pt2d2.x() + delta.x(), pt2d2.y() + delta.y()),
                     col,
                     1);
        }

        for (const auto &feat : frame->getSensors().at(0)->getFeatures()["linexd"]) {
            cv::Scalar col;

            if (feat->getLandmark().lock() == nullptr) {
                col = cv::Scalar(255, 0, 0);
            } else {
                col = cv::Scalar(255, 0, 255);
            }
            Eigen::Vector2d pt2d  = feat->getPoints().at(0);
            Eigen::Vector2d pt2d2 = feat->getPoints().at(1);
            cv::line(img_2_pub, cv::Point(pt2d.x(), pt2d.y()), cv::Point(pt2d2.x(), pt2d2.y()), col, 2);
        }

        auto img_kps_msg = cv_bridge::CvImage(header, "rgb8", img_2_pub).toImageMsg();

        _pub_image_kps->publish(*img_kps_msg.get());
    }

    void publishMatches(const isae::typed_vec_match matches, bool in_time) {

        std_msgs::msg::Header header;
        header.frame_id = "world";
        header.stamp    = rclcpp::Node::now();

        // Display keypoints
        cv::Mat img_matches;
        cv::Mat img_2_pub_line, img_2_pub_line_2, img_2_pub_pts;

        for (const auto &tmatches : matches) {
            if (tmatches.second.size() == 0)
                continue;

            if (img_matches.empty())
                cv::cvtColor(tmatches.second.at(0).first->getSensor()->getRawData(), img_matches, CV_GRAY2RGB);

            if (tmatches.first == "pointxd") {

                for (const auto &match : tmatches.second) {
                    cv::Scalar colpt   = cv::Scalar(255, 0, 0);
                    cv::Scalar colline = cv::Scalar(0, 255, 0);

                    Eigen::Vector2d pt2d1 = match.first->getPoints().at(0);
                    cv::circle(img_matches, cv::Point(pt2d1.x(), pt2d1.y()), 4, colpt, -1);
                    Eigen::Vector2d pt2d2 = match.second->getPoints().at(0);
                    cv::circle(img_matches, cv::Point(pt2d2.x(), pt2d2.y()), 4, colpt, -1);
                    cv::line(img_matches, cv::Point(pt2d1.x(), pt2d1.y()), cv::Point(pt2d2.x(), pt2d2.y()), colline, 2);
                }
            }

            if (tmatches.first == "linexd") {
                for (const auto &match : tmatches.second) {
                    // Display first feature
                    cv::Scalar colpt        = cv::Scalar(0, 0, 255);
                    cv::Scalar colline      = cv::Scalar(0, 0, 255);
                    cv::Scalar colline2     = cv::Scalar(0, 255, 255);
                    cv::Scalar collinematch = cv::Scalar(255, 0, 255);

                    Eigen::Vector2d pt2d  = match.first->getPoints().at(0);
                    Eigen::Vector2d pt2d2 = match.first->getPoints().at(1);
                    cv::circle(img_matches, cv::Point(pt2d.x(), pt2d.y()), 4, colpt, -1);
                    cv::circle(img_matches, cv::Point(pt2d2.x(), pt2d2.y()), 4, colpt, -1);
                    cv::line(img_matches, cv::Point(pt2d.x(), pt2d.y()), cv::Point(pt2d2.x(), pt2d2.y()), colline, 2);

                    // Display second feature
                    Eigen::Vector2d pt2d_2  = match.second->getPoints().at(0);
                    Eigen::Vector2d pt2d2_2 = match.second->getPoints().at(1);
                    cv::circle(img_matches, cv::Point(pt2d_2.x(), pt2d_2.y()), 4, colpt, -1);
                    cv::circle(img_matches, cv::Point(pt2d2_2.x(), pt2d2_2.y()), 4, colpt, -1);
                    cv::line(img_matches,
                             cv::Point(pt2d_2.x(), pt2d_2.y()),
                             cv::Point(pt2d2_2.x(), pt2d2_2.y()),
                             colline2,
                             2);

                    // Display matching line between the centers
                    cv::line(img_matches,
                             0.5 * (cv::Point(pt2d_2.x(), pt2d_2.y()) + cv::Point(pt2d2_2.x(), pt2d2_2.y())),
                             0.5 * (cv::Point(pt2d.x(), pt2d.y()) + cv::Point(pt2d2.x(), pt2d2.y())),
                             collinematch,
                             2);
                }
            }
        }

        auto imgTrackMsg = cv_bridge::CvImage(header, "rgb8", img_matches).toImageMsg();

        // Choose the good publisher if it is tracked in frame or in time
        if (in_time) {
            _pub_image_matches_in_time->publish(*imgTrackMsg.get());
        } else {
            _pub_image_matches_in_frame->publish(*imgTrackMsg.get());
        }
    }

    void publishFrame(const std::shared_ptr<isae::Frame> frame) {

        geometry_msgs::msg::PoseStamped Twc_msg;
        Twc_msg.header.stamp    = rclcpp::Time(frame->getTimestamp());
        Twc_msg.header.frame_id = "world";

        // Deal with position
        geometry_msgs::msg::Point p;
        const Eigen::Vector3d twc = frame->getFrame2WorldTransform().translation();
        p.x                       = twc.x();
        p.y                       = twc.y();
        p.z                       = twc.z();
        Twc_msg.pose.position     = p;

        // Deal with orientation
        geometry_msgs::msg::Quaternion q;
        const Eigen::Quaterniond eigen_q = (Eigen::Quaterniond)frame->getFrame2WorldTransform().linear();
        q.x                              = eigen_q.x();
        q.y                              = eigen_q.y();
        q.z                              = eigen_q.z();
        q.w                              = eigen_q.w();
        Twc_msg.pose.orientation         = q;

        // Publish transform
        geometry_msgs::msg::TransformStamped Twc_tf;
        Twc_tf.header.stamp            = rclcpp::Time(frame->getTimestamp());
        Twc_tf.header.frame_id         = "world";
        Twc_tf.child_frame_id          = "robot";
        Twc_tf.transform.translation.x = twc.x();
        Twc_tf.transform.translation.y = twc.y();
        Twc_tf.transform.translation.z = twc.z();
        Twc_tf.transform.rotation      = Twc_msg.pose.orientation;
        _tf_broadcaster->sendTransform(Twc_tf);

        // publish messages
        _pub_vo_pose->publish(Twc_msg);
    }

    void publishLocalMap(const std::shared_ptr<isae::LocalMap> map) {

        _vo_traj_msg.header.stamp    = rclcpp::Node::now();
        _vo_traj_msg.header.frame_id = "world";
        _vo_traj_msg.points.clear();
        geometry_msgs::msg::Point p;

        for (auto &frame : map->getOldFramesPoses()) {
            const Eigen::Vector3d twc = frame.translation();
            p.x                       = twc.x();
            p.y                       = twc.y();
            p.z                       = twc.z();
            _vo_traj_msg.points.push_back(p);
        }

        for (auto &frame : map->getFrames()) {
            const Eigen::Vector3d twc = frame->getFrame2WorldTransform().translation();
            p.x                       = twc.x();
            p.y                       = twc.y();
            p.z                       = twc.z();
            _vo_traj_msg.points.push_back(p);
        }

        // publish message
        _pub_vo_traj->publish(_vo_traj_msg);
    }

    void publishLocalMapCloud(const std::shared_ptr<isae::LocalMap> map, const bool no_fov_mode = false) {
        isae::typed_vec_landmarks ldmks = map->getLandmarks();

        _points_local.header.frame_id    = "world";
        _points_local.header.stamp       = rclcpp::Node::now();
        _points_local.action             = visualization_msgs::msg::Marker::ADD;
        _points_local.pose.orientation.w = 1.0;

        _points_local1.header.frame_id    = "world";
        _points_local1.header.stamp       = rclcpp::Node::now();
        _points_local1.action             = visualization_msgs::msg::Marker::ADD;
        _points_local1.pose.orientation.w = 1.0;

        // build the point cloud from point3D lmks
        _points_local.points.clear();
        _points_local1.points.clear();

        for (auto &l : ldmks["pointxd"]) {
            if (l->isOutlier())
                continue;
            Eigen::Vector3d pt3d = l->getPose().translation();

            geometry_msgs::msg::Point pt;
            pt.x = pt3d.x();
            pt.y = pt3d.y();
            pt.z = pt3d.z();

            if (no_fov_mode) {
                if (l->getFeatures().at(0).lock()->getSensor() ==
                    l->getFeatures().at(0).lock()->getSensor()->getFrame()->getSensors().at(1))
                    _points_local1.points.push_back(pt);
                else
                    _points_local.points.push_back(pt);
            } else
                _points_local.points.push_back(pt);
        }

        _pub_local_map_cloud1->publish(_points_local1);
        _pub_local_map_cloud->publish(_points_local);

        _lines_local.header.frame_id    = "world";
        _lines_local.header.stamp       = rclcpp::Node::now();
        _lines_local.action             = visualization_msgs::msg::Marker::ADD;
        _lines_local.pose.orientation.w = 1.0;

        // build the point cloud from line3D lmks
        _lines_local.points.clear();
        for (auto &l : ldmks["linexd"]) {
            if (l->isOutlier())
                continue;
            Eigen::Affine3d T_w_ldmk                = l->getPose();
            std::vector<Eigen::Vector3d> ldmk_model = l->getModelPoints();
            for (const auto &p3d_model : ldmk_model) {
                // conversion to the world coordinate system
                Eigen::Vector3d t_w_lmk = T_w_ldmk * p3d_model.cwiseProduct(l->getScale());
                geometry_msgs::msg::Point pt;
                pt.x = t_w_lmk.x();
                pt.y = t_w_lmk.y();
                pt.z = t_w_lmk.z();
                _lines_local.points.push_back(pt);
            }
        }

        _pub_local_map_lines->publish(_lines_local);
    }

    void publishGlobalMapCloud(const std::shared_ptr<isae::GlobalMap> map) {
        isae::typed_vec_landmarks ldmks = map->getLandmarks();
        
        RCLCPP_INFO(this->get_logger(),
            "GlobalMap contains %zu point landmarks and %zu line landmarks",
            ldmks["pointxd"].size(),
            ldmks["linexd"].size());

        _points_global.header.frame_id    = "world";
        _points_global.header.stamp       = rclcpp::Node::now();
        _points_global.action             = visualization_msgs::msg::Marker::ADD;
        _points_global.pose.orientation.w = 1.0;

        // build the point cloud from point3D lmks
        _points_global.points.clear();
        for (auto &l : ldmks["pointxd"]) {
            if (l->isOutlier())
                continue;
            Eigen::Vector3d pt3d = l->getPose().translation();
            geometry_msgs::msg::Point pt;
            pt.x = pt3d.x();
            pt.y = pt3d.y();
            pt.z = pt3d.z();
            _points_global.points.push_back(pt);
        }

        _pub_global_map_cloud->publish(_points_global);

        _lines_global.header.frame_id    = "world";
        _lines_global.header.stamp       = rclcpp::Node::now();
        _lines_global.action             = visualization_msgs::msg::Marker::ADD;
        _lines_global.pose.orientation.w = 1.0;

        // build the point cloud from line3D lmks
        _lines_global.points.clear();
        for (auto &l : ldmks["linexd"]) {
            if (l->isOutlier())
                continue;
            Eigen::Affine3d T_w_ldmk                = l->getPose();
            std::vector<Eigen::Vector3d> ldmk_model = l->getModelPoints();
            for (const auto &p3d_model : ldmk_model) {
                // conversion to the world coordinate system
                Eigen::Vector3d t_w_lmk = T_w_ldmk * p3d_model.cwiseProduct(l->getScale());
                geometry_msgs::msg::Point pt;
                pt.x = t_w_lmk.x();
                pt.y = t_w_lmk.y();
                pt.z = t_w_lmk.z();
                _lines_global.points.push_back(pt);
            }
        }

        _pub_global_map_lines->publish(_lines_global);
    }

    void publishMesh(const std::shared_ptr<isae::Mesh3D> mesh) {

        _mesh_line_list.points.clear();
        _mesh_line_list.colors.clear();

        _mesh_line_list.header.frame_id = "world";
        _mesh_line_list.header.stamp    = rclcpp::Node::now();

        for (auto &polygon : mesh->getPolygonVector()) {

            // Handles the points of the polygon
            std::vector<geometry_msgs::msg::Point> p_vector;
            std::vector<std_msgs::msg::ColorRGBA> c_vector;
            for (auto &vertex : polygon->getVertices()) {
                geometry_msgs::msg::Point p;
                std_msgs::msg::ColorRGBA color;
                Eigen::Vector3d lmk_coord = vertex->getVertexPosition();

                p.x = lmk_coord.x();
                p.y = lmk_coord.y();
                p.z = lmk_coord.z();
                p_vector.push_back(p);

                // Color triangle with its slope
                double trav_score = polygon->getPolygonNormal().dot(Eigen::Vector3d(0, 0, 1));
                color.r           = (1 - trav_score);
                color.g           = trav_score;
                color.b           = 0;
                color.a           = 1.0;

                c_vector.push_back(color);
            }

            // Set the lines of the polygon
            _mesh_line_list.points.push_back(p_vector.at(0));
            _mesh_line_list.colors.push_back(c_vector.at(0));
            _mesh_line_list.points.push_back(p_vector.at(1));
            _mesh_line_list.colors.push_back(c_vector.at(1));
            _mesh_line_list.points.push_back(p_vector.at(2));
            _mesh_line_list.colors.push_back(c_vector.at(2));
        }

        _pub_marker->publish(_mesh_line_list);

        // Publish the dense point cloud from the mesh 3D
        std::vector<Eigen::Vector3d> pt_cloud = mesh->getPointCloud();
        if (pt_cloud.empty())
            return;

        sensor_msgs::msg::PointCloud2::SharedPtr pc2_msg_ = std::make_shared<sensor_msgs::msg::PointCloud2>();
        pc2_msg_                                          = convertToPointCloud2(pt_cloud);

        _pub_cloud->publish(*pc2_msg_);
    }

    void publishGlobalMesh(const std::shared_ptr<isae::GlobalMesh> mesh) {
        _global_mesh_line_list.points.clear();
        _global_mesh_line_list.colors.clear();

        _global_mesh_line_list.header.frame_id = "world";
        _global_mesh_line_list.header.stamp = rclcpp::Node::now();

        for (auto &polygon : mesh->getPolygonVector()) {
            std::vector<geometry_msgs::msg::Point> p_vector;
            std::vector<std_msgs::msg::ColorRGBA> c_vector;
            for (auto &vertex : polygon->getVertices()) {
                geometry_msgs::msg::Point p;
                Eigen::Vector3d lmk_coord = vertex->getVertexPosition();
                p.x = lmk_coord.x();
                p.y = lmk_coord.y();
                p.z = lmk_coord.z();
                p_vector.push_back(p);

                std_msgs::msg::ColorRGBA color;
                double trav_score = polygon->getPolygonNormal().dot(Eigen::Vector3d(0, 0, 1));
                color.r = (1 - trav_score); 
                color.g = 0;       
                color.b = trav_score;
                color.a = 1.0;              
                c_vector.push_back(color);
            }
            _global_mesh_line_list.points.push_back(p_vector.at(0));
            _global_mesh_line_list.colors.push_back(c_vector.at(0));
            _global_mesh_line_list.points.push_back(p_vector.at(1));
            _global_mesh_line_list.colors.push_back(c_vector.at(1));
            _global_mesh_line_list.points.push_back(p_vector.at(2));
            _global_mesh_line_list.colors.push_back(c_vector.at(2));
        }
    _pub_global_mesh->publish(_global_mesh_line_list);
    }

    void runVisualizer(std::shared_ptr<isae::SLAMCore> SLAM) {
        _slam_core = SLAM;
        
        if (!std::filesystem::exists(_log_path)) {
            RCLCPP_INFO(this->get_logger(), "Creating logging directory: %s", _log_path.c_str());
            std::filesystem::create_directories(_log_path);
        }

        auto last_save_time = std::chrono::steady_clock::now();

        while (true) {

            auto current_time = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::seconds>(current_time - last_save_time).count() >= 10) {
                saveDataCallback();
                last_save_time = current_time; 
            }

            if (SLAM->_frame_to_display) {
                publishImage(SLAM->_frame_to_display);
                publishFrame(SLAM->_frame_to_display);
                SLAM->_frame_to_display.reset();
            }

            if (SLAM->_local_map_to_display) {
                publishLocalMap(SLAM->_local_map_to_display);
                publishLocalMapCloud(SLAM->_local_map_to_display);
                SLAM->_local_map_to_display.reset();
            }

            if (SLAM->_mesh_to_display) {
                publishMesh(SLAM->_mesh_to_display);
                SLAM->_mesh_to_display.reset();
            }

            if (SLAM->_global_mesh_to_display) {
                publishGlobalMesh(SLAM->_global_mesh_to_display);
                // MODIFICATION: Latch the latest valid mesh pointer before it's reset
                {
                    std::lock_guard<std::mutex> lock(_map_mutex);
                    _latest_global_mesh = SLAM->_global_mesh_to_display;
                }
                SLAM->_global_mesh_to_display.reset();
            }

            if (SLAM->_global_map_to_display) {
                publishGlobalMapCloud(SLAM->_global_map_to_display);
                {
                    std::lock_guard<std::mutex> lock(_map_mutex);
                    _latest_global_map = SLAM->_global_map_to_display;
                }
                SLAM->_global_map_to_display.reset();
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr _pub_marker, _pub_vo_traj, _pub_global_map_cloud,
        _pub_local_map_cloud, _pub_local_map_cloud1, _pub_global_map_lines, _pub_local_map_lines, _pub_global_mesh;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr _pub_image_kps, _pub_image_matches_in_time,
        _pub_image_matches_in_frame;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr _pub_cloud;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr _pub_vo_pose;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr _pub_lvr_reconstructed_mesh;
    std::shared_ptr<tf2_ros::TransformBroadcaster> _tf_broadcaster;
    visualization_msgs::msg::Marker _vo_traj_msg;
    visualization_msgs::msg::Marker _points_local, _points_global, _points_local1, _lines_local, _lines_global;
    visualization_msgs::msg::Marker _mesh_line_list;
    visualization_msgs::msg::Marker _global_mesh_line_list;
    visualization_msgs::msg::Marker _lvr_mesh_marker;

private:
    void reconstructAndPublishLVRMesh(const std::string& cloud_filename, const std::string& mesh_filename)
    {
        // // Define file paths
        //std::string cloud_filename = _log_path + "global_cloud.ply";
        //std::string mesh_filename = _log_path + "lvr_reconstructed_mesh.ply";

        // // --- 1. Save the point cloud from the global map ---
        // savePointCloudToPly(map, cloud_filename);
        // RCLCPP_INFO(this->get_logger(), "Saved point cloud to %s for LVR reconstruction", cloud_filename.c_str());

        // // --- 2. Execute lvr2_reconstruct ---
        // // IMPORTANT: Replace this with the absolute path to YOUR lvr2_reconstruct executable!
        static std::atomic_bool recon_running{false};
        if (recon_running.exchange(true)) {
            RCLCPP_WARN(this->get_logger(), "LVR reconstruction already running; skipping this cycle.");
            return;
        }

        auto reset_flag = std::unique_ptr<void, std::function<void(void*)>>(
            (void*)1, [&](void*){ recon_running = false; });

        const std::string lvr2_executable_path = "/root/lvr2/build/bin/lvr2_reconstruct";

        std::filesystem::path out_path(mesh_filename);
        std::filesystem::path out_dir  = out_path.parent_path();
        std::filesystem::path out_name = out_path.filename();

        // Ensure directory exists
        std::error_code ec;
        std::filesystem::create_directories(out_dir, ec);
        if (ec) {
            RCLCPP_ERROR(this->get_logger(), "Failed to create output dir %s: %s",
                     out_dir.string().c_str(), ec.message().c_str());
            return;
        }

        std::stringstream command;
        command << "cd " << out_dir.string()
                << " && " << lvr2_executable_path
                << " --inputFile " << cloud_filename
                << " --outputFile " << out_name.string()
                << " -v 0.7"; // Your specified voxel size

        RCLCPP_INFO(this->get_logger(), "Executing LVR reconstruction: %s", command.str().c_str());
        int return_code = std::system(command.str().c_str());

        if (return_code != 0) {
            RCLCPP_ERROR(this->get_logger(), "lvr2_reconstruct failed with return code %d", return_code);
            return;
        }

        RCLCPP_INFO(this->get_logger(), "LVR reconstruction successful. Loading mesh from %s", mesh_filename.c_str());

        // --- 3. Load the newly created mesh ---
        lvr2::ModelPtr loaded_model = lvr2::ModelFactory::readModel(mesh_filename);
        if (!loaded_model || !loaded_model->m_mesh) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load the reconstructed mesh file: %s", mesh_filename.c_str());
            return;
        }

        lvr2::MeshBufferPtr mesh = loaded_model->m_mesh;
        lvr2::floatArr vertices = mesh->getVertices();
        lvr2::indexArray face_indices = mesh->getFaceIndices();
        size_t num_faces = mesh->numFaces();

        if (!vertices || !face_indices || num_faces == 0)
        {
            RCLCPP_ERROR(this->get_logger(), "Loaded mesh is missing vertex or face data.");
            return;
        }

        // --- 4. Convert mesh to a ROS Marker and publish ---
        _lvr_mesh_marker.points.clear();
        _lvr_mesh_marker.header.stamp = this->now();
        
        for (size_t i = 0; i < num_faces; i ++)
        {
            
            // {
            //     geometry_msgs::msg::Point p;
            //     size_t vertex_idx = face_indices[i + j];
            //     p.x = vertices[vertex_idx].x;
            //     p.y = vertices[vertex_idx].y;
            //     p.z = vertices[vertex_idx].z;
            //     _lvr_mesh_marker.points.push_back(p);
            // }

             // The face_indices array is flat, so we access it like this:
            size_t idx1 = face_indices[i * 3 + 0];
            size_t idx2 = face_indices[i * 3 + 1];
            size_t idx3 = face_indices[i * 3 + 2];

            // Define the three vertices of the triangle
            geometry_msgs::msg::Point p1, p2, p3;

            // The vertices array is also flat (x, y, z, x, y, z, ...)
            p1.x = vertices[idx1 * 3 + 0];
            p1.y = vertices[idx1 * 3 + 1];
            p1.z = vertices[idx1 * 3 + 2];

            p2.x = vertices[idx2 * 3 + 0];
            p2.y = vertices[idx2 * 3 + 1];
            p2.z = vertices[idx2 * 3 + 2];

            p3.x = vertices[idx3 * 3 + 0];
            p3.y = vertices[idx3 * 3 + 1];
            p3.z = vertices[idx3 * 3 + 2];

            // Add the three vertices to the marker's point list
            _lvr_mesh_marker.points.push_back(p1);
            _lvr_mesh_marker.points.push_back(p2);
            _lvr_mesh_marker.points.push_back(p3);
        }

        _pub_lvr_reconstructed_mesh->publish(_lvr_mesh_marker);
        RCLCPP_INFO(this->get_logger(), "Published LVR reconstructed mesh to RViz.");
    }

    void saveDataCallback() {
        RCLCPP_INFO(this->get_logger(), "Periodic save triggered.");

        // MODIFICATION: Create local copies of the shared pointers to use for saving.
        // This minimizes the time we hold the mutex lock.
        std::shared_ptr<isae::GlobalMap> map_to_save;
        std::shared_ptr<isae::GlobalMesh> mesh_to_save;
        {
            std::lock_guard<std::mutex> lock(_map_mutex);
            map_to_save = _latest_global_map;
            mesh_to_save = _latest_global_mesh;
        }

        // Generate a timestamp for unique filenames
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d_%H-%M-%S");
        std::string timestamp = ss.str();

        // MODIFICATION: Check the latched pointers, not the volatile ones from _slam_core
        if (map_to_save) {
            std::string cloud_filename = _log_path + "global_cloud.ply";
            std::string lvr_mesh_filename = _log_path + "lvr_reconstructed_mesh.ply";
            savePointCloudToPly(map_to_save, cloud_filename);
            reconstructAndPublishLVRMesh(cloud_filename, lvr_mesh_filename);
        } else {
            RCLCPP_INFO(this->get_logger(), "No global map available to save.");
        }

        if (mesh_to_save) {
            std::string mesh_filename = _log_path + "global_mesh.ply";
            saveMeshToPly(mesh_to_save, mesh_filename);
        } else {
            RCLCPP_INFO(this->get_logger(), "No global mesh available to save.");
        }
    }

    // ... (savePointCloudToPly and saveMeshToPly remain the same) ...
    void savePointCloudToPly(const std::shared_ptr<isae::GlobalMap> map, const std::string &filename) {
        isae::typed_vec_landmarks ldmks = map->getLandmarks();
        auto& points = ldmks["pointxd"];

        std::vector<Eigen::Vector3d> valid_pts;

        valid_pts.reserve(points.size());
        for (const auto& l : points) {
            if (l->isOutlier()) continue;
            valid_pts.push_back(l->getPose().translation());
        }

        if (valid_pts.empty()) {
            RCLCPP_INFO(this->get_logger(), "Point cloud is empty. Skipping save for %s.", filename.c_str());
            return;
        }

        std::ofstream file(filename);
        if (!file.is_open()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open file for writing: %s", filename.c_str());
            return;
        }

        // Write PLY header
        file << "ply\n";
        file << "format ascii 1.0\n";
        file << "comment Generated by RosVisualizer\n";
        file << "element vertex " << valid_pts.size() << "\n";
        file << "property float x\n";
        file << "property float y\n";
        file << "property float z\n";
        file << "end_header\n";

        // Write points data
        for (const auto &p : valid_pts) {
            file << p.x() << " " << p.y() << " " << p.z() << "\n";
        }
        file.close();
        RCLCPP_INFO(this->get_logger(), "Successfully saved point cloud to %s", filename.c_str());
    }
    void saveMeshToPly(const std::shared_ptr<isae::GlobalMesh> mesh, const std::string &filename) {
        const auto& polygons = mesh->getPolygonVector();
        if (polygons.empty()) {
            RCLCPP_INFO(this->get_logger(), "Mesh is empty. Skipping save for %s.", filename.c_str());
            return;
        }

        std::ofstream file(filename);
        if (!file.is_open()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open file for writing: %s", filename.c_str());
            return;
        }

        // Map unique vertices to an index to define faces correctly
        std::vector<std::shared_ptr<isae::Vertex>> unique_vertices;
        std::unordered_map<std::shared_ptr<isae::Vertex>, int> vertex_to_index;

        for (const auto& poly : polygons) {
            for (const auto& vertex : poly->getVertices()) {
                if (vertex_to_index.find(vertex) == vertex_to_index.end()) {
                    vertex_to_index[vertex] = unique_vertices.size();
                    unique_vertices.push_back(vertex);
                }
            }
        }

        // Write PLY header
        file << "ply\n";
        file << "format ascii 1.0\n";
        file << "comment Generated by RosVisualizer\n";
        file << "element vertex " << unique_vertices.size() << "\n";
        file << "property float x\n";
        file << "property float y\n";
        file << "property float z\n";
        file << "element face " << polygons.size() << "\n";
        file << "property list uchar int vertex_indices\n";
        file << "end_header\n";

        // Write vertex data
        for (const auto& vertex : unique_vertices) {
            Eigen::Vector3d pos = vertex->getVertexPosition();
            file << pos.x() << " " << pos.y() << " " << pos.z() << "\n";
        }

        // Write face data (assuming triangles)
        for (const auto& poly : polygons) {
            const auto& vertices = poly->getVertices();
            if (vertices.size() == 3) {
                file << "3 " << vertex_to_index[vertices[0]] << " "
                     << vertex_to_index[vertices[1]] << " "
                     << vertex_to_index[vertices[2]] << "\n";
            }
        }

        file.close();
        RCLCPP_INFO(this->get_logger(), "Successfully saved mesh to %s", filename.c_str());
    }

    std::shared_ptr<isae::SLAMCore> _slam_core;
    std::string _log_path = "/root/Cosys-AirSim-EXECO/ros2/log_slam/";

    // MODIFICATION: Add member variables to latch the latest map and mesh data
    std::shared_ptr<isae::GlobalMap> _latest_global_map;
    std::shared_ptr<isae::GlobalMesh> _latest_global_mesh;
    std::mutex _map_mutex;

};

// } // namespace isae

#endif // ROSVISUALIZER_H