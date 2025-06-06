#ifndef IMU_H
#define IMU_H

#include "isaeslam/data/sensors/ASensor.h"

namespace isae {

static Eigen::Vector3d g(0, 0, -9.81);

/*!
 * @brief Config strucuture for an IMU.
 */
struct imu_config : sensor_config {
    double gyr_noise;
    double bgyr_noise;
    double acc_noise;
    double bacc_noise;
    double rate_hz;
    double dt_imu_cam;
};

/*!
 * @brief An Inertial Measurement Unit (IMU) sensor class.
 *
 * This class represents an IMU sensor that provides acceleration and gyroscope measurements.
 * It handles the preintegration of IMU data for use in SLAM systems.
 */
class IMU : public ASensor {
  public:
    IMU(Eigen::Vector3d acc, Eigen::Vector3d gyr) : ASensor("imu"), _acc(acc), _gyr(gyr) {
        _v     = Eigen::Vector3d::Zero();
        _ba    = Eigen::Vector3d::Zero();
        _bg    = Eigen::Vector3d::Zero();
        _Sigma = Eigen::Matrix<double, 9, 9>::Zero();
    }
    IMU(std::shared_ptr<imu_config> config, Eigen::Vector3d acc, Eigen::Vector3d gyr)
        : ASensor("imu"), _acc(acc), _gyr(gyr) {
        _v          = Eigen::Vector3d::Zero();
        _gyr_noise  = config->gyr_noise;
        _acc_noise  = config->acc_noise;
        _bgyr_noise = config->bgyr_noise;
        _bacc_noise = config->bacc_noise;
        _rate_hz    = config->rate_hz;
        _Sigma      = Eigen::Matrix<double, 9, 9>::Zero();
        _ba         = Eigen::Vector3d::Zero();
        _bg         = Eigen::Vector3d::Zero();
        _delta_R    = Eigen::Matrix3d::Identity();

        _eta << _gyr_noise, _gyr_noise, _gyr_noise, _acc_noise, _acc_noise, _acc_noise;
        _eta = _eta.cwiseAbs2();
        _eta = _eta * _rate_hz;
    }
    ~IMU() {}

    Eigen::Vector3d getAcc() { return _acc; }
    Eigen::Vector3d getGyr() { return _gyr; }

    void setBa(Eigen::Vector3d ba) {
        std::lock_guard<std::mutex> lock(_imu_mtx);
        _ba = ba;
    }
    void setBg(Eigen::Vector3d bg) {
        std::lock_guard<std::mutex> lock(_imu_mtx);
        _bg = bg;
    }
    void setDeltaP(const Eigen::Vector3d delta_p) {
        std::lock_guard<std::mutex> lock(_imu_mtx);
        _delta_p = delta_p;
    }
    void setDeltaV(const Eigen::Vector3d delta_v) {
        std::lock_guard<std::mutex> lock(_imu_mtx);
        _delta_v = delta_v;
    }
    void setDeltaR(const Eigen::Matrix3d delta_R) {
        std::lock_guard<std::mutex> lock(_imu_mtx);
        _delta_R = delta_R;
    }
    void setVelocity(const Eigen::Vector3d v) {
        std::lock_guard<std::mutex> lock(_imu_mtx);
        _v = v;
    }
    Eigen::Vector3d getBa() {
        std::lock_guard<std::mutex> lock(_imu_mtx);
        return _ba;
    }
    Eigen::Vector3d getBg() {
        std::lock_guard<std::mutex> lock(_imu_mtx);
        return _bg;
    }
    Eigen::Vector3d getDeltaP() const {
        std::lock_guard<std::mutex> lock(_imu_mtx);
        return _delta_p;
    }
    Eigen::Vector3d getDeltaV() const {
        std::lock_guard<std::mutex> lock(_imu_mtx);
        return _delta_v;
    }
    Eigen::Matrix3d getDeltaR() const {
        std::lock_guard<std::mutex> lock(_imu_mtx);
        return _delta_R;
    }
    Eigen::Vector3d getVelocity() const {
        std::lock_guard<std::mutex> lock(_imu_mtx);
        return _v;
    }
    Eigen::MatrixXd getCov() const {
        std::lock_guard<std::mutex> lock(_imu_mtx);
        return _Sigma;
    }
    double getGyrNoise() const { return _gyr_noise; }
    double getAccNoise() const { return _acc_noise; }
    double getbGyrNoise() const { return _bgyr_noise; }
    double getbAccNoise() const { return _bacc_noise; }

    void setLastKF(std::shared_ptr<Frame> frame) {
        std::lock_guard<std::mutex> lock(_imu_mtx);
        _last_kf = frame;
    }
    std::shared_ptr<Frame> getLastKF() {
        std::lock_guard<std::mutex> lock(_imu_mtx);
        if (_last_kf.lock())
            return _last_kf.lock();
        else
            return nullptr;
    }
    void setLastIMU(std::shared_ptr<IMU> imu) {
        std::lock_guard<std::mutex> lock(_imu_mtx);
        _last_IMU = imu;
    }
    std::shared_ptr<IMU> getLastIMU() {
        std::lock_guard<std::mutex> lock(_imu_mtx);
        return _last_IMU;
    }

    /*!
     * @brief Process IMU data to compute pre integration deltas and covariances.
     *
     * This function processes the IMU data to compute the deltas in position, velocity, and orientation on SO(3)
     * It is based on "On-Manifold Preintegration for Real-Time Visual-Inertial Odometry" by Forster et al.
     * Source: https://arxiv.org/abs/1512.02363
     */
    bool processIMU();

    /*!
     * @brief Estimate the transformation between the Last KF and the current frame with pre integration deltas.
     */
    void estimateTransformIMU(Eigen::Affine3d &dT);

    /*!
     * @brief  Update deltas with biases variations
     */
    void biasDeltaCorrection(Eigen::Vector3d d_ba, Eigen::Vector3d d_bg);

    /*!
     * @brief Update biases w.r.t the previous KF (e.g. after optimization)
     */
    void updateBiases();

    Eigen::Matrix3d _J_dR_bg; //!> Jacobian of the delta rotation w.r.t the gyro bias
    Eigen::Matrix3d _J_dv_ba; //!> Jacobian of the delta velocity w.r.t the accel bias
    Eigen::Matrix3d _J_dv_bg; //!> Jacobian of the delta velocity w.r.t the gyro bias
    Eigen::Matrix3d _J_dp_ba; //!> Jacobian of the delta position w.r.t the accel bias
    Eigen::Matrix3d _J_dp_bg; //!> Jacobian of the delta position w.r.t the gyro bias

    unsigned long long _timestamp_imu; //!> Timestamp stored to avoid dependency on the frame
    Eigen::Affine3d _T_w_f_imu;        //!> Transform from the world to the frame to avoid dependency on the frame

  private:
    // Measurements
    Eigen::Vector3d _acc; //!< Acceleration measurement
    Eigen::Vector3d _gyr; //!< Gyroscope measurement

    // IMU noise
    double _gyr_noise;  //!< Gyroscope noise
    double _bgyr_noise; //!< Gyroscope bias random walk noise
    double _acc_noise;  //!< Accelerometer noise
    double _bacc_noise; //! Accelerometer bias random walk noise
    double _rate_hz;    //!< IMU rate in Hz
    Vector6d _eta;      //!< Noise vector for the IMU measurements

    // States computed by processIMU()
    Eigen::Vector3d _delta_p; //!< Pre integration delta in position
    Eigen::Vector3d _delta_v; //!> Pre integration delta in velocity
    Eigen::Matrix3d _delta_R; //!> Pre integration delta in orientation (SO(3))
    Eigen::Vector3d _ba;      //!< Accelerometer bias
    Eigen::Vector3d _bg;      //! Gyroscope bias
    Eigen::Vector3d _v;       //!< Velocity of the IMU in the world frame

    // Covariance computation
    Eigen::MatrixXd _Sigma; //!< Covariance of the pre integration deltas

    std::shared_ptr<IMU> _last_IMU; //!< Last IMU measurement used for pre integration
    std::weak_ptr<Frame> _last_kf;  //!< Last keyframe used for pre integration

    // Mutex
    mutable std::mutex _imu_mtx;
};

} // namespace isae

#endif