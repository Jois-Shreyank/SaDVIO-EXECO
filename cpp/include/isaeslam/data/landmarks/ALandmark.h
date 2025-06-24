#ifndef ALANDMARK_H
#define ALANDMARK_H

#include "isaeslam/data/landmarks/Model3D.h"
#include "utilities/geometry.h"

namespace isae {

class AFeature;

/*!
 * @brief Abstract class for 3D Landmarks.
 *
 * It contains a 6DoF pose and a 3D model for a general formulation.
 * It also contains a set of features that are associated to the landmark.
 */
class ALandmark : public std::enable_shared_from_this<ALandmark> {
  public:
    ALandmark() {}
    ALandmark(Eigen::Affine3d T_w_l, std::vector<std::shared_ptr<isae::AFeature>> features);
    ~ALandmark() {}

    /*!
     * @brief Virtual Function to properly initialize a landmark.
     */
    virtual void init(Eigen::Affine3d T_w_l, std::vector<std::shared_ptr<isae::AFeature>> features);

    void addFeature(std::shared_ptr<AFeature> feature);
    std::vector<std::weak_ptr<AFeature>> getFeatures() { return _features; }

    /*!
     * @brief Remove features that are not linked to anything.
     */
    void removeExpiredFeatures();

    void removeFeature(std::shared_ptr<AFeature> f);

    /*!
     * @brief Fuse this landmark with another one, the feature sets are merged.
     *
     * If there is a wrong behavior in the association, it returns false
     */
    bool fuseWithLandmark(std::shared_ptr<ALandmark> landmark);

    void setPose(Eigen::Affine3d T_w_l) {
        std::lock_guard<std::mutex> lock(_lmk_mtx);
        _T_w_l = T_w_l;
    }

    /*!
     * @brief Set only the translation of the landmark pose.
     */
    void setPosition(Eigen::Vector3d t_w_l) {
        std::lock_guard<std::mutex> lock(_lmk_mtx);
        _T_w_l.translation() = t_w_l;
    }

    Eigen::Affine3d getPose() const {
        std::lock_guard<std::mutex> lock(_lmk_mtx);
        return _T_w_l;
    }

    std::vector<Eigen::Vector3d> getModelPoints() { return _model->getModel(); }
    std::shared_ptr<AModel3d> getModel() const { return _model; }

    cv::Mat getDescriptor() const {
        std::lock_guard<std::mutex> lock(_lmk_mtx);
        return _descriptor;
    }
    void setDescriptor(cv::Mat descriptor) {
        std::lock_guard<std::mutex> lock(_lmk_mtx);
        _descriptor = descriptor;
    }

    bool isInitialized() const {
        std::lock_guard<std::mutex> lock(_lmk_mtx);
        return _initialized;
    }
    void setUninitialized() {
        std::lock_guard<std::mutex> lock(_lmk_mtx);
        _initialized = false;
    }

    void setInMap() {
        std::lock_guard<std::mutex> lock(_lmk_mtx);
        _in_map = true;
    }
    bool isInMap() {
        std::lock_guard<std::mutex> lock(_lmk_mtx);
        return _in_map;
    }

    /*!
     * @brief Check if the landmark is valid in terms of reprojection error and number of features.
     */
    bool sanityCheck();

    /*!
     * @brief Compute the chi2 error for a given feature.
     */
    virtual double chi2err(std::shared_ptr<AFeature> f);

    /*!
     * @brief Compute the average chi2 error for all features associated to the landmark.
     */
    double avgChi2err();

    bool isOutlier() const {
        std::lock_guard<std::mutex> lock(_lmk_mtx);
        return _outlier;
    }
    void setOutlier() {
        std::lock_guard<std::mutex> lock(_lmk_mtx);
        _outlier = true;
    }
    void setInlier() {
        std::lock_guard<std::mutex> lock(_lmk_mtx);
        _outlier = false;
    }

    bool isResurected() const {
        std::lock_guard<std::mutex> lock(_lmk_mtx);
        return _is_resurected;
    }
    void setResurected() {
        std::lock_guard<std::mutex> lock(_lmk_mtx);
        _is_resurected = true;
    }

    bool hasPrior() const {
        std::lock_guard<std::mutex> lock(_lmk_mtx);
        return _has_prior;
    }
    void setPrior() {
        std::lock_guard<std::mutex> lock(_lmk_mtx);
        _has_prior = true;
    }

    bool isMarg() const {
        std::lock_guard<std::mutex> lock(_lmk_mtx);
        return _is_marg;
    }
    void setMarg() {
        std::lock_guard<std::mutex> lock(_lmk_mtx);
        _is_marg = true;
    }

    static int _landmark_count; //!< Static counter for landmarks, used for unique id generation
    int _id;                    //!< Unique id for the landmark, used for bookeeping
    std::string _label;         //!< Label for the landmark

  protected:
    bool _initialized   = false; //!< Flag to check if the landmark is initialized
    bool _in_map        = false; //!< Flag to check if the landmark is in the map
    bool _outlier       = false; //!< Flag to check if the landmark is an outlier
    bool _is_resurected = false; //!< Flag to check if the landmark has been resurected
    bool _has_prior     = false; //!< Flag to check if the landmark has a prior
    bool _is_marg       = false; //!< Flag to check if the landmark is marginalised

    Eigen::Affine3d _T_w_l;           //!< Landmark pose in world frame
    cv::Mat _descriptor;              //!< Descriptor of the landmark, used for matching
    std::shared_ptr<AModel3d> _model; //!< 3D model of the landmark

    std::vector<std::weak_ptr<AFeature>> _features; //!< Features associated to the landmark

    mutable std::mutex _lmk_mtx;
};

} // namespace isae

#endif
