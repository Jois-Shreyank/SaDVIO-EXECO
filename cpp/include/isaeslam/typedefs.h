#ifndef TYPEDEFS_H
#define TYPEDEFS_H

#include <vector>
#include <memory>
#include <utility>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include <unordered_map>


namespace isae {
    class AFeature;
    class ALandmark;

/*! @brief A double Eigen vector in 6D */
typedef Eigen::Matrix<double, 6, 1> Vector6d;

/*! @brief A pair of feature, useful to represent matches */
typedef std::pair<std::shared_ptr<isae::AFeature>,std::shared_ptr<isae::AFeature>> feature_pair;
/*! @brief A vector of feature pairs i.e. matches */
typedef std::vector<feature_pair> vec_match;
/*! @brief An unordered map to link match vector with their type */
typedef std::unordered_map<std::string, vec_match> typed_vec_match;
/*! @brief A typed vector of features to handle hetereogeneous feature sets */
typedef std::unordered_map<std::string, std::vector<std::shared_ptr<isae::AFeature> >> typed_vec_features;
/*! @brief A typed vector of landmarks to handle hetereogeneous landmark sets */
typedef std::unordered_map<std::string, std::vector<std::shared_ptr<isae::ALandmark> >> typed_vec_landmarks;
}

#endif //TYPEDEFS_H
