#ifndef AFEATUREMATCHER_H
#define AFEATUREMATCHER_H

#include "isaeslam/typedefs.h"
#include <type_traits>

#include "isaeslam/data/features/AFeature2D.h"
#include "isaeslam/data/sensors/Camera.h"
#include "isaeslam/featuredetectors/aFeatureDetector.h"

namespace isae {

/*! @brief Unordered map to store matches between two feature lists */
typedef std::unordered_map<std::shared_ptr<AFeature>, std::vector<std::shared_ptr<AFeature>>> vec_feat_matches;

/*! @brief Unordered map to store scores of matches between two feature lists */
typedef std::unordered_map<std::shared_ptr<AFeature>, std::vector<double>> vec_feat_matches_scores;

/*!
 * @brief Class for matching features between two sets of features.
 *
 * This class provides methods to match features between two sets of features, filter matches, and link landmarks.
 * It uses a feature detector to compute distances between feature descriptors.
 */
class AFeatureMatcher {
  public:
    AFeatureMatcher() {}
    AFeatureMatcher(std::shared_ptr<AFeatureDetector> detector) : _detector(detector) {}

    /*!
     * @brief Match features between two sets of features.
     *
     * @param features1 First set of features.
     * @param features2 Second set of features.
     * @param features_init Initial set of features for matching.
     * @param matches Output vector to store matched features.
     * @param matches_with_ldmks Output vector to store matched features with landmarks.
     * @param searchAreaWidth Width of the search area for matching (default is 51).
     * @param searchAreaHeight Height of the search area for matching (default is 51).
     *
     * @return The number of matched features.
     */
    virtual uint match(std::vector<std::shared_ptr<AFeature>> &features1,
                       std::vector<std::shared_ptr<AFeature>> &features2,
                       std::vector<std::shared_ptr<AFeature>> &features_init,
                       vec_match &matches,
                       vec_match &matches_with_ldmks,
                       int searchAreaWidth  = 51,
                       int searchAreaHeight = 51);
    /*!
     * @brief Match landmarks with features in a given sensor.
     *
     * @param sensor1 The sensor in which to match landmarks.
     * @param ldmks Vector of landmarks to match.
     * @param searchAreaWidth Width of the search area for matching (default is 51).
     * @param searchAreaHeight Height of the search area for matching (default is 51).
     *
     * @return The number of matched landmarks.
     */
    virtual uint ldmk_match(std::shared_ptr<ImageSensor> &sensor1,
                            std::vector<std::shared_ptr<ALandmark>> &ldmks,
                            int searchAreaWidth  = 51,
                            int searchAreaHeight = 51);

  protected:
    /*!
     * @brief Get possible matches between two sets of features.
     *
     * This method computes the matches between two sets of features based on their descriptors and spatial proximity.
     *
     * @param features1 First set of features.
     * @param features2 Second set of features.
     * @param features_init Initial set of features for matching.
     * @param searchAreaWidth Width of the search area for matching.
     * @param searchAreaHeight Height of the search area for matching.
     * @param matches Output vector to store matched features.
     * @param all_scores Output vector to store scores of all matches.
     */
    virtual void getPossibleMatchesBetween(const std::vector<std::shared_ptr<AFeature>> &features1,
                                           const std::vector<std::shared_ptr<AFeature>> &features2,
                                           const std::vector<std::shared_ptr<AFeature>> &features_init,
                                           const uint &searchAreaWidth,
                                           const uint &searchAreaHeight,
                                           vec_feat_matches &matches,
                                           vec_feat_matches_scores &all_scores);
    /*!
     * @brief Filter matches based on the first and second best match scores.
     *
     * This method filters the matches based on the ratio of the first and second best match scores.
     *
     * @param matches12 Matches from the first to the second set of features.
     * @param matches21 Matches from the second to the first set of features.
     * @param all_scores12 Scores of all matches from the first to the second set.
     * @param all_scores21 Scores of all matches from the second to the first set.
     *
     * @return A vector of filtered matches.
     */
    vec_match filterMatches(vec_feat_matches &matches12,
                            vec_feat_matches &matches21,
                            vec_feat_matches_scores &all_scores12,
                            vec_feat_matches_scores &all_scores21);

    std::shared_ptr<AFeatureDetector> _detector; //!< feature detector for distance measurement
    double _first_second_match_score_ratio =
        0.9;                    //!< ratio between the first and second best match score to consider a match valid
    std::string _feature_label; //!< label for the features being matched
};

} // namespace isae
#endif // AFEATUREMATCHER_H
