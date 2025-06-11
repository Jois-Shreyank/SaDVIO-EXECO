#ifndef ALANDMARKINITIALIZER_H
#define ALANDMARKINITIALIZER_H

#include <iostream>
#include <opencv2/core.hpp>

#include "isaeslam/data/landmarks/ALandmark.h"
#include "isaeslam/typedefs.h"

namespace isae {

class ImageSensor;
class AFeature;
class ALandmark;

/*!
* \brief Abstract class for initializing landmarks.
*
* This class provides methods to initialize landmarks from feature matches or to update existing landmarks
* Each novel type of landmark must have its own derived class that implements the specific initialization logic.                             
*/
class ALandmarkInitializer : public std::enable_shared_from_this<ALandmarkInitializer> {
  public:
    ALandmarkInitializer() = default;
    ~ALandmarkInitializer() {}
    /*!
     * \brief Initialize landmarks from a feature pair.
     * \param match The feature pair to initialize the landmark from.
     * \return The number of initialized landmarks.
     */
    uint initFromMatch(feature_pair match);

    /*!
     * \brief Initialize landmarks from a vector of feature matches.
     * \param matches The vector of feature matches to initialize the landmarks from.
     * \return The number of initialized landmarks.
     */
    uint initFromMatches(vec_match matches);

    /*!
     * \brief Initialize landmarks from a vector of feature tracks.
     * \param tracks The vector of feature tracks to initialize the landmarks from.
     * \return The number of initialized landmarks.
     */
    uint initFromFeatures(std::vector<std::shared_ptr<AFeature>> feats);

  protected:
    /*!
     * \brief Create a new landmark from a pair of features.
     * \param f1 The first feature.
     * \param f2 The second feature.
     * \return A shared pointer to the newly created landmark.
     */
    std::shared_ptr<ALandmark> createNewLandmark(std::shared_ptr<AFeature> f1, std::shared_ptr<AFeature> f2);

    /*!
     * \brief Initialize a landmark from a set of features.
     * \param features A vector of features.
     * \param landmark A shared pointer to the landmark to be initialized.
     * \return True if the landmark was successfully initialized, false otherwise.
     */
    virtual bool initLandmark(std::vector<std::shared_ptr<AFeature>> features,
                              std::shared_ptr<ALandmark> &landmark)          = 0;

    /*!
     * \brief Initialize a landmark from a set of features with depth information.
     * \param features A vector of features.
     * \param landmark A shared pointer to the landmark to be initialized.
     * \return True if the landmark was successfully initialized with depth, false otherwise.
     */
    virtual bool initLandmarkWithDepth(std::vector<std::shared_ptr<AFeature>> features,
                                       std::shared_ptr<ALandmark> &landmark) = 0;

    virtual std::shared_ptr<ALandmark> createNewLandmark(std::shared_ptr<AFeature> f) = 0;
};

} // namespace isae

#endif // ALANDMARKINITIALIZER_H
