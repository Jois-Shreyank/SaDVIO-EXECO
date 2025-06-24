#ifndef MESHER_H
#define MESHER_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <queue>
#include <thread>

#include "isaeslam/data/mesh/mesh.h"
#include "isaeslam/data/sensors/ASensor.h"
#include "isaeslam/featuredetectors/aFeatureDetector.h"
#include "utilities/timer.h"

namespace isae {

/*! @brief A class to handles 3D meshing via 2D triangulation on successive KFs
 *
 * It works as a thread that receives new KFs and computes the 2D mesh for each image sensor then update the 3D
 * Mesh accordingly
 */
class Mesher {

  public:
    Mesher(std::string slam_mode, double ZNCC_tsh, double max_length_tsh);

    /*! @brief Create a 2D mesh for a given image sensor */
    std::vector<FeatPolygon> createMesh2D(std::shared_ptr<ImageSensor> sensor);

    /*! @brief Compute the Delaunnay triangulation of a set of openCV points in an image */
    std::vector<cv::Vec6f> computeMesh2D(const cv::Size img_size, const std::vector<cv::Point2f> p2f_to_triangulate);

    /*! @brief add a new KF in the queue (for the thread) */
    void addNewKF(std::shared_ptr<Frame> frame);
    bool getNewKf();

    /*! @brief The thread function */
    void run();

    std::queue<std::shared_ptr<Frame>> _kf_queue; //!< The queue of keyframes to process
    std::shared_ptr<Frame> _curr_kf;              //!< The current keyframe being processed
    std::shared_ptr<Mesh3D> _mesh_3d;             //!< The 3D mesh object
    std::string _slam_mode;                       //!< The SLAM mode (e.g., "nofov", "bimono", etc.)
    double _avg_mesh_t;                           //!< Average time taken to process a mesh update for profiling
    int _n_kf;                                    //!< Number of keyframes processed for profiling

    mutable std::mutex _mesher_mtx;
};

} // namespace isae

#endif // MESHER_H