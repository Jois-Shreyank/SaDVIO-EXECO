#include <ceres/ceres.h>
#include <unordered_map>

#include "isaeslam/data/frame.h"
#include "isaeslam/data/landmarks/ALandmark.h"
#include "isaeslam/data/maps/localmap.h"
#include "isaeslam/typedefs.h"

namespace isae {

/*!
 * @brief Marginalization block struct that stores a factor and the indices of the variables involved
 */
struct MarginalizationBlockInfo {

    MarginalizationBlockInfo(ceres::CostFunction *cost_function,
                             std::vector<int> parameter_idx,
                             std::vector<double *> parameter_blocks)
        : _cost_function(cost_function), _parameter_idx(parameter_idx), _parameter_blocks(parameter_blocks) {}

    void Evaluate();

    ceres::CostFunction *_cost_function;
    std::vector<int> _parameter_idx;
    std::vector<double *> _parameter_blocks;

    double **_raw_jacobians;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> _jacobians;
    Eigen::VectorXd _residuals;
};

/*!
 * @brief Marginalization class that handles marginalization (and sparsification) for fixed-lag smoothing.
 *
 * This class is responsible for selecting the variables to keep and to marginalize, computing the information matrix,
 * computing the Schur complement, and computing the Jacobians and residuals for the marginalization. It also provides
 * methods for sparsifying the dense prior in the case of VIO and VO.
 */
class Marginalization {
  public:
    /*!
     * @brief Select all the variables to keep and to marginalize in the Markov Blanket for fixed lag smoothing
     * @param frame0 frame to marginalize
     * @param frame1 frame linked to frame0
     * @param marginalization_last previous marginalization scheme
     */
    void preMarginalize(std::shared_ptr<Frame> &frame0,
                        std::shared_ptr<Frame> &frame1,
                        std::shared_ptr<Marginalization> &marginalization_last);

    /*!
     * @brief Select all the variables to keep and marginalize to derive the relative pose factor between two frames
     * @param frame0 The first frame
     * @param frame1 The second frame
     */
    void preMarginalizeRelative(std::shared_ptr<Frame> &frame0, std::shared_ptr<Frame> &frame1);

    /*!
     * @brief Sparsify the dense prior factor in the VIO case
     */
    bool sparsifyVIO();

    /*!
     * @brief Sparsify the dense prior factor in the VO case
     */
    bool sparsifyVO();

    /*!
     * @brief Compute the information matrix and the gradient for a set of factors
     * @param blocks The vector of factors stored in Marginalization Blocks
     * @param A The information matrix
     * @param B The gradient
     */
    void computeInformationAndGradient(std::vector<std::shared_ptr<MarginalizationBlockInfo>> blocks,
                                       Eigen::MatrixXd &A,
                                       Eigen::VectorXd &b);

    /*!
     * @brief Compute the SVD of a given matrix to reveal its rank
     * @param A The input matrix
     * @param U The Eigen vectors of non null eigen values (up to a threshold)
     * @param d Non null Eigen values
     */
    void rankReveallingDecomposition(Eigen::MatrixXd A, Eigen::MatrixXd &U, Eigen::VectorXd &d);

    /*!
     * @brief Compute the dense prior with the Schur complement on _Ak
     */
    bool computeSchurComplement();

    /*!
     * @brief Compute the jacobian and the residual of the dense prior factor
     */
    bool computeJacobiansAndResiduals();

    /*!
     * @brief Compute the Entropy of a given landmark
     */
    double computeEntropy(std::shared_ptr<ALandmark> lmk);

    /*!
     * @brief Compute the Mutual Information between two landmarks
     */
    double computeMutualInformation(std::shared_ptr<ALandmark> lmk_i, std::shared_ptr<ALandmark> lmk_j);

    /*!
     * @brief Approximate the Mutual Information between two landmarks using off diagonal blocks of _Ak
     */
    double computeOffDiag(std::shared_ptr<ALandmark> lmk_i, std::shared_ptr<ALandmark> lmk_j);

    /*!
     * @brief Compute the KLD between the multivariate Gaussian with their Information Matrix assuming their mean is
     * equal
     */
    double computeKLD(Eigen::MatrixXd A_p, Eigen::MatrixXd A_q);

    int _m;                    //!> Parametric size of the variables to marginalize
    int _n;                    //!> Parametric size of the variables to keep
    int _n_full;               //!> Parametric size of the variables to keep after rank reveilling
    const double _eps = 1e-12; //!> Threshold to consider a null eigen value

    // Bookeeping of the variables to keep and to marginalize
    std::shared_ptr<Frame> _frame_to_marg;                            //!> Frame to marginalize
    std::shared_ptr<Frame> _frame_to_keep;                            //!> Frame to keep
    typed_vec_landmarks _lmk_to_keep;                                 //!> Set of landmarks to keep
    typed_vec_landmarks _lmk_to_marg;                                 //!> Set of landmarks to marginalize
    std::unordered_map<std::shared_ptr<Frame>, int> _map_frame_idx;   //!> Map between frames and indices in _Ak
    std::unordered_map<std::shared_ptr<ALandmark>, int> _map_lmk_idx; //!> Map between landmarks and indices in _Ak
    std::unordered_map<std::shared_ptr<Frame>, Eigen::MatrixXd>
        _map_frame_inf; //!> Map between frame and their marginal information matrix
    std::vector<std::shared_ptr<MarginalizationBlockInfo>>
        _marginalization_blocks; //!> Vector of Marginalization blocks to derive _Ak

    // Sparsification info
    std::unordered_map<std::shared_ptr<ALandmark>, Eigen::Matrix3d>
        _map_lmk_inf; //!> Map between landmarks and info mat of sparse relative prior factors
    std::unordered_map<std::shared_ptr<ALandmark>, Eigen::Vector3d>
        _map_lmk_prior;                         //!> Map between landmarks and priors of sparse prior relative factors
    std::shared_ptr<ALandmark> _lmk_with_prior; //!> Landmark that has an absolute prior factor
    Eigen::Matrix3d _info_lmk;                  //!> Information matrix of the landmark absolute prior
    Eigen::Vector3d _prior_lmk;                 //!> Prior of the landmark absolute prior

    // Matrices and vectors of the dense prior
    Eigen::MatrixXd _Ak;                       //!> Information matrix of the subproblem
    Eigen::VectorXd _bk;                       //!> Gradient of the subproblem
    Eigen::MatrixXd _Sigma_k;                  //!> Covariance of the dense prior
    Eigen::MatrixXd _U;                        //!> Eigen vectors that have non null eigen values
    Eigen::VectorXd _Lambda;                   //!> Non null Eigen values
    Eigen::VectorXd _Sigma;                    //!> Inverse of _Lambda
    Eigen::MatrixXd _marginalization_jacobian; //!> Jacobian of the dense prior factor
    Eigen::VectorXd _marginalization_residual; //!> Residual of the dense prior factor
};

/*!
 * @brief Ceres cost function of the dense prior factor
 */
class MarginalizationFactor : public ceres::CostFunction {
  public:
    MarginalizationFactor(std::shared_ptr<Marginalization> marginalization_info)
        : _marginalization_info(marginalization_info) {

        // Add frame block size
        if (_marginalization_info->_frame_to_keep) {
            this->mutable_parameter_block_sizes()->push_back(6);
            this->mutable_parameter_block_sizes()->push_back(3);
            this->mutable_parameter_block_sizes()->push_back(3);
            this->mutable_parameter_block_sizes()->push_back(3);
        }

        // Add landmark block size
        for (auto tlmk : _marginalization_info->_lmk_to_keep) {
            for (auto lmk : tlmk.second) {
                (tlmk.first == "pointxd" ? this->mutable_parameter_block_sizes()->push_back(3)
                                         : this->mutable_parameter_block_sizes()->push_back(6));
            }
        }

        // Set the number of residuals
        this->set_num_residuals(_marginalization_info->_n_full);
    }

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
        int n = _marginalization_info->_n_full;
        Eigen::VectorXd dx(_marginalization_info->_n);
        dx.setZero();

        // Add frame dx
        int block_id = 0;
        if (_marginalization_info->_frame_to_keep) {
            dx.segment<6>(_marginalization_info->_map_frame_idx.at(_marginalization_info->_frame_to_keep)) =
                Eigen::Map<const Eigen::Matrix<double, 6, 1>>(parameters[block_id]);
            block_id++;
            dx.segment<3>(_marginalization_info->_map_frame_idx.at(_marginalization_info->_frame_to_keep) + 6) =
                Eigen::Map<const Eigen::Vector3d>(parameters[block_id]);
            block_id++;
            dx.segment<3>(_marginalization_info->_map_frame_idx.at(_marginalization_info->_frame_to_keep) + 9) =
                Eigen::Map<const Eigen::Vector3d>(parameters[block_id]);
            block_id++;
            dx.segment<3>(_marginalization_info->_map_frame_idx.at(_marginalization_info->_frame_to_keep) + 12) =
                Eigen::Map<const Eigen::Vector3d>(parameters[block_id]);
            block_id++;
        }

        // Add landmarks dx
        for (auto tlmk : _marginalization_info->_lmk_to_keep) {
            for (auto lmk : tlmk.second) {
                if (_marginalization_info->_map_lmk_idx.at(lmk) == -1)
                    continue;

                dx.segment<3>(_marginalization_info->_map_lmk_idx.at(lmk)) =
                    Eigen::Map<const Eigen::Vector3d>(parameters[block_id]);
                block_id++;
            }
        }

        // Compute the residual
        Eigen::Map<Eigen::VectorXd>(residuals, n) =
            _marginalization_info->_marginalization_residual + _marginalization_info->_marginalization_jacobian * dx;

        // Fill the jacobians
        if (jacobians) {

            block_id = 0;

            // Jacobians on frame
            if (_marginalization_info->_frame_to_keep) {
                if (jacobians[block_id]) {

                    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobian(
                        jacobians[block_id], n, 6);
                    jacobian.setZero();
                    jacobian.leftCols(6) = _marginalization_info->_marginalization_jacobian.middleCols(
                        _marginalization_info->_map_frame_idx.at(_marginalization_info->_frame_to_keep), 6);
                }
                block_id++;
                if (jacobians[block_id]) {

                    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobian(
                        jacobians[block_id], n, 3);
                    jacobian.setZero();
                    jacobian.leftCols(3) = _marginalization_info->_marginalization_jacobian.middleCols(
                        _marginalization_info->_map_frame_idx.at(_marginalization_info->_frame_to_keep) + 6, 3);
                }
                block_id++;
                if (jacobians[block_id]) {

                    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobian(
                        jacobians[block_id], n, 3);
                    jacobian.setZero();
                    jacobian.leftCols(3) = _marginalization_info->_marginalization_jacobian.middleCols(
                        _marginalization_info->_map_frame_idx.at(_marginalization_info->_frame_to_keep) + 9, 3);
                }
                block_id++;
                if (jacobians[block_id]) {

                    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobian(
                        jacobians[block_id], n, 3);
                    jacobian.setZero();
                    jacobian.leftCols(3) = _marginalization_info->_marginalization_jacobian.middleCols(
                        _marginalization_info->_map_frame_idx.at(_marginalization_info->_frame_to_keep) + 12, 3);
                }
                block_id++;
            }

            // Jacobians on landmarks
            for (auto tlmk : _marginalization_info->_lmk_to_keep) {
                for (auto lmk : tlmk.second) {
                    if (_marginalization_info->_map_lmk_idx.at(lmk) == -1)
                        continue;

                    if (jacobians[block_id]) {

                        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobian(
                            jacobians[block_id], n, 3);
                        jacobian.setZero();
                        jacobian.leftCols(3) = _marginalization_info->_marginalization_jacobian.middleCols(
                            _marginalization_info->_map_lmk_idx.at(lmk), 3);
                    }
                    block_id++;
                }
            }
        }
        return true;
    }

    std::shared_ptr<Marginalization> _marginalization_info;
};

} // namespace isae