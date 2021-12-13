#pragma once

#include <vector>
#include <opencv2/core.hpp>

namespace toolbox {

    /*!
     * Transformation of projective to euclidean coordinates
     *
     * @param u_e: result d by n matrix; n euclidean vectors of dimension d
     * @param u_p: d+1 by n matrix; n homogeneous vectors of dimension d+1
     */
    [[maybe_unused]] void p2e(cv::Mat3d &u_e,
                              const cv::Mat4d &u_p);

    /*!
     * Transformation of euclidean to projective coordinates
     *
     * @param u_p: result d+1 by n matrix; n homogeneous vectors of dimension d+1
     * @param u_e: d by n matrix; n euclidean vectors of dimension d
     */
    [[maybe_unused]] void e2p(cv::Mat4d &u_p,
                              const cv::Mat3d &u_e);

    /*!
     * Construct homography matrix from 4 points
     *
     * @param H: result H a 3×3 homography matrix (np.array), or an empty array [] if there is no solution.
     * @param u1:  (3×4) the image coordinates of points in the first image
     * @param u2: (3×4) the image coordinates of the corresponding points in the second image.
     */
    [[maybe_unused]] void u2H(cv::Matx33d &H,
                              const cv::Matx34d &u1,
                              const cv::Matx34d &u2);

    /*!
     *  Skew-symmetric matrix for cross-product
     *
     * @param x: vector 3×1
     * @param X: skew symmetric matrix (3×3) for cross product with x
     */
    [[maybe_unused]] void sqc(cv::Matx33d &X,
                              const cv::Vec3d &x);

    /*!
     * Rotation matrix (9 parameters) to rotation vector (3 parameters) using rodrigues formula
     * source: https://courses.cs.duke.edu/fall13/compsci527/notes/rodrigues.pdf
     *
     * @param r: return rotation vector
     * @param R: rotation matrix
     */
    [[maybe_unused]] void R2mrp(cv::Vec3d &r,
                                const cv::Matx33d &R);

    /*!
     * modified rodrigues parameters to matrix
     * source: https://courses.cs.duke.edu/fall13/compsci527/notes/rodrigues.pdf
     *
     * @param R: result rotation matrix
     * @param r: rotation vector
     */
    [[maybe_unused]] void mrp2R(cv::Matx33d &R,
                                const cv::Vec3d &r);

    /*!
     * Essential matrix decomposition with cheirality
     * Notes: The sessential matrix E is decomposed such that E = R * sqc( b )
     *
     * @param R: result relative rotation (3×3) or [None, None] if cheirality fails
     * @param t: result relative translation, euclidean (3×1), unit length
     * @param E: result essential matrix (3×3)
     * @param u1: corresponding image points in homogeneous coordinates (3×n), used for cheirality test
     * @param u2: corresponding image points in homogeneous coordinates (3×n), used for cheirality test
     */
    [[maybe_unused]] void Eu2Rt(cv::Matx33d &R,
                                cv::Vec3d &t,
                                const cv::Matx33d &E,
                                const cv::Vec3d &u1,
                                const cv::Vec3d &u2);

    /*!
     * Binocular reconstruction by DLT triangulation
     *
     * @param Xs: reconstructed 3D points, homogeneous (4×n)
     * @param P1: projective camera1 matrix (3×4)
     * @param P2: projective camera2 matrix (3×4)
     * @param u1: corresponding image points in homogeneous coordinates (3×n) image 1
     * @param u2: corresponding image points in homogeneous coordinates (3×n) image 2
     */
    [[maybe_unused]] void Pu2X(cv::Mat4d &Xs,
                               const cv::Matx34d &P1,
                               const cv::Matx34d &P2,
                               const cv::Mat3d &u1,
                               const cv::Mat3d &u2);

    /*!
     *
     * @param errors: computed reprojection errors
     * @param P1: camera 1 projection matrix
     * @param P2: camera 2 projection matrix
     * @param u1: 3*n matrix, 2d points in homogeneous coordinate system
     * @param u2: 3*n matrix, 2d points in homogeneous coordinate system
     * @param X: 4*n matrix, 3d points in homogeneous coordinate system
     */
    [[maybe_unused]] void err_reprojection(cv::Mat1d &errors,
                                           const cv::Matx34d &P1,
                                           const cv::Matx34d &P2,
                                           const cv::Mat3d &u1,
                                           const cv::Mat3d &u2,
                                           const cv::Mat4d &X);

    /*!
     * Compute projection error given P, X, u
     *
     * @param errors: computed reprojection errors
     * @param P: camera projection matrix
     * @param u: 3*n matrix, 2d points in homogeneous coordinate system
     * @param X: 4*n matrix, 3d points in homogeneous coordinate system
     */
    [[maybe_unused]] void err_reprojection(cv::Mat1d &errors,
                                           const cv::Matx34d &P,
                                           const cv::Mat3d &u,
                                           const cv::Mat4d &X);

    /*!
     * Sampson error approximation on epipolar geometry
     *
     * @param errors: Squared Sampson error for each correspondence (1×n).
     * @param F: fundamental matrix (3×3)
     * @param u1: corresponding image 1 points in homogeneous coordinates (3×n)
     * @param u2: corresponding image 2 points in homogeneous coordinates (3×n)
     */
    [[maybe_unused]] void err_F_sampson(cv::Mat1d &errors,
                                        const cv::Matx33d &F,
                                        const cv::Mat3d &u1,
                                        const cv::Mat3d &u2);

    /*!
     * Sampson correction of correspondences
     *
     * @param u1_res: corrected corresponding points, homog. (3×n).
     * @param u2_res corrected corresponding points, homog. (3×n).
     * @param F: fundamental matrix (3×3)
     * @param u1: corresponding image points in homogeneous coordinates (3×n)
     * @param u2: corresponding image points in homogeneous coordinates (3×n)
     */
    [[maybe_unused]] void u_correct_sampson(cv::Mat3d &u1_res,
                                            cv::Mat3d &u2_res,
                                            const cv::Matx33d &F,
                                            const cv::Mat3d &u1,
                                            const cv::Mat3d &u2);

    /*!
     * function to minimise R and t from correspondent points and calibration matrix
     *
     * @param R: camera rotation matrix
     * @param t: camera translation vector
     * @param u1: corresponding image points in homogeneous coordinates (3×n)
     * @param u2: corresponding image points in homogeneous coordinates (3×n)
     * @param mask: which from u1 and u2 points to use
     * @param K: camera calibration matrix
     */
    [[maybe_unused]] void Rt_optimise_im2im(cv::Matx33d &R,
                                            cv::Vec3d &t,
                                            const cv::Mat3d &u1,
                                            const cv::Mat3d &u2,
                                            const cv::Mat1d &mask,
                                            const cv::Matx33d &K);

    /*!
     * function to minimise R and t from correspondent points and calibration matrix
     *
     * @param R: camera rotation matrix
     * @param t: camera translation vector
     * @param X: 4*n matrix, 3d points in homogeneous coordinate system
     * @param u: corresponding image points in homogeneous coordinates (3×n)
     * @param mask: which from u1 and u2 points to use
     * @param K: camera calibration matrix
     */
    [[maybe_unused]] void Rt_optimise_X2im(cv::Matx33d &R,
                                           cv::Vec3d &t,
                                           const cv::Mat4d &X,
                                           const cv::Mat3d &u,
                                           const cv::Mat1d &mask,
                                           const cv::Matx33d &K);
} // toolbox