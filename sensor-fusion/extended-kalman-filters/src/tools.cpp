#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

using namespace std;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /** TODO:  **/

  // Initializes the RMSE vector.
  VectorXd RMSE(4);
  RMSE << 0, 0, 0, 0;

  const int E_SIZE = estimations.size();

  // Validate that estimations vector is not zero.
  if (E_SIZE == 0) {
    cerr << "Error: Estimations vector should not be zero.";
    cerr << endl;
    return RMSE;
  }
  // Validate that both ground_truth and estimations vectors have same sizes.
  if (E_SIZE != ground_truth.size()) {
    cerr << "Error: ground_truth and estimations vectors have different sizes.";
    return RMSE;
  }

  // Calculate the Root Mean Squared Error (RMSE) in three steps:

  //  s the minimum mean square error estimate for the estimations
  // vector and the ground_truth with linear functions fk and hk.
  for (int i = 0; i < E_SIZE; ++i) {
    VectorXd diff = estimations[i] - ground_truth[i];
    VectorXd mult = diff.array() * diff.array();
    RMSE += mult;
  }

  //calculate the mean
  RMSE /= E_SIZE;

  //calculate the squared root
  RMSE = RMSE.array().sqrt();

  //return the result
  return RMSE;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
  TODO:
    * Calculate a Jacobian here.
  */
  MatrixXd Hj(3,4);

  //recover state parameters
  double px = x_state(0);
  double py = x_state(1);
  double vx = x_state(2);
  double vy = x_state(3);

  double px_2_plus_py_2 = fmax(pow(px, 2) + pow(py, 2), 0.0000001);
  double sqrt_px_2_plus_py_2 = sqrt(px_2_plus_py_2);
  double px_2_plus_py_2_3_2 = sqrt_px_2_plus_py_2 * px_2_plus_py_2;

  Hj << px / sqrt_px_2_plus_py_2, py / sqrt_px_2_plus_py_2, 0, 0,
        -py / px_2_plus_py_2, px / px_2_plus_py_2, 0, 0,
        py * (vx * py - vy * px) / px_2_plus_py_2_3_2, px * (vy * px - vx * py) / px_2_plus_py_2_3_2, px / sqrt_px_2_plus_py_2, py / sqrt_px_2_plus_py_2;

  return Hj;
}
