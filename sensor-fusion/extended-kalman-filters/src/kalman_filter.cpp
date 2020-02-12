#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  // Init Initializes Kalman filter

  // Initial state vector
  x_ = x_in;

  // Initial state covariance matrix
  P_ = P_in;

  // Transistion matrix
  F_ = F_in;

  // Process covariance matrix
  Q_ = Q_in;

  // Measurement matrix
  H_ = H_in;

  // Measurement covariance matrix
  R_ = R_in;
}

void KalmanFilter::Predict() {
  /**  TODO_DONE: Taken from lesson "Laser Measurements Part 4" **/
  // Prediction Predicts the state and the state covariance
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**  TODO_DONE: Taken from lesson "Laser Measurements Part 4" **/
  // Updates the state by using standard Kalman Filter equations
  VectorXd z_predict = H_ * x_;
  VectorXd y = z - z_predict;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  // new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**  TODO: **/
  // Updates the state by using Extended Kalman Filter equations

  // MatrixXd h_x = tools.ConvertCartesianToPolar(x_);
  //recover state parameters

  double px = x_(0);
  double py = x_(1);
  double vx = x_(2);
  double vy = x_(3);

  double px_2_plus_py_2 = fmax(pow(px, 2) + pow(py, 2), 0.0000001);
  double sqrt_px_2_plus_py_2 = sqrt(px_2_plus_py_2);

  MatrixXd h_x(3, 1);
  h_x << sqrt_px_2_plus_py_2,
         atan2(py, px),
         (px * vx + py * vy) / sqrt_px_2_plus_py_2;

  VectorXd y = z - h_x;

  // tools.NormalizePhi(y);
  double phi = y(1);
  phi = atan2(sin(phi), cos(phi));
  y(1) = phi;

  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}
