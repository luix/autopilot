#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::Array;
using Eigen::ArrayXd;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 5 / 3;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = M_PI_2 / 3;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
  DONE:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */

  // size of the state vector
  n_x_ = 5;

  // size of the augmented state vector
  n_aug_ = 7;

  // object as not initialized from start
  is_initialized_ = false;

  // lambda parameter for the sigma points generation
  lambda_ = 3 - n_x_;

  int sigma_points_size = 2 * n_aug_ + 1;
  // allocate predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, sigma_points_size);
  // allocate vector for weights
  weights_ = VectorXd(sigma_points_size);
  // initialize the weights
  weights_.fill(1 / (2 * (lambda_ + n_aug_)));
  weights_(0) = lambda_ / (lambda_ + n_aug_);

  // initialize the covariance matrix
  P_ <<
    pow(5 / 2, 2), 0, 0, 0, 0,
    0, pow(5 / 2, 2), 0, 0, 0,
    0, 0, pow(10 / 2, 2), 0, 0,
    0, 0, 0, pow(M_PI / 2, 2), 0,
    0, 0, 0, 0, pow(M_PI_2 / 2, 2);
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  DONE:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */

  /********************************************
                  Initialization
   ********************************************/

  if (!is_initialized_) {
    time_us_ = meas_package.timestamp_;

    // initialize the state x_ with the first measurement
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      // convert radar from polar to cartesian coordinates and initialize state
      double r = meas_package.raw_measurements_[0];
      double phi = meas_package.raw_measurements_[1];
      double x = r * cos(phi);
      double y = r * sin(phi);

      x_ << x, y, 0, 0, 0;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      // initialize state.
      x_ << meas_package.raw_measurements_[0],
            meas_package.raw_measurements_[1], 0, 0, 0;
    }

    // done initializing
    is_initialized_ = true;
    return;
  }

  /********************************************
                    Prediction
   ********************************************/

  // compute the time elapsed between the current and previous measurements
  double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;	//dt - expressed in seconds
  time_us_ = meas_package.timestamp_;

  this->Prediction(dt);

  /********************************************
                      Update
   ********************************************/

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    if (use_radar_) {
      this->UpdateRadar(meas_package);
    }
  } else {
    // Laser updates
    if (use_laser_) {
      this->UpdateLidar(meas_package);
    }
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  DONE:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */

  /********************************************
             GenerateSigmaPoints
   ********************************************/

  // create sigma point matrix
  MatrixXd Xsig = MatrixXd(n_x_, 2 * n_x_ + 1);

  // calculate square root of P
  MatrixXd A = P_.llt().matrixL();

  // set sigma points as columns of matrix Xsig
  Xsig.col(0) = x_;
  double lambda_sqrt = sqrt(lambda_ + n_x_);
  for ( int i = 0 ; i < n_x_ ; i++ )
  {
      Xsig.col(i + 1) = x_ + lambda_sqrt * A.col(i);
      Xsig.col(i + 1 + n_x_) = x_ - lambda_sqrt * A.col(i);
  }

  /********************************************
             AugmentedSigmaPoints
   ********************************************/

  // create augmented mean state
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug.head(n_x_) = x_;
  x_aug(n_x_) = 0;
  x_aug(n_x_ + 1) = 0;

  // create augmented covariance matrix
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_x_, n_x_) = std_a_ * std_a_;
  P_aug(n_x_ + 1, n_x_ + 1) = std_yawdd_ * std_yawdd_;

  // create square root matrix
  MatrixXd A_aug = P_aug.llt().matrixL();

  //create augmented sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  Xsig_aug.col(0) = x_aug;
  double lambda_aug_sqrt = sqrt(lambda_ + n_aug_);
  for ( int i = 0; i < n_aug_ ; i++ )
  {
    Xsig_aug.col(i + 1) = x_aug + lambda_aug_sqrt * A_aug.col(i);
    Xsig_aug.col(i + n_aug_ + 1) = x_aug - lambda_aug_sqrt * A_aug.col(i);
  }

  /********************************************
             SigmaPointPrediction
   ********************************************/

  //predict sigma points
  for ( int i = 0; i < 2 * n_aug_ + 1 ; i++ )
  {
   //extract values for better readability
   double p_x = Xsig_aug(0,i);
   double p_y = Xsig_aug(1,i);
   double v = Xsig_aug(2,i);
   double yaw = Xsig_aug(3,i);
   double yawd = Xsig_aug(4,i);
   double nu_a = Xsig_aug(5,i);
   double nu_yawdd = Xsig_aug(6,i);

   double sin_yaw = sin(yaw);
   double cos_yaw = cos(yaw);

   //predicted state values
   double px_p, py_p;

   //avoid division by zero
   if (fabs(yawd) > 0.001) {
       px_p = p_x + v / yawd * ( sin (yaw + yawd * delta_t) - sin_yaw);
       py_p = p_y + v / yawd * ( cos_yaw - cos(yaw + yawd * delta_t) );
   }
   else {
       px_p = p_x + v * cos_yaw * delta_t;
       py_p = p_y + v * sin_yaw * delta_t;
   }

   double v_p = v;
   double yaw_p = yaw + yawd * delta_t;
   double yawd_p = yawd;

   //add noise
   px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos_yaw;
   py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin_yaw;
   v_p  = v_p + nu_a * delta_t;

   yaw_p  = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
   yawd_p = yawd_p + nu_yawdd * delta_t;

   //write predicted sigma point into right column
   Xsig_pred_(0,i) = px_p;
   Xsig_pred_(1,i) = py_p;
   Xsig_pred_(2,i) = v_p;
   Xsig_pred_(3,i) = yaw_p;
   Xsig_pred_(4,i) = yawd_p;
  }

  /********************************************
             PredictMeanAndCovariance
   ********************************************/

  // set weights
  weights_(0) = lambda_/(lambda_+n_aug_);
  for ( int i = 1 ; i < 2 * n_aug_ + 1 ; i++)  //2n+1 weights
  {
   weights_(i) = 0.5 / (n_aug_ + lambda_);
  }

  //predicted state mean
  x_.fill(0.0);
  for ( int i = 0 ; i < 2 * n_aug_ + 1 ; i++) //iterate over sigma points
  {
   x_ = x_ + weights_(i) * Xsig_pred_.col(i);
  }

  //predicted state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) //iterate over sigma points
  {
   // state difference
   VectorXd x_diff = Xsig_pred_.col(i) - x_;
   // angle normalization
   while (x_diff(3) >  M_PI) x_diff(3) -= 2. * M_PI;
   while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

   P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */

  //set measurement dimension, lidar can measure x and y
  const int n_z = 2;

  // 2n+1 simga points
  const int sigma_points_size = 2 * n_aug_ + 1;

  /********************************************
             PredictLidarMeasurement
   ********************************************/

  MatrixXd H_laser = MatrixXd(2, n_x_);
  H_laser <<
    1, 0, 0, 0, 0,
    0, 1, 0, 0, 0;

  // matrix for sigma points in measurement space
  MatrixXd Zsig = H_laser * Xsig_pred_;

  // measurement covariance matrix for laser
  MatrixXd R_laser = MatrixXd(n_z,n_z);
  R_laser <<
    std_laspx_ * std_laspx_, 0,
    0, std_laspy_ * std_laspy_;

  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for ( int i = 0 ; i < sigma_points_size ; i++ )
  {
      z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0.0);
  for ( int i = 0 ; i < sigma_points_size ; i++ )
  {
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    //angle normalization
    while (z_diff(1) >  M_PI) z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  S = S + R_laser;

  /********************************************
                   UpdateState
   ********************************************/

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  // calculate cross correlation matrix
  Tc.fill(0.0);
  for ( int i = 0 ; i < sigma_points_size ; i++ )
  {
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    //angle normalization
    while (z_diff(1) >  M_PI) z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3) >  M_PI) x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  // calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  // update state mean and covariance matrix
  VectorXd z = meas_package.raw_measurements_;
  // residual
  VectorXd z_diff = z - z_pred;

  // angle normalization
  while (z_diff(1) >  M_PI) z_diff(1) -= 2. * M_PI;
  while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

  x_ = x_ + K * (z - z_pred);
  P_ = P_ - K * S * K.transpose();

  NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  DONE:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */

  //set measurement dimension, radar can measure r, phi, and r_dot
  const int n_z = 3;

  // 2n+1 simga points
  const int sigma_points_size = 2 * n_aug_ + 1;

  /********************************************
             PredictRadarMeasurement
   ********************************************/

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, sigma_points_size);
  // transform sigma points into measurement space
  for ( int i = 0 ; i < sigma_points_size ; i++ )
  {
   // extract values for better readibility
   double p_x = Xsig_pred_(0,i);
   double p_y = Xsig_pred_(1,i);
   double v  = Xsig_pred_(2,i);
   double yaw = Xsig_pred_(3,i);

   double v1 = cos(yaw) * v;
   double v2 = sin(yaw) * v;

   double sqrt_p2x_p2y = sqrt(p_x * p_x + p_y * p_y);

   // measurement model
   Zsig(0,i) = sqrt_p2x_p2y;                        //rho
   Zsig(1,i) = atan2(p_y,p_x);                      //phi
   Zsig(2,i) = (p_x * v1 + p_y * v2 ) / sqrt_p2x_p2y;   //rho_dot
  }

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for ( int i = 0 ; i < sigma_points_size ; i++ )
  {
     z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0.0);
  for ( int i = 0 ; i < sigma_points_size ; i++ )
  {
   //residual
   VectorXd z_diff = Zsig.col(i) - z_pred;

   //angle normalization
   while (z_diff(1) >  M_PI) z_diff(1) -= 2. * M_PI;
   while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

   S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z,n_z);
  R <<    std_radr_ * std_radr_, 0, 0,
         0, std_radphi_ * std_radphi_, 0,
         0, 0, std_radrd_ * std_radrd_;
  S = S + R;

  /********************************************
                   UpdateState
   ********************************************/

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  // calculate cross correlation matrix
  Tc.fill(0.0);
  for ( int i = 0 ; i < sigma_points_size ; i++ )
  {
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    //angle normalization
    while (z_diff(1) >  M_PI) z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3) >  M_PI) x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  VectorXd z = meas_package.raw_measurements_;
  //residual
  VectorXd z_diff = z - z_pred;

  //angle normalization
  while (z_diff(1) >  M_PI) z_diff(1) -= 2. * M_PI;
  while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();

  NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
}
