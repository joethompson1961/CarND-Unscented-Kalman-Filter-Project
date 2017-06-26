#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  n_x_ = 5;

  n_aug_ = n_x_ + 2;

  lambda_ = 3 - n_aug_;

  //set weights
  weights_ = VectorXd(2 * n_aug_ + 1);
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  for (int i = 1 ; i < 2 * n_aug_ + 1 ; i++) {
    weights_(i) = 1/(2 * (lambda_ + n_aug_));
  }

  // radar measurement dimension (r, phi, and r_dot)
  n_z_ = 3;

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);
  P_.fill(0.0);

  //create matrix for radar sigma points in measurement space
  Zsig_pred_ = MatrixXd(n_z_, 2 * n_aug_ + 1);

  //create vector for mean predicted radar measurement
  z_pred_ = VectorXd(n_z_);

  //create matrix for predicted radar measurement covariance
  S_ = MatrixXd(n_z_, n_z_);

  //create vector for incoming radar measurement
  z_ = VectorXd(n_z_);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.2;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.2;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;  // homework used 0.0175

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;    // homework used 0.1

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
  R_ = MatrixXd(n_z_,n_z_);
  R_ << std_radr_*std_radr_, 0, 0,
        0, std_radphi_*std_radphi_, 0,
        0, 0,std_radrd_*std_radrd_;

  // Sigma point matrices
  Xsig_aug_ = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  Xsig_pred_ = MatrixXd(n_aug_, 2 * n_aug_ + 1);

}

UKF::~UKF() {}

void UKF::GenerateSigmaPoints(VectorXd &x, MatrixXd &P, MatrixXd &Xsig) {

  //define spreading parameter
  double lambda = 3 - n_x_;

  //calculate square root of P
  MatrixXd A = P.llt().matrixL();

  //calculate sigma points ...
  //set sigma points as columns of matrix Xsig
  double c = sqrt(lambda+n_x_);

  //set first column of sigma point matrix equal to current state (the mean state)
  Xsig.col(0) = x_;

  //set remaining sigma points
  for (int i = 0; i < n_x_; i++)
  {
    Xsig.col(i+1)      = x_ + c * A.col(i);
    Xsig.col(i+1+n_x_) = x_ - c * A.col(i);
  }
}

void UKF::GenerateAugmentedSigmaPoints(VectorXd &x, MatrixXd &P, MatrixXd &Xsig) {
  //create augmented mean state vector
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug.head(n_x_) = x;
  x_aug(n_x_) = 0.0;
  x_aug(n_x_ + 1) = 0.0;

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_, n_x_) = P;
  P_aug(n_x_, n_x_) = std_a_ * std_a_;
  P_aug(n_x_ + 1, n_x_ + 1) = std_yawdd_ * std_yawdd_;

  //calculate square root of P
  MatrixXd A = P_aug.llt().matrixL();

  //calculate sigma points ...
  //set sigma points as columns of matrix Xsig
  double c = sqrt(lambda_ + n_aug_);

  //set first column of sigma point matrix equal to current state (the mean state)
  //set remaining sigma points
  Xsig.col(0) = x_aug;
  for (int i = 0; i < n_aug_; i++)
  {
    Xsig.col(i+1)          = x_aug + c * A.col(i);
    Xsig.col(i+1 + n_aug_) = x_aug - c * A.col(i);
  }
}

void UKF::SigmaPointPrediction(double dt, MatrixXd &Xsig_aug, MatrixXd &Xsig_out) {
  double p1 = dt * dt * 0.5;
  double p2;
  double p3;
  double p4;

  //create matrix with predicted sigma points as columns
  MatrixXd Xsig_pred = MatrixXd(n_x_, 2 * n_aug_ + 1);

  //predict sigma points, write predicted sigma points into right column
  VectorXd x_k = VectorXd(n_x_);
  VectorXd f_process = VectorXd(n_x_);
  VectorXd f_noise = VectorXd(n_x_);
  for (int i = 0; i < n_aug_*2 + 1; i++)
  {
    x_k = Xsig_aug.col(i).head(n_x_);

    double px        = Xsig_aug(0, i);
    double py        = Xsig_aug(1, i);
    double v         = Xsig_aug(2, i);
    double yaw       = Xsig_aug(3, i);
    double yaw_d     = Xsig_aug(4, i);
    double nu_a      = Xsig_aug(5, i);
    double nu_yaw_dd = Xsig_aug(6, i);

    //process function
    if (fabs(yaw_d) > 0.0001) {
      p2 = yaw + yaw_d * dt;
      p3 = v / yaw_d;
      f_process(0) = p3 * (sin(p2) - sin(yaw));
      f_process(1) = p3 * (-cos(p2) + cos(yaw));
    } else {
      //avoid division by zero
      p2 = v * dt;
      f_process(0) = p2 * cos(yaw);
      f_process(1) = p2 * sin(yaw);
    }
    f_process(2) = 0;
    f_process(3) = yaw_d * dt;
    f_process(4) = 0;

    //noise function
    p4 = p1 * nu_a;
    f_noise(0) = p4 * cos(yaw);
    f_noise(1) = p4 * sin(yaw);
    f_noise(2) = nu_a * dt;
    f_noise(3) = nu_yaw_dd * p1;
    f_noise(4) = nu_yaw_dd * dt;

    //predicted state
	Xsig_pred.col(i) = x_k + f_process + f_noise;
  }

  //write result
  Xsig_out = Xsig_pred;
}

void UKF::PredictMeanAndCovariance(MatrixXd &Xsig_pred, VectorXd &x_out, MatrixXd &P_out) {
  //create vector for predicted state
  VectorXd x = VectorXd(n_x_);

  //create covariance matrix for prediction
  MatrixXd P = MatrixXd(n_x_, n_x_);

  //predict state mean
  x = Xsig_pred * weights_;

  //predict state covariance matrix
  P.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points

    // state difference
    VectorXd x_diff = Xsig_pred.col(i) - x;
    //angle normalization
    while (x_diff(3) >  M_PI) x_diff(3) -= 2.0 * M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2.0 * M_PI;

    P = P + weights_(i) * x_diff * x_diff.transpose();
  }

  x_out = x;
  P_out = P;
}

void UKF::PredictRadarMeasurement(MatrixXd Xsig_pred, MatrixXd &Zsig_pred, VectorXd &z_out, MatrixXd &S_out) {
  //transform sigma points into radar measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
    double p_x = Xsig_pred(0,i);
    double p_y = Xsig_pred(1,i);
    double v   = Xsig_pred(2,i);
    double yaw = Xsig_pred(3,i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // measurement model
    Zsig_pred(0,i) = sqrt(p_x*p_x + p_y*p_y);                        //r
    Zsig_pred(1,i) = atan2(p_y,p_x);                                 //phi
    Zsig_pred(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);   //r_dot
  }

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z_);
  z_pred.fill(0.0);
  for (int i=0; i < 2*n_aug_+1; i++) {
      z_pred = z_pred + weights_(i) * Zsig_pred.col(i);
  }

  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z_,n_z_);
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
    //residual
    VectorXd z_diff = Zsig_pred.col(i) - z_pred;

    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  S = S + R_;

  //write result
  z_out = z_pred;
  S_out = S;
}


/*******************************************************************************
* Programming assignment functions:
*******************************************************************************/


/**
 * @param {MeasurementPackage} measurement_pack The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage measurement_pack) {
  /**
  TODO:
      Complete this function! Make sure you switch between lidar and radar
      measurements.
  */

  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    /**
      * Initialize the state x_ with the first measurement.
    */
    x_ = VectorXd(n_x_);
    x_ << 1, 1, 0, 0, 0;  // Note: default velocities vx=0, vy=0 are based on observation of dataset 1 and are probably wrong for dataset 2.

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
        * Initialize state vector with location.
        * For radar, convert from polar to cartesian coordinates
      */
      float ro = measurement_pack.raw_measurements_[0];
  	  float theta = measurement_pack.raw_measurements_[1];
  	  float ro_dot = measurement_pack.raw_measurements_[2];
      x_[0] = ro * cos(theta);
      x_[1] = ro * sin(theta);
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
        * Initialize state vector with location.
      */
      x_[0] = measurement_pack.raw_measurements_[0];
      x_[1] = measurement_pack.raw_measurements_[1];
    }

    previous_timestamp_ = measurement_pack.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/
  double delta_t = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;	// delta time is expressed in seconds
  // Only do a prediction update if more than zero time has passed since last measurement
  if (delta_t > ZERO) {
    previous_timestamp_ = measurement_pack.timestamp_;
    Prediction(delta_t);
  }
  else
    cout << "Zero time measurement update!";

  /*****************************************************************************
   *  Update
   ****************************************************************************/
  /**
     * Based on the sensor type, perform the appropriate measurement update step
     * and update the state and covariance matrices.
   */
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
	UpdateRadar(measurement_pack);
//    cout << "x_ = " << endl << x_ << endl;
//    cout << "P_ = " << endl << P_ << endl;
  } else {
    // Laser updates
    UpdateLidar(measurement_pack);
//    cout << "x_ = " << endl << x_ << endl;
//    cout << "P_ = " << endl << P_ << endl;
  }

}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */

//  //set example state
//  x_ <<   5.7441,
//		 1.3800,
//		 2.2049,
//		 0.5015,
//		 0.3528;
//
//  //set example covariance matrix
//  P_ <<     0.0043,   -0.0013,    0.0030,   -0.0022,   -0.0020,
//		  -0.0013,    0.0077,    0.0011,    0.0071,    0.0060,
//		   0.0030,    0.0011,    0.0054,    0.0007,    0.0008,
//		  -0.0022,    0.0071,    0.0007,    0.0098,    0.0100,
//		  -0.0020,    0.0060,    0.0008,    0.0100,    0.0123;

  GenerateAugmentedSigmaPoints(x_, P_, Xsig_aug_);
  SigmaPointPrediction(delta_t, Xsig_aug_, Xsig_pred_);
  PredictMeanAndCovariance(Xsig_pred_, x_, P_);
  PredictRadarMeasurement(Xsig_pred_, Zsig_pred_, z_pred_, S_);

  // print result
  std::cout << "Xsig = " << std::endl << Xsig_pred_ << std::endl;
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
	  z_ << meas_package.raw_measurements_;

	  //create matrix for cross correlation Tc
	  MatrixXd Tc = MatrixXd(n_x_, 2);

	  //calculate cross correlation matrix
	  Tc.fill(0.0);
	  for (int i = 0; i < 2 * n_aug_ + 1; i++) {

	    //residual
//      VectorXd y_diff = Ysig_pred_.col(i) - y_pred_;

	    // state difference
	    VectorXd x_diff = Xsig_pred_.col(i) - x_;

//      Tc = Tc + weights_(i) * x_diff * y_diff.transpose();
	  }

	  //calculate Kalman gain K;
//	  MatrixXd K = Tc * S_.inverse();
//
//	  //residual
//	  VectorXd z_diff = z_ - z_pred_;
//
//	  //angle normalization
//	  while (z_diff(1) >  M_PI) z_diff(1) -= 2.0 * M_PI;
//	  while (z_diff(1) < -M_PI) z_diff(1) += 2.0 * M_PI;
//
//	  //update state mean and covariance matrix
//	  x_ = x_ + K * z_diff;
//	  P_ = P_ - K * S_ * K.transpose();
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
    TODO:

    Complete this function! Use radar data to update the belief about the object's
    position. Modify the state vector, x_, and covariance, P_.

    You'll also need to calculate the radar NIS.
  */
  z_ << meas_package.raw_measurements_;

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z_);

  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {

    //residual
    VectorXd z_diff = Zsig_pred_.col(i) - z_pred_;

    //angle normalization
    while (z_diff(1) >  M_PI) z_diff(1) -= 2.0 * M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2.0 * M_PI;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    //angle normalization
    while (x_diff(3) >  M_PI) x_diff(3) -= 2.0 * M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2.0 * M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //calculate Kalman gain K;
  MatrixXd K = Tc * S_.inverse();

  //residual
  VectorXd z_diff = z_ - z_pred_;

  //angle normalization
  while (z_diff(1) >  M_PI) z_diff(1) -= 2.0 * M_PI;
  while (z_diff(1) < -M_PI) z_diff(1) += 2.0 * M_PI;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S_ * K.transpose();
}
