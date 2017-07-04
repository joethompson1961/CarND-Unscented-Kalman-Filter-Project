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
  init_cnt_ = 0;
  time_step_ = 0;

  previous_timestamp_ = 0;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  n_x_ = 5;

  n_aug_ = n_x_ + 2;

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);
  P_.fill(0.0);

  // Sigma point matrices
  Xsig_aug_ = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  Xsig_pred_ = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  lambda_ = 3 - n_aug_;

  //set weights
  weights_ = VectorXd(2 * n_aug_ + 1);
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  for (int i = 1 ; i < 2 * n_aug_ + 1 ; i++) {
    weights_(i) = 1/(2 * (lambda_ + n_aug_));
  }

  // lidar measurement dimension (p_x, p_y)
  n_y_ = 2;

  //create vector for incoming radar measurement
  y_ = VectorXd(n_y_);

//  // Process noise standard deviation longitudinal acceleration in m/s^2
//  std_a_ = 0.2;
//
//  // Process noise standard deviation yaw acceleration in rad/s^2
//  std_yawdd_ = 0.2;
//
//  // Laser measurement noise standard deviation position1 in m
//  std_laspx_ = 0.15;
//
//  // Laser measurement noise standard deviation position2 in m
//  std_laspy_ = 0.15;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.22;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.25;

//  // Laser measurement noise standard deviation position1 in m
//  std_laspx_ = 0.2;
//
//  // Laser measurement noise standard deviation position2 in m
//  std_laspy_ = 0.15;

  L_ = MatrixXd(n_y_,n_y_);
  L_ << std_a_*std_a_, 0,
        0, std_yawdd_*std_yawdd_;


  // radar measurement dimension (r, phi, and r_dot)
  n_z_ = 3;

  //create vector for incoming radar measurement
  z_ = VectorXd(n_z_);

  //create matrix for radar sigma points in measurement space
  Zsig_pred_ = MatrixXd(n_z_, 2 * n_aug_ + 1);

  //create vector for mean predicted radar measurement
  z_pred_ = VectorXd(n_z_);

  // Radar measurement noise standard deviation radius in m
//  std_radr_ = 0.3;
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
//  std_radphi_ = 0.03;  // homework used 0.0175
  std_radphi_ = 0.017;  // homework used 0.0175

  // Radar measurement noise standard deviation radius change in m/s
//  std_radrd_ = 0.3;    // homework used 0.1
  std_radrd_ = 0.2;    // homework used 0.1

  R_ = MatrixXd(n_z_,n_z_);
  R_ << std_radr_*std_radr_, 0, 0,
        0, std_radphi_*std_radphi_, 0,
        0, 0,std_radrd_*std_radrd_;
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

// From predicted sigma points predict the next state vector and covariance matrix
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

// Transform predicted sigma points into radar measurement space and predict the next radar measurement
void UKF::PredictRadarMeasurement(MatrixXd Xsig_pred, MatrixXd &Zsig_pred, VectorXd &z_pred) {
  z_pred.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
    double p_x = Xsig_pred(0,i);
    double p_y = Xsig_pred(1,i);
    double v   = Xsig_pred(2,i);
    double yaw = Xsig_pred(3,i);

    double v1 = cos(yaw) * v;
    double v2 = sin(yaw) * v;

    // transform sigma points into radar measurement space
    Zsig_pred(0,i) = sqrt(p_x*p_x + p_y*p_y);                        //r
    Zsig_pred(1,i) = atan2(p_y,p_x);                                 //phi
    Zsig_pred(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);   //r_dot

    // predict radar measurement - z_pred is the weighted average of the predicted sigma points
    z_pred = z_pred + weights_(i) * Zsig_pred.col(i);
  }
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
  float dx;
  float dy;
  float d;
  double delta_t;

  time_step_ += 1;

  if ((measurement_pack.sensor_type_ == MeasurementPackage::RADAR) && !use_radar_)
    return;
  if ((measurement_pack.sensor_type_ == MeasurementPackage::LASER) && !use_laser_)
    return;

  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    init_cnt_ += 1;
    x_ = VectorXd(n_x_);
    x_ << 0, 0, 0, 0, 0;  // Initial velocity is estimated based on first 2 measurements.

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

//    x_[2] = 5.0;
////    x_[3] = 3.14;
//    previous_timestamp_ = measurement_pack.timestamp_;
//    is_initialized_ = true;
//    return;

    /**
     * Initialize the state x_ with the first measurement.
     * [pos_x pos_y vel_abs yaw_angle yaw_rate]
     */
    if (init_cnt_ == 1) {
      // Initialization of velocity and yaw requires multiple measurements, all from the same sensor type.
      px_ = x_[0];
      py_ = x_[1];
      previous_timestamp_ = measurement_pack.timestamp_;
    } else if (init_cnt_ >= 2) {
      delta_t = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
      dx = x_[0] - px_;
      dy = x_[1] - py_;
      d = sqrt(dx*dx + dy*dy);

      //initial velocity is an average from first N samples
      x_[2] = d / delta_t;
      x_[2] = 4.95;
//      x_[3] = 3.14;

      previous_timestamp_ = measurement_pack.timestamp_;
      is_initialized_ = true;
      cout << endl << "initial x_:" << endl << x_ << endl;
    }
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/
  delta_t = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;	// delta time is expressed in seconds
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
  } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
    // Laser updates
    UpdateLidar(measurement_pack);
  }
  cout << endl << "x_(" << time_step_ << ") = " << endl << x_ << endl;
//  cout << "P_ = " << endl << P_ << endl;
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
  GenerateAugmentedSigmaPoints(x_, P_, Xsig_aug_);
  SigmaPointPrediction(delta_t, Xsig_aug_, Xsig_pred_);
  PredictMeanAndCovariance(Xsig_pred_, x_, P_);
  PredictRadarMeasurement(Xsig_pred_, Zsig_pred_, z_pred_);

  // print result
//  std::cout << "Xsig = " << std::endl << Xsig_pred_ << std::endl;
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
  int n_sig = 2 * n_aug_ + 1;  // number of sigma points
  z_ << meas_package.raw_measurements_;

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z_);

  //calculate apriori radar measurement innovation and cross correlation matrices
  MatrixXd S = MatrixXd(n_z_, n_z_);  // radar innovation (residual) covariance
  S.fill(0.0);
  Tc.fill(0.0);
  for (int i = 0; i < n_sig; i++) {
    //predicted radar measurement innovation (residual)
    VectorXd z_diff = Zsig_pred_.col(i) - z_pred_;

    //angle normalization
    while (z_diff(1) >  M_PI) z_diff(1) -= 2.0 * M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2.0 * M_PI;

    //predicted state innovation
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    //angle normalization
    while (x_diff(3) >  M_PI) x_diff(3) -= 2.0 * M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2.0 * M_PI;

    S = S + weights_(i) * z_diff * z_diff.transpose();
    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }
  S = S + R_;  // innovation (residual) covariance

  //calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //radar measurement vs. prediction difference
  VectorXd z_diff = z_ - z_pred_;

  //angle normalization
  while (z_diff(1) >  M_PI) z_diff(1) -= 2.0 * M_PI;
  while (z_diff(1) < -M_PI) z_diff(1) += 2.0 * M_PI;

  //calculate radar NIS (Normalized Innovation Squared); NIS is a cost function useful for parameter tuning.
  NIS_radar_ = 0.0;
  for (int i = 0; i < n_sig; i++) {
    NIS_radar_ = NIS_radar_ + (z_diff.transpose() * S.inverse() * z_diff);  // 1x3 * 3x3 * 3x1
  }
  NIS_radar_ = NIS_radar_ / n_sig;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();
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

  For lidar measurement only includes px and py. This doesn't require a non-linear conversion from the predicted state
  to measurement space. Instead simply use px and py from the sigma points.  In fact the UKF is overkill for lidar
  and really this should be implemented with EKF.
  */
  int n_sig = 2 * n_aug_ + 1;  // number of sigma points
  y_ << meas_package.raw_measurements_;

  //create matrix for lidar sigma points in measurement space
  MatrixXd Ysig_pred = MatrixXd(n_y_, n_sig);
  for (int i = 0; i < n_sig; i++) {
    Ysig_pred(0, i) = Xsig_pred_(0, i);
    Ysig_pred(1, i) = Xsig_pred_(1, i);
  }

  //create vector for mean predicted lidar measurement
  VectorXd y_pred = VectorXd(n_y_);
  y_pred(0) = x_(0);
  y_pred(1) = x_(1);

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_y_);

  //calculate apriori lidar measurement innovation and cross correlation matrices
  MatrixXd S = MatrixXd(n_y_, n_y_);  // lidar innovation (residual) covariance
  S.fill(0.0);
  Tc.fill(0.0);
  for (int i = 0; i < n_sig; i++) {

    //predicted radar measurement innovation (residual)
    VectorXd y_diff = Ysig_pred.col(i) - y_pred;

    //predicted state innovation
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    S = S + weights_(i) * y_diff * y_diff.transpose();
    Tc = Tc + weights_(i) * x_diff * y_diff.transpose();
  }
  S = S + L_;  // innovation (residual) covariance

  //calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //lidar measurement innovation (residual)
  VectorXd y_diff = y_ - y_pred;

  //calculate radar NIS (Normalized Innovation Squared); NIS is a cost function useful for parameter tuning.
  NIS_lidar_ = 0.0;
  for (int i = 0; i < n_sig; i++) {
    NIS_lidar_ = NIS_lidar_ + (y_diff.transpose() * S.inverse() * y_diff);  // 1x2 * 2x2 * 2x1
  }
  NIS_lidar_ = NIS_lidar_ / n_sig;

  //update state mean and covariance matrix
  x_ = x_ + K * y_diff;
  P_ = P_ - K * S * K.transpose();
}
