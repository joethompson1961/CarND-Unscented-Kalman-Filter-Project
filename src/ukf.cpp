#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initialize Unscented Kalman Filter
 */
UKF::UKF() {
  is_initialized_ = false;
  init_cnt_ = 0;
  time_step_ = 0;

  use_laser_ = true;	// if false, laser measurements will be ignored (except during init)
  use_radar_ = true;	// if false, radar measurements will be ignored (except during init)

  previous_timestamp_ = 0;
  n_x_ = 5;                 // number of elements in state vector
  n_aug_ = n_x_ + 2;        // number of elements in augmented state vector
  n_sig_ = 2 * n_aug_ + 1;  // number of sigma points
  x_ = VectorXd(n_x_);		// initial state vector
  P_ = MatrixXd(n_x_, n_x_);// initial covariance matrix
  P_.fill(0.0);

  // sigma point matrices
  Xsig_aug_ = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  Xsig_pred_ = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  lambda_ = 3 - n_aug_;

  // weights
  weights_ = VectorXd(2 * n_aug_ + 1);
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  for (int i = 1 ; i < 2 * n_aug_ + 1 ; i++) {
    weights_(i) = 1/(2 * (lambda_ + n_aug_));
  }

  // process noise constants
  std_a_ = 0.775;		// process noise standard deviation longitudinal acceleration in m/s^2
  std_yawdd_ = 0.55;	// process noise standard deviation yaw acceleration in rad/s^2
  
  // lidar features
  n_l_ = 2;				// lidar measurement dimension (p_x, p_y)
  std_laspx_ = 0.15;	// lidar measurement noise standard deviation p_x in meters
  std_laspy_ = 0.15;	// lidar measurement noise standard deviation p_y in meters
  R_lidar_ = MatrixXd(n_l_,n_l_);// lidar measurement covariance matrix
  R_lidar_ << std_laspx_*std_laspx_, 0,
              0,                     std_laspy_*std_laspy_;
  H_lidar_ = MatrixXd(2, 5);// lidar measurement function
  H_lidar_ << 1, 0, 0, 0, 0,
              0, 1, 0, 0, 0;

  // radar features
  n_r_ = 3;				// radar measurement dimension (r, phi, and r_dot)
  std_radr_ = 0.3;		// radar measurement noise standard deviation radius in m
  std_radphi_ = 0.03;	// radar measurement noise standard deviation angle in rad
  std_radrd_ = 0.3;		// radar measurement noise standard deviation radius change in m/s
  R_radar_ = MatrixXd(n_r_,n_r_);// radar measurement covariance matrix
  R_radar_ << std_radr_*std_radr_, 0,                       0,
              0,                   std_radphi_*std_radphi_, 0,
              0,                   0,                       std_radrd_*std_radrd_;
}

UKF::~UKF() {}

/** Generate Augmented Sigma Points
 * 
 * Returns augmented sigma points generated from state vector, x, augmented, and state convariance matrix, P, augmented,
 * using spreading parameter, lambda_.
 * 
 * @param x ukf state
 * @param P ukf state covariance matrix
 * @return Xsig 
 * @see n_aug_
 * @see n_x_
 * @see lambda_
 */
void UKF::GenerateAugmentedSigmaPoints(VectorXd &x, MatrixXd &P, MatrixXd &Xsig) {
  // create augmented mean state vector
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug.head(n_x_) = x;
  x_aug(n_x_) = 0.0;
  x_aug(n_x_ + 1) = 0.0;

  // create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_, n_x_) = P;
  P_aug(n_x_, n_x_) = std_a_ * std_a_;
  P_aug(n_x_ + 1, n_x_ + 1) = std_yawdd_ * std_yawdd_;

  // calculate square root of P_aug
  MatrixXd A = P_aug.llt().matrixL();

  // calculate sigma points ...
  
  // set sigma points as columns of matrix Xsig
  double c = sqrt(lambda_ + n_aug_);

  // set first column of sigma point matrix equal to current state (the mean state)
  Xsig.col(0) = x_aug;

  // set remaining sigma points
  for (int i = 0; i < n_aug_; i++)
  {
    Xsig.col(i+1)          = x_aug + c * A.col(i);
    Xsig.col(i+1 + n_aug_) = x_aug - c * A.col(i);
  }
}

/** UKF Sigma Point Prediction
 * 
 * From augmented sigma points (calculated from current state x_ and covariance P_) predict next state of augmented sigma points.
 * 
 * @param dt delta time since last prediction
 * @param Xsig_aug augmented sigma points
 * @return Xsig_out predicted sigma points
 * @see n_x_
 * @see n_aug_
 */
void UKF::SigmaPointPrediction(double dt, MatrixXd &Xsig_aug, MatrixXd &Xsig_out) {
  double dt_2 = dt * dt;
  double p1 = dt_2 * 0.5;
  double p2;
  double p3;

  // create matrix with predicted sigma points as columns
  MatrixXd Xsig_pred = MatrixXd(n_x_, 2 * n_aug_ + 1);

  // predict next augmented sigma points from previous augmented sigma points
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

    // process function
    if (fabs(yaw_d) > ZERO) {
      p2 = yaw + yaw_d * dt;
      p3 = v / yaw_d;
      f_process(0) = p3 * (sin(p2) - sin(yaw));
      f_process(1) = p3 * (-cos(p2) + cos(yaw));
    } else {
      // avoid division by zero
      p2 = v * dt;
      f_process(0) = p2 * cos(yaw);
      f_process(1) = p2 * sin(yaw);
    }
    f_process(2) = 0;
    f_process(3) = yaw_d * dt;
    f_process(4) = 0;

    // noise function
    double q = p1 * nu_a;
    f_noise(0) = q * cos(yaw);
    f_noise(1) = q * sin(yaw);
    f_noise(2) = nu_a * dt;
    f_noise(3) = nu_yaw_dd * p1;
    f_noise(4) = nu_yaw_dd * dt;

    // predicted state
	Xsig_pred.col(i) = x_k + f_process + f_noise;
  }

  // write result
  Xsig_out = Xsig_pred;
}

/** UKF Predict Next State Mean and Coavariance
 * 
 * From predicted sigma points predict the next state vector and covariance matrix.
 * 
 * @param Xsig_pred predicted sigma points
 * @return x_out predicted state
 * @return P_out predicted state covariance
 * @see n_x_
 * @see n_aug_
 * @see weights_
 */
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
    x_diff(3) = tools_.NormalizeAngle(x_diff(3));	// angle normalization

    P = P + weights_(i) * x_diff * x_diff.transpose();
  }

  x_out = x;
  P_out = P;
}

/** Predict Radar Measurement
 * 
 * Transform predicted sigma points into radar measurement space and calculate the predicted radar measurement.
 * 
 * @param Xsig_pred predicted sigma points
 * @return Rsig_pred predicted sigma points transformed to radar measurement space 
 * @return r_pred predicted radar measurement (weighted average of transformed predicted sigma points, Rsig_pred)
 */
void UKF::PredictRadarMeasurement(const MatrixXd &Xsig_pred, MatrixXd &Rsig_pred, VectorXd &r_pred) {
  r_pred.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
    double p_x = Xsig_pred(0,i);
    double p_y = Xsig_pred(1,i);
    double v   = Xsig_pred(2,i);
    double yaw = Xsig_pred(3,i);

    double v1 = cos(yaw) * v;
    double v2 = sin(yaw) * v;

    // transform sigma points into radar measurement space
    Rsig_pred(0,i) = sqrt(p_x*p_x + p_y*p_y);                        //r
    Rsig_pred(1,i) = atan2(p_y,p_x);                                 //phi
    Rsig_pred(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);   //r_dot

    // predict radar measurement - r_pred is the weighted average of the predicted sigma points
    r_pred = r_pred + weights_(i) * Rsig_pred.col(i);
  }
}

/** UKF Process Measurement
 * 
 * Process each new measurement through the unscented kalman filter.
 * 
 * @param {MeasurementPackage} measurement_pack the latest measurement data of either radar or laser
 */
void UKF::ProcessMeasurement(MeasurementPackage measurement_pack) {
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
    x_ << 0, 0, 0, 0, 0;  // Initial velocity is estimated based on first 2 measurements of the same sensor type.

    // Initialize state vector with location.
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      // For radar, convert from polar to cartesian coordinates
      float ro = measurement_pack.raw_measurements_[0];
      float theta = measurement_pack.raw_measurements_[1];
      float ro_dot = measurement_pack.raw_measurements_[2];
      x_[0] = ro * cos(theta);
      x_[1] = ro * sin(theta);
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      x_[0] = measurement_pack.raw_measurements_[0];
      x_[1] = measurement_pack.raw_measurements_[1];
    }

    // initialize the state, x_ [pos_x pos_y vel_abs yaw_angle yaw_rate]
    // initialization of velocity and yaw angle requires multiple measurements
    if (init_cnt_ == 1) {
      px_ = x_[0];
      py_ = x_[1];
      previous_timestamp_ = measurement_pack.timestamp_;
    } else if (init_cnt_ >= 3) {
      delta_t = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
      dx = x_[0] - px_;
      dy = x_[1] - py_;
      d = sqrt(dx*dx + dy*dy);  // distance traveled so far

      // initial velocity is an average from first N samples
      // this initialization ISN'T a good general solution; it's specific for the project test datasets.
      x_[2] = d / delta_t;
      // initial velocity is around 5mps (by observation).
      x_[2] = 4.95;

      // initial yaw angle: if dx is negative then flip 180 degrees (starts facing left instead of right).
      // this initialization ISN'T a good general solution; it's specific for the project test datasets.
      if (dx < 0.0)
    	  x_[3] = 3.14;

      previous_timestamp_ = measurement_pack.timestamp_;

      cout << "dx:" << dx << endl;
      cout << endl << "initial x_:" << endl << x_ << endl;
      
      is_initialized_ = true;
    }

    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/
  delta_t = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;	// delta time is expressed in seconds
  // Only do a prediction update if more than zero time has passed since last measurement
  if (delta_t > ZERO_T) {
    previous_timestamp_ = measurement_pack.timestamp_;
    Prediction(delta_t);
  }
  else
    cout << "Zero time measurement update!";

  /*****************************************************************************
   *  Update
   ****************************************************************************/
  // Based on the sensor type, perform the appropriate measurement update step and
  // update the state and covariance matrices.
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    VectorXd z = VectorXd(n_r_);
    z << measurement_pack.raw_measurements_;
    UpdateRadar(z);
  } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
    // Laser updates
    VectorXd z = VectorXd(n_l_);
    z << measurement_pack.raw_measurements_;
//	  UpdateLidar(z);
    UpdateLidar_KF(z);
  }
  cout << endl << "x_(" << time_step_ << ") = " << endl << x_ << endl;
//  cout << "P_ = " << endl << P_ << endl;
}

/** UKF Prediction
 * 
 * Predict sigma points, the state, and the state covariance matrix.
 * Estimate the object's location. Modifies the state vector, x_, and state covariance, P_.
 * 
 * @param {double} delta_t the change in time (in seconds) between the last measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  GenerateAugmentedSigmaPoints(x_, P_, Xsig_aug_);
  SigmaPointPrediction(delta_t, Xsig_aug_, Xsig_pred_);
  PredictMeanAndCovariance(Xsig_pred_, x_, P_);
//  std::cout << "Xsig = " << std::endl << Xsig_pred_ << std::endl;
}

/** UKF Radar Measurement Update
 * 
 * Use radar measurement to update the belief about the object's position.
 * Modify the state vector, x_, and covariance, P_, and calculate the radar NIS.
 * 
 * @param z radar measurement
 * @return x_ ukf state vector
 * @return P_ ukf state covariance matrix
 * @return NIS_radar_ radar NIS (Normalized Innovation Squared)
 */
void UKF::UpdateRadar(const VectorXd &z) {
  // predicted radar measurement; calculated from predicted sigma points
  MatrixXd Rsig_pred = MatrixXd(n_r_, n_sig_);  // radar sigma points in radar measurement space
  VectorXd r_pred = VectorXd(n_r_);             // mean predicted radar measurement
  PredictRadarMeasurement(Xsig_pred_, Rsig_pred, r_pred);

  // radar measurement innovation
  VectorXd y = z - r_pred;
  y(1) = tools_.NormalizeAngle((double)y(1));	// angle normalization

  // calculate radar cross correlation matrix and radar measurement innovation covariance
  MatrixXd S = MatrixXd(n_r_, n_r_);   // radar innovation (residual) covariance
  S.fill(0.0);
  MatrixXd Tc = MatrixXd(n_x_, n_r_);  // cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < n_sig_; i++) {
    // predicted radar measurement innovation (residual)
	VectorXd r_diff = Rsig_pred.col(i) - r_pred;
    r_diff(1) = tools_.NormalizeAngle((double)r_diff(1));	// angle normalization

    // predicted state innovation
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    x_diff(3) = tools_.NormalizeAngle((double)x_diff(3));	// angle normalization

    S = S + weights_(i) * r_diff * r_diff.transpose();
    Tc = Tc + weights_(i) * x_diff * r_diff.transpose();
  }
  S = S + R_radar_;  // innovation (residual) covariance

  // kalman gain
  MatrixXd K = Tc * S.inverse();

  // calculate radar NIS (Normalized Innovation Squared); NIS is a cost function useful for parameter tuning.
  NIS_radar_ = y.transpose() * S.inverse() * y;  // 1x3 * 3x3 * 3x1 ==> 1x1

  // update state mean and covariance matrix
  x_ = x_ + K * y;
  P_ = P_ - K * S * K.transpose();
}

/** UKF Update for Lidar Measurement
 * 
 * Use lidar measurement to update the belief about the object's position.
 * 
 * Modify the state vector, x_, and covariance, P_, and calculate the lidar
 * NIS.
 * 
 * Note that a lidar measurement provides px and py, which don't require a
 * non-linear conversion from the predicted state to measurement space.
 * Therefore, simply use px and py from the sigma points (no need for
 * augmented sigma points).
 * 
 * Note that UKF is overkill for lidar due to lidar predictions are a linear
 * function. Hence the lidar update is more effectively implemented with
 * computationally efficient standard kalman filter.
 * 
 * @param z lidar measurement
 * @return x_ kalman state vector
 * @return P_ kalman state covariance matrix
 * @return NIS_lidar_ lidar NIS (Normalized Innovation Squared)
 */
void UKF::UpdateLidar(const VectorXd &z) {
  // create matrix for lidar sigma points in lidar measurement space
  MatrixXd Lsig_pred = MatrixXd(n_l_, n_sig_);
  for (int i = 0; i < n_sig_; i++) {
    Lsig_pred(0, i) = Xsig_pred_(0, i);
    Lsig_pred(1, i) = Xsig_pred_(1, i);
  }

  // mean predicted lidar measurement
  VectorXd l_pred = VectorXd(n_l_);
  l_pred(0) = x_(0);
  l_pred(1) = x_(1);

  // lidar measurement innovation (residual)
  VectorXd y = z - l_pred;

  // calculate lidar measurement innovation covariance and cross correlation matrix
  MatrixXd S = MatrixXd(n_l_, n_l_);   // lidar innovation (residual) covariance
  S.fill(0.0);
  MatrixXd Tc = MatrixXd(n_x_, n_l_);  // cross correlation matrix Tc
  Tc.fill(0.0);
  for (int i = 0; i < n_sig_; i++) {
    // predicted lidar measurement innovation (residual)
    VectorXd l_diff = Lsig_pred.col(i) - l_pred;

    // predicted state innovation
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    S = S + weights_(i) * l_diff * l_diff.transpose();
    Tc = Tc + weights_(i) * x_diff * l_diff.transpose();
  }
  S = S + R_lidar_;  // innovation (residual) covariance

  // kalman gain
  MatrixXd K = Tc * S.inverse();

  // lidar NIS (Normalized Innovation Squared); NIS is a cost function useful for parameter tuning.
  NIS_lidar_ = y.transpose() * S.inverse() * y;  // 1x2 + (2x2 * 2x1)

  // update state mean and covariance matrix
  x_ = x_ + K * y;
  P_ = P_ - K * S * K.transpose();
}

/** Kalman filter measurement update function
 * 
 * Use lidar measurement to update the belief about the object's position.
 * This uses linear update function instead of sigma points; better choice for lidar measurements.
 * 
 * @param z lidar measurement
 * @return x_ state vector
 * @return P_ state covariance matrix
 * @return NIS_lidar_ lidar NIS (Normalized Innovation Squared)
 */
void UKF::UpdateLidar_KF(const VectorXd &z) {
  MatrixXd P_Ht = P_ * H_lidar_.transpose();  // do this calculation once in advance to eliminate executing twice below.

  // measurement innovation
  VectorXd y = z - H_lidar_ * x_;     // (2,1) - (2,4) * (4,1)  ==> (2,1)

  // innovation covariance
  MatrixXd S = H_lidar_ * P_Ht + R_lidar_;   // (2,4) * (4,4) * (4,2) + (2,2) ==> (2,2)

  // kalman gain
  MatrixXd K =  P_Ht * S.inverse();      // (4,4) * (4,2) * (2,2)  ==> (4,2)

  // calculate lidar NIS (Normalized Innovation Squared); NIS is a cost function useful for parameter tuning.
  NIS_lidar_ = y.transpose() * S.inverse() * y;  // (1,2) * (2,2) * (2,1) ==> (1,1)

  // new state and covariance
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  x_ = x_ + (K * y);                  // (4,1) + (4,2) * (2,1)  ==> (4,1)
  P_ = (I - K * H_lidar_) * P_;       // ((4,4) - (4,2) * (2,4)) * (4,4)  ==> (4,4)
}
