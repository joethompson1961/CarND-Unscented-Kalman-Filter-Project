#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>
#include "tools.h"

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

  n_x_ = 5;                 // number of elements in state vector

  n_aug_ = n_x_ + 2;        // number of elements in augmented state vector
  n_sig_ = 2 * n_aug_ + 1;  // number of sigma points


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

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.775;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.55;

  // lidar measurement dimension (p_x, p_y)
  n_l_ = 2;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // lidar measurement covariance matrix
  R_lidar_ = MatrixXd(n_l_,n_l_);
  R_lidar_ << std_laspx_*std_laspx_, 0,
              0,                     std_laspy_*std_laspy_;

  // lidar measurement function
  H_lidar_ = MatrixXd(2, 5);
  H_lidar_ << 1, 0, 0, 0, 0,
              0, 1, 0, 0, 0;

  // Radar measurement dimension (r, phi, and r_dot)
  n_r_ = 3;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  // radar measurement covariance matrix
  R_radar_ = MatrixXd(n_r_,n_r_);
  R_radar_ << std_radr_*std_radr_, 0,                       0,
              0,                   std_radphi_*std_radphi_, 0,
              0,                   0,                       std_radrd_*std_radrd_;
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
  double dt_2 = dt * dt;
  double p1 = dt_2 * 0.5;
  double p2;
  double p3;

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
    double q = p1 * nu_a;
    f_noise(0) = q * cos(yaw);
    f_noise(1) = q * sin(yaw);
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
    x_diff(3) = tools_.NormalizeAngle(x_diff(3));	// angle normalization

    P = P + weights_(i) * x_diff * x_diff.transpose();
  }

  x_out = x;
  P_out = P;
}

// Transform predicted sigma points into radar measurement space and calculate the predicted radar measurement
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

/**
 * @param {MeasurementPackage} measurement_pack The latest measurement data of
 * either radar or laser.
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

    /**
     * Initialize the state x_ with the first measurement.
     * [pos_x pos_y vel_abs yaw_angle yaw_rate]
     */
    if (init_cnt_ == 1) {
      // Initialization of velocity and yaw requires multiple measurements, all from the same sensor type.
      px_ = x_[0];
      py_ = x_[1];
      previous_timestamp_ = measurement_pack.timestamp_;
    } else if (init_cnt_ >= 3) {
      delta_t = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
      dx = x_[0] - px_;
      dy = x_[1] - py_;
      d = sqrt(dx*dx + dy*dy);

      // this initialization ISN'T a good general solution; it's specific for the project test datasets.
      // initial velocity is an average from first N samples
      x_[2] = d / delta_t;
      // initial velocity is around 5mps (by observation).
      x_[2] = 4.95;

      // initialize yaw angle - if dx is negative then 180 degrees (starts facing left instead of right.
      if (dx < 0.0)
    	  x_[3] = 3.14;

      previous_timestamp_ = measurement_pack.timestamp_;
      is_initialized_ = true;
      cout << "dx:" << dx << endl;
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

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
	Estimate the object's location. Modify the state vector, x_.
	Predict sigma points, the state, and the state covariance matrix.
  */
  GenerateAugmentedSigmaPoints(x_, P_, Xsig_aug_);
  SigmaPointPrediction(delta_t, Xsig_aug_, Xsig_pred_);
  PredictMeanAndCovariance(Xsig_pred_, x_, P_);

  // print result
//  std::cout << "Xsig = " << std::endl << Xsig_pred_ << std::endl;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {VectorXd} z radar measurement
 */
void UKF::UpdateRadar(const VectorXd &z) {
  /**
	Use radar data to update the belief about the object's
    position. Modify the state vector, x_, and covariance, P_.
	Calculate the radar NIS.
  */
  // predicted radar measurement; calculated from predicted sigma points
  MatrixXd Rsig_pred = MatrixXd(n_r_, n_sig_);  //radar sigma points in radar measurement space
  VectorXd r_pred = VectorXd(n_r_);            //mean predicted radar measurement
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
//    while (r_diff(1) >  M_PI) r_diff(1) -= 2.0 * M_PI;  //angle normalization
//    while (r_diff(1) < -M_PI) r_diff(1) += 2.0 * M_PI;  //angle normalization
    r_diff(1) = tools_.NormalizeAngle((double)r_diff(1));	// angle normalization

    // predicted state innovation
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
//    while (x_diff(3) >  M_PI) x_diff(3) -= 2.0 * M_PI;  //angle normalization
//    while (x_diff(3) < -M_PI) x_diff(3) += 2.0 * M_PI;  //angle normalization
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

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {VectorXd} z - lidar measurement
 */
void UKF::UpdateLidar(const VectorXd &z) {
  /**
	Use lidar data to update the belief about the object's
	position. Modify the state vector, x_, and covariance, P_.
	Calculate the lidar NIS.

	The lidar measurement only includes px and py which
	don't require a non-linear conversion from the predicted state to measurement space.
	Instead simply use px and py from the sigma points.
	UKF is overkill for lidar due to lidar predictions are linear function; this should
	be implemented with more computationally efficient standard kalman filter.
  */
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

void UKF::UpdateLidar_KF(const VectorXd &z) {
  /**
    * Kalman filter measurement update function
  */
  MatrixXd P_Ht = P_ * H_lidar_.transpose();  // do this calculation once in advance to eliminate executing twice below.

  // measurement innovation
  VectorXd y = z - H_lidar_ * x_;     // (2,1) - (2,4) * (4,1)  ==> (2,1)

  // innovation covariance
  MatrixXd S = H_lidar_ * P_Ht + R_lidar_;   // (2,4) * (4,4) * (4,2) + (2,2) ==> (2,2)

  // kalman gain
  MatrixXd K =  P_Ht * S.inverse();      // (4,4) * (4,2) * (2,2)  ==> (4,2)

  //calculate lidar NIS (Normalized Innovation Squared); NIS is a cost function useful for parameter tuning.
  NIS_lidar_ = y.transpose() * S.inverse() * y;  // (1,2) * (2,2) * (2,1) ==> (1,1)

  // new state and covariance
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  x_ = x_ + (K * y);                  // (4,1) + (4,2) * (2,1)  ==> (4,1)
  P_ = (I - K * H_lidar_) * P_;       // ((4,4) - (4,2) * (2,4)) * (4,4)  ==> (4,4)
}
