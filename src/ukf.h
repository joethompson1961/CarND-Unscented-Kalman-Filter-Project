#ifndef UKF_H
#define UKF_H

#include "measurement_package.h"
#include "Eigen/Dense"
#include <vector>
#include <string>
#include <fstream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

#define ZERO (0.001F)

class UKF {
public:

  //initially set to false, set to true in first call of ProcessMeasurement
  bool is_initialized_;

  //previous timestamp
  long long previous_timestamp_;

  //if this is false, laser measurements will be ignored (except for init)
  bool use_laser_;

  //if this is false, radar measurements will be ignored (except for init)
  bool use_radar_;


  ///* State variables
  //state dimension
  int n_x_;

  //augmented state dimension
  int n_aug_;

  //state vector: [pos_x pos_y vel_abs yaw_angle yaw_rate] in SI units and rad
  VectorXd x_;

  //state covariance matrix
  MatrixXd P_;

  //augmented sigma points matrix
  MatrixXd Xsig_aug_;

  //predicted sigma points matrix
  MatrixXd Xsig_pred_;


  ///* Lidar measurement variables
  //lidar measurement dimension
  int n_y_;

  //create matrix for lidar sigma points in measurement space
  MatrixXd Ysig_pred_;

  //create vector for mean predicted lidar measurement
  VectorXd y_pred_;

  //create vector for incoming lidar measurement
  VectorXd y_;

  //create matrix for predicted lidar measurement covariance
  MatrixXd L_;

  ///* Laser measurement noise standard deviation position1 in m
  double std_laspx_;

  ///* Laser measurement noise standard deviation position2 in m
  double std_laspy_;

  ///* Process noise standard deviation longitudinal acceleration in m/s^2
  double std_a_;

  ///* Process noise standard deviation yaw acceleration in rad/s^2
  double std_yawdd_;

  //create matrix for predicted lidar measurement covariance
  MatrixXd S_l_;

  ///* Radar measurement variables
  // radar measurement dimension
  int n_z_;

  //create matrix for radar sigma points in measurement space
  MatrixXd Zsig_pred_;

  //create vector for mean predicted radar measurement
  VectorXd z_pred_;

  //create vector for incoming radar measurement
  VectorXd z_;

  //create matrix for predicted radar measurement covariance
  MatrixXd R_;

  ///* Radar measurement noise standard deviation radius in m
  double std_radr_;

  ///* Radar measurement noise standard deviation angle in rad
  double std_radphi_;

  ///* Radar measurement noise standard deviation radius change in m/s
  double std_radrd_ ;

  //create matrix for predicted radar measurement covariance
  MatrixXd S_r_;

  ///* Weights of sigma points
  VectorXd weights_;

  ///* Sigma point spreading parameter
  double lambda_;

  /**
   * Constructor
   */
  UKF();

  /**
   * Destructor
   */
  virtual ~UKF();

  void GenerateSigmaPoints(VectorXd &x, MatrixXd &P, MatrixXd &Xsig_out);

  void GenerateAugmentedSigmaPoints(VectorXd &x, MatrixXd &P, MatrixXd &Xsig_out);

  void SigmaPointPrediction(double dt, MatrixXd &Xsig_aug, MatrixXd &Xsig_out);

  void PredictMeanAndCovariance(MatrixXd &Xsig_pred, VectorXd &x_out, MatrixXd &P_out);

  void PredictRadarMeasurement(MatrixXd Xsig_pred, MatrixXd &Zsig_pred_, VectorXd &z_out, MatrixXd &S_out);

 /**
   * ProcessMeasurement
   * @param meas_package The latest measurement data of either radar or laser
   */
  void ProcessMeasurement(MeasurementPackage meas_package);

  /**
   * Prediction Predicts sigma points, the state, and the state covariance
   * matrix
   * @param delta_t Time between k and k+1 in s
   */
  void Prediction(double delta_t);

  /**
   * Updates the state and the state covariance matrix using a laser measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateLidar(MeasurementPackage meas_package);

  /**
   * Updates the state and the state covariance matrix using a radar measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateRadar(MeasurementPackage meas_package);
};

#endif /* UKF_H */
