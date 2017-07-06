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

  // initially set to false, set to true in first call of ProcessMeasurement
  bool is_initialized_;

  // for initialization of state vector at startup:
  int init_cnt_;
  int init_sensor_;
  float px_;        // previous x
  float py_;        // previous y
  int time_step_;

  //previous timestamp
  long long previous_timestamp_;

  //if this is false, laser measurements will be ignored (except for init)
  bool use_laser_;

  //if this is false, radar measurements will be ignored (except for init)
  bool use_radar_;

  ///* State variables
  // state vector: [pos_x pos_y vel_abs yaw_angle yaw_rate] in SI units and rad
  VectorXd x_;

  // state covariance matrix
  MatrixXd P_;

  // predicted sigma points matrix
  MatrixXd Xsig_pred_;

  // augmented sigma points matrix
  MatrixXd Xsig_aug_;

  // state dimension
  int n_x_;

  // augmented state dimension
  int n_aug_;

  // number of UKF sigma points
  int n_sig_;

  // weights of sigma points
  VectorXd weights_;

  // sigma point spreading parameter
  double lambda_;

  // process noise standard deviation longitudinal acceleration in m/s^2
  double std_a_;

  // process noise standard deviation yaw acceleration in rad/s^2
  double std_yawdd_;

  ///* Lidar measurement variables
  // lidar measurement dimension
  int n_l_;

  // lidar measurement matrix
  // projects the state space, e.g. 5D {x, y, v, yaw, yaw_dot}, into the lidar measurement space, e.g. 2D (x, y)
  MatrixXd H_lidar_;

  // lidar measurement noise standard deviation position1 in m
  double std_laspx_;

  // lidar measurement noise standard deviation position2 in m
  double std_laspy_;

  // lidar measurement covariance
  MatrixXd R_lidar_;

  // lidar NIS (normalized innovation squared)
  double NIS_lidar_;


  ///* Radar measurement variables
  // radar measurement dimension
  int n_r_;

  // radar measurement noise standard deviation radius in m
  double std_radr_;

  // radar measurement noise standard deviation angle in rad
  double std_radphi_;

  // radar measurement noise standard deviation radius change in m/s
  double std_radrd_ ;

  // predicted radar measurement covariance
  MatrixXd R_radar_;

  // radar NIS (normalized innovation squared)
  double NIS_radar_;

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

  void PredictRadarMeasurement(const MatrixXd &Xsig_pred, MatrixXd &Rsig_pred, VectorXd &r_pred);

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
   * @param VectorXd The lidar measurement z at k+1
   */
  void UpdateLidar(const VectorXd &z);

  /**
   * Updates the state and the state covariance matrix using a laser measurement
   * @param VectorXd The lidar measurement z at k+1
   */
  void UpdateLidar_KF(const VectorXd &z);

  /**
   * Updates the state and the state covariance matrix using a radar measurement
   * @param VectorXd The radar measurement z at k+1
   */
  void UpdateRadar(const VectorXd &z);
};

#endif /* UKF_H */
