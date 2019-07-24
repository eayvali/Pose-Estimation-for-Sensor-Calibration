## POSE ESTIMATION FOR SENSOR CALIBRATION

Many sensor calibration and pose estimation problems in robotics require solving equations of the form AX=XB or AX=YB to obtain the homogeneous transforms X and Y from a set of (A,B) paired measurements.

![Example Applications](./Figures/pose_estimation_examples.png)

Pose_Estimation.py has several implementations that can be used based on the application needs.

## Batch Processing

Batch processing class formulates the problem using Kronecker product and finds the least square solution using n pairs of data. If there is no sensor noise, theoretically, you need 3 unique poses (rotation axis) to get a solution. In practice, there is almost always sensor noise.

![Preliminaries for Batch Processing Solution](./Figures/Kronecker_Product.png)
![Least-square estimation](./Figures/LSE_solution.png)

## Iterated Extended Kalman Filter 
EKf is well-suited for applications where it takes significant time to collect and process paired measurements. The implementation consumes a single pair of measurements to update the state at every time step and requires many iterations to converge.  For highly nonlinear functions such as this application, where we estimate the pose, EKF can significantly underestimate the covariance. If the estimated state is not close the true state, the first-order approximation will no longer be a good approximation. To alleviate this, an iterated extended kalman filter (IEKF) can be used. The main difference between EKF and IEKF is the measurement update step. IEKF iterates over the measurement update step for a fixed number of iterations or until a stopping criterion is met. Innovation or the norm of the difference between the consequent state estimates can be used as a stopping criterion. It's important to note that, the state covariance matrix is kept fixed during this iterative step and only the state estimate is iterated. If we were to also update the covariance matrix, it would mean we process two identical measurements.  Once the iterations end, we use the same EKF update equations to update the state and covariance. Please see [1] for the mathematical formulation of the EKF and IEKF. The implementation here follows the same notation. 

![IEKF implementations](./Figures/IEKF.png)

State parametrization is also very important. See the summary table below. I highly recommend reading [2] and [3] to understand representations of rotations. The implementation for EKF and IEKF uses so3 parameters to represent rotation. Last but not least, remember to check the consistency of the filter to make sure the filter is doing its job.

![Representations of Rotation](./Figures/Rotation_representations.png)

## References
_[1]_  Havlík, Jindřich, and Ondřej Straka. "Performance evaluation of iterated extended Kalman filter with variable step-length." Journal of Physics: Conference Series. Vol. 659. No. 1. IOP Publishing, 2015.

_[2]_ Blanco, Jose-Luis. "A tutorial on se (3) transformation parameterizations and on-manifold optimization." University of Malaga, Tech. Rep 3 (2010).

_[3]_ Grassia, F. Sebastian. "Practical parameterization of rotations using the exponential map." Journal of graphics tools 3.3 (1998): 29-48.


