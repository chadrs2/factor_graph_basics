from typing import Optional, List
from functools import partial

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Wedge

import gtsam
# from gtsam.symbol_shorthand import A

from gen2ddata import sine_data, line_data, square_data, combine


def cvt_poselist_to_arr(poselist):
    arr = np.zeros((len(poselist),3))
    for i in range(len(poselist)):
        arr[i,0] = poselist[i].x()
        arr[i,1] = poselist[i].y()
        arr[i,2] = poselist[i].theta()
    return arr

def error_gps2(measurement: np.ndarray, 
               this: gtsam.CustomFactor,
               values: gtsam.Values,
               jacobians: Optional[List[np.ndarray]]) -> float:
    """GPS Factor error function
    :param measurement: GPS measurement, to be filled with `partial`
    :param this: gtsam.CustomFactor handle
    :param values: gtsam.Values
    :param jacobians: Optional list of Jacobians
    :return: the unwhitened error
    """

    # Get agent pose
    key = this.keys()[0]
    estimate = values.atPose2(key)

    # Three rows bc [x, y] from gps state, three columns bc the state of this node is Pose2
    H = np.zeros((2, 3))
    eps = 1e-6

    # Get Jacobians. Perturb in each direction of the delta vector one at a time.
    for i in range(3):
        delta_step = np.zeros(3)
        delta_step[i] = eps
        delta_step_forward = gtsam.Pose2.Expmap(delta_step)
        delta_step_backward = gtsam.Pose2.Expmap(-delta_step)

        q_exp_forward = estimate.compose(delta_step_forward)
        q_exp_backward = estimate.compose(delta_step_backward)
        h_forward = q_exp_forward.translation()
        h_backward = q_exp_backward.translation()
        error_forward = h_forward - measurement
        error_backward = h_backward - measurement
        hdot =  (error_forward - error_backward) / (2*eps)
        H[:,i] = hdot
    
    error = estimate.translation() - measurement

    if jacobians is not None:
        # Error wrt agent A
        jacobians[0] = H

    return error

def compute_ellipse_radiirot(cov):
    # See https://cookierobotics.com/007/ for info on equations
    a = cov[0,0]; b = cov[0,1]; c = cov[1,1]
    l1 = abs((a+c)/2 + np.sqrt(((a-c) / 2)**2 + b**2))
    l2 = abs((a+c)/2 - np.sqrt(((a-c) / 2)**2 + b**2))
    if abs(b) <= 1e-10 and a >= c:
        theta = 0.0
    elif abs(b) <= 1e-10 and a < c:
        theta = np.pi/2
    else:
        theta = np.arctan2(l1-a, b)
    return l1, l2, theta

def main(args):
    np.random.seed(args.seed)
    use_GPS = args.use_gps
    do_loopclosure = args.do_loopclosure

    ## Generate Simulated Data
    dt = args.dt
    Tmax = args.Tmax
    if args.data_type == 'line':
        x,y,theta = line_data(Tmax, dt)
    elif args.data_type == 'sine':
        x,y,theta = sine_data(Tmax, dt)
    elif args.data_type == 'lsqr':
        x,y,theta = combine(Tmax, dt, long=True)
    else:
        x,y,theta = square_data(Tmax, dt)
    # plt.figure()
    # plt.scatter(x,y,marker="^")
    # for i in range(len(theta)):
    #     plt.text(x[i],y[i],s=f"{round(np.degrees(theta[i]),2)} deg")
    # plt.show()
    N = len(x)
    dtrans_p = args.translational_noise # noise percent in position
    drot_p = args.rotational_noise # noise percent in rotation  
    
    ## Define true poses in World Frame
    true_pose_arr = np.vstack((x,y,theta)).T # N x 3
    true_Pose2 = []
    for i in range(N):
        # Pose of the robot at t=i w.r.t. the world frame; H_i_to_w
        true_Pose2.append(gtsam.Pose2(true_pose_arr[i,0],true_pose_arr[i,1],true_pose_arr[i,2]))
    
    ## Build Odometry measurements
    odometry_meas = []
    od_noise = []
    for i in range(1,N+1 if do_loopclosure else N):
        # Note: pi_wrt_w.matrix() = H_i_to_w
        if do_loopclosure and i == N:
            pi_wrt_w = true_Pose2[0]
        else:
            pi_wrt_w = true_Pose2[i] # Pose of the robot at t=i w.r.t. the world frame
        pimin1_wrt_w = true_Pose2[i-1] # Pose of the robot at t=i-1 w.r.t. the world frame
        H_i_to_imin1 = pimin1_wrt_w.inverse().compose(pi_wrt_w) # Transformation from pose i to pose i-1
        ## Add noise
        dx_noise = abs(H_i_to_imin1.x()) * dtrans_p + 1e-3
        dy_noise = abs(H_i_to_imin1.y()) * dtrans_p + 1e-3
        dtheta_noise = abs(H_i_to_imin1.theta()) * drot_p + 2e-2
        od_noise.append([dx_noise,dy_noise,dtheta_noise])
        noise = np.random.normal([0,0,0],od_noise[-1])
        delta_noise = gtsam.Pose2(noise[0],noise[1],noise[2])
        odometry_meas.append(H_i_to_imin1.compose(delta_noise))
    
    ## Estimated Trajectory from raw Odometry
    raw_odometry_pose = []
    for i in range(N):
        if i == 0:
            raw_odometry_pose.append(true_Pose2[i]) # H_0_to_w
        else:
            H_imin1_to_w = raw_odometry_pose[-1]
            H_i_to_imin1 = odometry_meas[i-1]
            raw_odometry_pose.append(H_imin1_to_w.compose(H_i_to_imin1)) # H_i_to_w
    raw_odometry_pose_arr = cvt_poselist_to_arr(raw_odometry_pose)

    ## Create factor graph
    isam = gtsam.ISAM2()
    factor_graph = gtsam.NonlinearFactorGraph()
    ## Prior on first pose at origin
    factor_graph.push_back(
        gtsam.PriorFactorPose2(
            0,
            true_Pose2[0],
            gtsam.noiseModel.Diagonal.Sigmas(np.array(od_noise[0]))))
    initial_estimate = gtsam.Values()
    initial_estimate.insert(0,raw_odometry_pose[0])
    ## Add GPS factors
    if use_GPS:
        GPS_NOISE = gtsam.noiseModel.Isotropic.Sigma(2, 0.1) # x,y noise
        gps_meas = gtsam.Point2(true_Pose2[0].translation() + np.random.normal(0,0.1,size=(2,)))
        gps_factor = gtsam.CustomFactor(GPS_NOISE, [0], partial(error_gps2, gps_meas))
        factor_graph.add(gps_factor)

    # Begin iterative robotic loop
    for i in range(1,N):
        print("Time step:",i)
        ## Add initial factor priors
        initial_estimate.insert(i, raw_odometry_pose[i])
        
        ## Add odometry factor
        ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array(od_noise[i-1]))
        odometry_factor = gtsam.BetweenFactorPose2(i-1, i, odometry_meas[i-1], ODOMETRY_NOISE)
        factor_graph.add(odometry_factor)

        ## Add GPS factors
        if use_GPS:
            GPS_NOISE = gtsam.noiseModel.Isotropic.Sigma(2, 0.1) # x,y noise
            if i % 2 == 0:
                gps_meas = gtsam.Point2(true_Pose2[i].translation() + np.random.normal(0,0.1,size=(2,)))
                gps_factor = gtsam.CustomFactor(GPS_NOISE, [i], partial(error_gps2, gps_meas))
                factor_graph.add(gps_factor)

        if i > 1:
            ## Incremental Solve
            isam.update(factor_graph, initial_estimate)
            # Estimated pose of the robot at w.r.t. the world frame; H_i_to_w
            optimized_estimate = isam.calculateEstimate()

            ## Plot covariances
            # marginals = gtsam.Marginals(isam.getFactorsUnsafe(), optimized_estimate)
            if args.do_incremental_plots:
                ## Plot estimates
                fig, ax = plt.subplots()
                ax.plot(true_pose_arr[:i,0],true_pose_arr[:i,1],'b*-',label="True")
                ax.plot(raw_odometry_pose_arr[:i,0],raw_odometry_pose_arr[:i,1],'r*--',label="Raw Odometry")
                estimated_path = np.array([[optimized_estimate.atPose2(t).x(), optimized_estimate.atPose2(t).y()]
                                   for t in range(i)])
                ax.plot(estimated_path[:,0],estimated_path[:,1],'g*--',label="Factor Graph Estimate")
                ## Plot covariances
                for j in range(i):
                    # Get rotation from local frame to world frame
                    R_i_to_w = optimized_estimate.atPose2(j).matrix()[:2,:2]
                    cov = isam.marginalCovariance(j)
                    xy_cov = R_i_to_w @ cov[:2,:2]
                    l1, l2, theta = compute_ellipse_radiirot(xy_cov)
                    if j == 0:
                        ellipse = Ellipse(tuple([estimated_path[j,0],estimated_path[j,1]]), 
                                        width=2 * 3*np.sqrt(l1),
                                        height=2 * 3*np.sqrt(l2),
                                        angle=np.degrees(theta),
                                        alpha=0.3,
                                        color="green",label=r"xy: $\pm$3$\sigma$")
                        heading = optimized_estimate.atPose2(j).theta()
                        arc = Wedge((optimized_estimate.atPose2(j).x(),optimized_estimate.atPose2(j).y()), 
                            r = dt/2,
                            theta1=np.degrees(heading - 3*np.sqrt(cov[2,2])), 
                            theta2=np.degrees(heading + 3*np.sqrt(cov[2,2])),
                            color='orange', alpha=0.8,
                            label=r'$\theta$: $\pm$3$\sigma$')
                    else:
                        ellipse = Ellipse(tuple([estimated_path[j,0],estimated_path[j,1]]), 
                                    width=2 * 3*np.sqrt(l1),
                                    height=2 * 3*np.sqrt(l2),
                                    angle=np.degrees(theta),
                                    alpha=0.3,
                                    color="green")
                        heading = optimized_estimate.atPose2(j).theta()
                        arc = Wedge((optimized_estimate.atPose2(j).x(),optimized_estimate.atPose2(j).y()), 
                            r = dt/2,
                            theta1=np.degrees(heading - 3*np.sqrt(cov[2,2])), 
                            theta2=np.degrees(heading + 3*np.sqrt(cov[2,2])),
                            color='orange', alpha=0.8)
                    ax.add_patch(ellipse)
                    ax.add_patch(arc)
                plt.axis('equal')
                plt.xlabel("X")
                plt.ylabel("Y")
                plt.legend()
                if use_GPS and do_loopclosure:
                    plt.title("iSAM: Noise, Odometry and GPS, Loop-Closure")
                elif use_GPS:
                    plt.title("iSAM: Noise, Odometry and GPS")
                elif do_loopclosure:
                    plt.title("iSAM: Noise, Odometry, Loop-Closure")
                else:
                    plt.title("iSAM: Noise, Odometry")
                plt.show()
            
            ## Reset
            initial_estimate.clear()
            factor_graph.resize(0)

    if do_loopclosure:
        print("Doing loop closure")
        ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array(od_noise[-1]))
        odometry_factor = gtsam.BetweenFactorPose2(N-1, 0, odometry_meas[-1], ODOMETRY_NOISE)
        factor_graph.add(odometry_factor)    
        isam.update(factor_graph, initial_estimate)

    # Build combined estimate
    optimized_estimate = isam.calculateEstimate()
    estimated_path = np.array([[optimized_estimate.atPose2(i).x(), optimized_estimate.atPose2(i).y()]
                                   for i in range(len(raw_odometry_pose))])
    
    ## Plot estimates
    print("Final results plot")
    fig, ax = plt.subplots()
    ax.plot(true_pose_arr[:,0],true_pose_arr[:,1],'b*-',label="True")
    ax.plot(raw_odometry_pose_arr[:,0],raw_odometry_pose_arr[:,1],'r*--',label="Estimate")
    ax.plot(estimated_path[:,0],estimated_path[:,1],'g*--',label="Factor Graph Estimate")
    ## Plot covariances
    for i in range(N):
        R_i_to_w = optimized_estimate.atPose2(i).matrix()[:2,:2]
        cov = isam.marginalCovariance(i)
        xy_cov = R_i_to_w @ cov[:2,:2]
        l1, l2, theta = compute_ellipse_radiirot(xy_cov)
        if i == 0:
            ellipse = Ellipse(tuple([estimated_path[i,0],estimated_path[i,1]]), 
                            width=2 * 3*np.sqrt(l1),
                            height=2 * 3*np.sqrt(l2),
                            angle=np.degrees(theta),
                            alpha=0.3,
                            color="green",label=r"xy: $\pm$3$\sigma$")
            heading = optimized_estimate.atPose2(i).theta()
            arc = Wedge((optimized_estimate.atPose2(i).x(),optimized_estimate.atPose2(i).y()), 
                r = dt/2,
                theta1=np.degrees(heading - 3*np.sqrt(cov[2,2])), 
                theta2=np.degrees(heading + 3*np.sqrt(cov[2,2])),
                color='orange', alpha=0.8,
                label=r'$\theta$: $\pm$3$\sigma$')
        else:
            ellipse = Ellipse(tuple([estimated_path[i,0],estimated_path[i,1]]), 
                        width=2 * 3*np.sqrt(l1),
                        height=2 * 3*np.sqrt(l2),
                        angle=np.degrees(theta),
                        alpha=0.3,
                        color="green")
            heading = optimized_estimate.atPose2(i).theta()
            arc = Wedge((optimized_estimate.atPose2(i).x(),optimized_estimate.atPose2(i).y()), 
                r = dt/2,
                theta1=np.degrees(heading - 3*np.sqrt(cov[2,2])), 
                theta2=np.degrees(heading + 3*np.sqrt(cov[2,2])),
                color='orange', alpha=0.8)
        ax.add_patch(ellipse)
        ax.add_patch(arc)
    plt.axis('equal')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    if use_GPS and do_loopclosure:
        plt.title("iSAM: Noise, Odometry and GPS, Loop-Closure")
    elif use_GPS:
        plt.title("iSAM: Noise, Odometry and GPS")
    elif do_loopclosure:
        plt.title("iSAM: Noise, Odometry, Loop-Closure")
    else:
        plt.title("iSAM: Noise, Odometry")
    plt.show()
    
def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type',default="square",help="Simulated data: ['square', 'sine', 'line', 'lsqr']")
    parser.add_argument('--seed',default=114,type=int,help="Numpy random seed")
    parser.add_argument('--do_incremental_plots',action='store_true')
    parser.add_argument('--use_gps',action='store_true')
    parser.add_argument('--do_loopclosure',action='store_true')
    parser.add_argument('--Tmax',default=20,type=float,help="Number of seconds to run simulation")
    parser.add_argument('--dt',default=0.5,type=float,help='Time difference between odometry factors')
    parser.add_argument('--translational_noise',default=0.025,type=float,help='Translational noise percentage')
    parser.add_argument('--rotational_noise',default=0.05,type=float,help='Rotational noise percentage')
    return parser

if __name__ == "__main__":
    args = parser().parse_args()
    main(args)