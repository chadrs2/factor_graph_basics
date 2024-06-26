from typing import Optional, List
from functools import partial

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Wedge
import matplotlib.transforms as transforms

import gtsam
from gtsam.symbol_shorthand import X, L

from gen2ddata import Landmarks, sine_data, line_data, square_data, combine

def cvt_poselist_to_arr(poselist):
    arr = np.zeros((len(poselist),3))
    for i in range(len(poselist)):
        arr[i,0] = poselist[i].x()
        arr[i,1] = poselist[i].y()
        arr[i,2] = poselist[i].theta()
    return arr

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
    for i in range(1,N):
        # Note: pi_wrt_w.matrix() = H_i_to_w
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
    factor_graph.push_back(
        gtsam.PriorFactorPose2(
            X(0),
            true_Pose2[0],
            gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-3+dtrans_p,1e-3+dtrans_p,2e-2+drot_p]))
        )
    )
    initial_estimate = gtsam.Values()
    initial_estimate.insert(X(0),raw_odometry_pose[0])
    
    ## Add Landmark detection factors
    nl = args.num_landmarks
    Rs = (abs(max(x) - min(x)))/2
    L_class = Landmarks(nl, xmin=min(x)-Rs//2, xmax=max(x)+Rs//2, ymin=min(y)-Rs//2, ymax=max(y)+Rs//2)
    true_landmarks = np.array([l for l in L_class.landmarks.values()])
    detected_landmarks = set()

    ## Begin iterative robotic loop
    for i in range(1,N):
        print("Time step:",i)
        ## Add initial factor priors
        initial_estimate.insert(X(i), raw_odometry_pose[i])

        ## Add odometry factor
        ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array(od_noise[i-1]))
        odometry_factor = gtsam.BetweenFactorPose2(X(i-1), X(i), odometry_meas[i-1], ODOMETRY_NOISE)
        factor_graph.add(odometry_factor)

        ## Detect landmarks
        BR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.01,0.2]))
        landmarks = L_class.detect_landmarks(true_pose_arr[i,:],Rs,add_noise=True)
        for key, values in landmarks.items():
            if key not in detected_landmarks:
                initial_estimate.insert(L(key), gtsam.Point2(L_class.landmarks[key][0],L_class.landmarks[key][1]))
                detected_landmarks.add(key)
            landmark_factor = gtsam.BearingRangeFactor2D(X(i),L(key),gtsam.Rot2(values[0]),values[1],BR_NOISE)
            factor_graph.add(landmark_factor)
        # factor_graph.print()

        if i > 1:
            ## Incremental Solve
            isam.update(factor_graph, initial_estimate)
            # Estimated pose of the robot at w.r.t. the world frame; H_i_to_w
            optimized_estimate = isam.calculateEstimate()

            ## Plot covariances
            if args.do_incremental_plots:
                ## Plot estimates
                fig, ax = plt.subplots()
                ax.plot(true_pose_arr[:i,0],true_pose_arr[:i,1],'b*-',label="True")
                ax.plot([true_pose_arr[:i,0],true_pose_arr[:i,0]+dt/4*np.cos(true_pose_arr[:i,2])],[true_pose_arr[:i,1],true_pose_arr[:i,1]+dt/4*np.sin(true_pose_arr[:i,2])],color='orange')
                ax.plot(raw_odometry_pose_arr[:i,0],raw_odometry_pose_arr[:i,1],'r*--',label="Raw Odometry")
                estimated_path = np.array([[optimized_estimate.atPose2(X(t)).x(), optimized_estimate.atPose2(X(t)).y()]
                                   for t in range(i)])
                ax.plot(estimated_path[:,0],estimated_path[:,1],'g*--',label="Factor Graph Estimate")
                # ax.scatter(true_landmarks[:,0],true_landmarks[:,1],marker='^',c='grey',label="Landmarks")
                
                ## Plot covariances
                for j in range(i):
                    landmarks = L_class.detect_landmarks(true_pose_arr[j,:],Rs)
                    for key, values in landmarks.items():
                        ax.plot([estimated_path[j,0],L_class.landmarks[key][0]],[estimated_path[j,1],L_class.landmarks[key][1]],linestyle='--',lw=0.5,color='grey')
                        ax.scatter(L_class.landmarks[key][0],L_class.landmarks[key][1],marker='^',c='grey')

                    cov = isam.marginalCovariance(X(j))
                    xy_cov = cov[:2,:2]
                    l1, l2, theta = compute_ellipse_radiirot(xy_cov)
                    theta += optimized_estimate.atPose2(X(j)).theta()
                    if j == 0:
                        ellipse = Ellipse(tuple([estimated_path[j,0],estimated_path[j,1]]), 
                                        width=2 * 3*np.sqrt(l1),
                                        height=2 * 3*np.sqrt(l2),
                                        angle=np.degrees(theta),
                                        alpha=0.3,
                                        color="green",label=r"xy: $\pm$3$\sigma$")
                        heading = optimized_estimate.atPose2(X(j)).theta()
                        arc = Wedge((optimized_estimate.atPose2(X(j)).x(),optimized_estimate.atPose2(X(j)).y()), 
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
                        heading = optimized_estimate.atPose2(X(j)).theta()
                        arc = Wedge((optimized_estimate.atPose2(X(j)).x(),optimized_estimate.atPose2(X(j)).y()), 
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
                plt.title(f"Noise, Odometry + {args.num_landmarks} Landmarks at t={i}")
                plt.show()
            
            ## Reset
            initial_estimate.clear()
            factor_graph.resize(0)
    
    # Build combined estimate
    optimized_estimate = isam.calculateEstimate()
    estimated_path = np.array([[optimized_estimate.atPose2(X(i)).x(), optimized_estimate.atPose2(X(i)).y()]
                                   for i in range(len(raw_odometry_pose))])
    est_landmarks = np.array([optimized_estimate.atPoint2(L(i)) for i in range(nl)])
    
    ## Plot estimates
    print("Final results plot")
    fig, ax = plt.subplots()
    ax.plot(true_pose_arr[:,0],true_pose_arr[:,1],'b*-',label="True")
    ax.plot(raw_odometry_pose_arr[:,0],raw_odometry_pose_arr[:,1],'r*--',label="Raw Odometry")
    ax.plot(estimated_path[:,0],estimated_path[:,1],'g*--',label="Factor Graph Estimate")
    # ax.scatter(est_landmarks[:,0],est_landmarks[:,1],marker='^',c='g',label="Factor Graph Landmark Estimate")
    ax.scatter(true_landmarks[:,0],true_landmarks[:,1],marker='^',c='grey',label="Landmarks")
    for i in range(N):
        landmarks = L_class.detect_landmarks(true_pose_arr[i,:],Rs)
        for key, values in landmarks.items():
            ax.plot([estimated_path[i,0],L_class.landmarks[key][0]],[estimated_path[i,1],L_class.landmarks[key][1]],linestyle='--',lw=0.5,color='grey')
    ## Plot covariances
    for i in range(N):
        cov = isam.marginalCovariance(X(i))
        xy_cov = cov[:2,:2]
        l1, l2, theta = compute_ellipse_radiirot(xy_cov)
        theta += optimized_estimate.atPose2(X(i)).theta()
        if i == 0:
            ellipse = Ellipse(tuple([estimated_path[i,0],estimated_path[i,1]]), 
                            width=2 * 3*np.sqrt(l1),
                            height=2 * 3*np.sqrt(l2),
                            angle=np.degrees(theta),
                            alpha=0.3,
                            color="green",label=r"xy: $\pm$3$\sigma$")
            heading = optimized_estimate.atPose2(X(i)).theta()
            arc = Wedge((optimized_estimate.atPose2(X(i)).x(),optimized_estimate.atPose2(X(i)).y()), 
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
            heading = optimized_estimate.atPose2(X(i)).theta()
            arc = Wedge((optimized_estimate.atPose2(X(i)).x(),optimized_estimate.atPose2(X(i)).y()), 
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
    plt.title(f"Noise, Odometry + {args.num_landmarks} Landmarks")
    plt.show()
    
def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type',default="square",help="Simulated data: ['square', 'sine', 'line', 'lsqr']")
    parser.add_argument('--seed',default=114,type=int,help="Numpy random seed")
    parser.add_argument('--do_incremental_plots',action='store_true')
    parser.add_argument('--num_landmarks',default=8,type=int,help="Number of landmarks to place in environment")
    parser.add_argument('--Tmax',default=20,type=float,help="Number of seconds to run simulation")
    parser.add_argument('--dt',default=0.5,type=float,help='Time difference between odometry factors')
    parser.add_argument('--translational_noise',default=0.01,type=float,help='Translational noise percentage')
    parser.add_argument('--rotational_noise',default=0.15,type=float,help='Rotational noise percentage')
    return parser

if __name__ == "__main__":
    args = parser().parse_args()
    main(args)