import numpy as np

def sine_data(T,dt,x0=0.0,y0=0.0,theta0=0.0):
    N = int(T * 1/dt)
    t = np.linspace(0,T-1,num=N)
    x = np.zeros_like(t)
    y = np.zeros_like(t)
    theta = np.zeros_like(t)
    for i in range(len(t)):
        x[i] = x0 + t[i]
        y[i] = y0 + np.sin(t[i])
        theta[i] = theta0 + (np.cos(t[i])) * np.pi / 4
    return x, y, theta

def line_data(T,dt,x0=0.0,y0=0.0,theta0=0.0):
    N = int(T * 1/dt)
    t = np.linspace(0,T-1,num=N)
    x = np.zeros_like(t)
    y = np.zeros_like(t)
    theta = np.zeros_like(t)
    for i in range(len(t)):
        x[i] = x0 + t[i]*np.cos(theta0)
        y[i] = y0 + t[i]*np.sin(theta0)
        theta[i] = theta0
    return x, y, theta

def square_data(T,dt,x0=0.0,y0=0.0,theta0=0.0):
    N = int(T * 1/dt)
    t = np.linspace(0,T-1,num=N)
    x = np.zeros_like(t)
    y = np.zeros_like(t)
    theta = np.zeros_like(t)
    for i in range(len(t)):
        if i < N//4: # Drive straight
            x[i] = x0 + t[i]*np.cos(theta0)
            y[i] = y0 + t[i]*np.sin(theta0)
            theta[i] = theta0
        elif i < N//2: # Drive up
            x[i] = x[i-1] + dt*np.cos(theta0 + np.pi/2)
            y[i] = y[i-1] + dt*np.sin(theta0 + np.pi/2)
            theta[i] = theta0 + np.pi / 2
        elif i < (3*N)//4: # Drive back
            x[i] = x[i-1] + dt*np.cos(theta0 + np.pi)
            y[i] = y[i-1] + dt*np.sin(theta0 + np.pi)
            theta[i] = theta0 + np.pi
        else: # Drive down
            x[i] = x[i-1] + dt*np.cos(theta0 - np.pi/2)
            y[i] = y[i-1] + dt*np.sin(theta0 - np.pi/2)
            theta[i] = theta0 - np.pi / 2
    return x, y, theta

def combine(T,dt,long=False):
    t_sub = T//3
    x_1,y_1,th_1 = square_data(t_sub, dt)
    x_2,y_2,th_2 = line_data(t_sub, dt, x_1[-1] + dt, y_1[-1], th_1[-1] + np.pi / 2)
    x_3,y_3,th_3 = square_data(t_sub, dt, x_2[-1] + dt, y_2[-1], th_2[-1])
    if long:
        x_4,y_4,th_4 = line_data(t_sub, dt, x_3[-1], y_3[-1] - dt, th_3[-1])
        x_5,y_5,th_5 = square_data(t_sub, dt, x_4[-1], y_4[-1] - dt, th_4[-1])
        x_6,y_6,th_6 = line_data(t_sub, dt, x_5[-1] - dt, y_5[-1], th_5[-1])
        x_7,y_7,th_7 = square_data(t_sub, dt, x_6[-1] - dt, y_6[-1], th_6[-1])
        x = np.hstack((x_1,x_2,x_3,x_4,x_5,x_6,x_7))
        y = np.hstack((y_1,y_2,y_3,y_4,y_5,y_6,y_7))
        theta = np.hstack((th_1,th_2,th_3,th_4,th_5,th_6,th_7))
    else:
        x = np.hstack((x_1,x_2,x_3))
        y = np.hstack((y_1,y_2,y_3))
        theta = np.hstack((th_1,th_2,th_3))
    return x,y,theta

class Landmarks():
    def __init__(self,nl,xmin=0,xmax=50,ymin=0,ymax=50):
        if ymax == 0:
            ymax = 1
        if xmax == 0:
            xmax = 1
        xs = np.random.uniform(xmin,xmax,size=nl)#np.random.randn(nl) * xmax - xmin
        ys = np.random.uniform(ymin,ymax,size=nl)
        self.nl = nl
        self.landmarks = {}
        for i in range(nl):
            self.landmarks[i] = np.array([xs[i],ys[i]])

    def detect_landmarks(self,xyth,Rs,add_noise=False):
        detected_landmarks = {}
        for i in range(self.nl):
            dlxdly = self.landmarks[i] - xyth[:2]
            rng = np.linalg.norm(dlxdly)
            if rng < Rs:
                bearing = np.arctan2(dlxdly[1],dlxdly[0]) - xyth[2]
                if add_noise:
                    rng += np.random.normal(0,0.2)
                else:
                    bearing += np.random.normal(0,0.01)
                detected_landmarks[i] = (bearing, rng)
        return detected_landmarks

