from scipy import linalg as la
import matplotlib.pyplot as pl
import numpy as np
import math

from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

import quadrotor as quad
import formation_distance as form
import quadlog
import animation as ani
import uwb_agent as range_agent

def get_dist_clean(p1, p2):
    return (np.linalg.norm(p1 - p2))

def get_dist(p1, p2):
    mu, sigma = 0, 2
    std_err = np.random.normal(mu, sigma, 1)[0]
    return (np.linalg.norm(p1 - p2)) + std_err

def vecLen(v):
    return math.sqrt(np.dot(v,v))

def getAngle(v1, v2):
    return np.arctan( (np.dot(v1, v2)) / (vecLen(v1)*vecLen(v2)) )

def getRotMat(q1, q2, q3, A, B, C):
    v1q = abs(q1 - q2)
    v1a = abs(A - B)
    v2q = abs(q1 - q3)
    v2a = abs(A - C)
    v3q = abs(q2 - q3)
    v3a = abs(B - C)

    a = (getAngle(v1q, v1a) + getAngle(v2q, v2a) + getAngle(v3q, v3a)) / 3
    #a = np.array([getAngle(v1q, v1a), getAngle(v2q, v2a), getAngle(v3q, v3a)])

    m11 = np.cos(a)
    m12 = np.sin(a)
    m21 = -np.sin(a)
    m22 = np.cos(a)

    return np.array([ [m11, m12],
                       [m21, m22]])

# Quadrotor
m = 0.65 # Kg
l = 0.23 # m
Jxx = 7.5e-3 # Kg/m^2
Jyy = Jxx
Jzz = 1.3e-2
Jxy = 0
Jxz = 0
Jyz = 0
J = np.array([[Jxx, Jxy, Jxz], \
              [Jxy, Jyy, Jyz], \
              [Jxz, Jyz, Jzz]])
CDl = 9e-3
CDr = 9e-4
kt = 3.13e-5  # Ns^2
km = 7.5e-7   # Ns^2
kw = 1/0.18   # rad/s

# Initial conditions
att_0 = np.array([0.0, 0.0, 0.0])
pqr_0 = np.array([0.0, 0.0, 0.0])
xyz1_0 = np.array([0.0, 0.0, 0.0])
xyz2_0 = np.array([5.0, 0.0, 0.0])
xyz3_0 = np.array([3.0, 4.5, 0.0])
v_ned_0 = np.array([0.0, 0.0, 0.0])
w_0 = np.array([0.0, 0.0, 0.0, 0.0])

# Setting quads
q1 = quad.quadrotor(1, m, l, J, CDl, CDr, kt, km, kw, \
        att_0, pqr_0, xyz1_0, v_ned_0, w_0)

q2 = quad.quadrotor(2, m, l, J, CDl, CDr, kt, km, kw, \
        att_0, pqr_0, xyz2_0, v_ned_0, w_0)

q3 = quad.quadrotor(3, m, l, J, CDl, CDr, kt, km, kw, \
        att_0, pqr_0, xyz3_0, v_ned_0, w_0)

# Simulation parameters
tf = 800
dt = 5e-2
time = np.linspace(0, tf, tf/dt)
it = 0
frames = 100

# Data log
q1_log = quadlog.quadlog(time)
q2_log = quadlog.quadlog(time)
q3_log = quadlog.quadlog(time)
Ed_log = np.zeros((time.size, 3))

# Plots
quadcolor = ['r', 'g', 'b']
pl.close("all")
pl.ion()
fig = pl.figure(0)
axis3d = fig.add_subplot(111, projection='3d')

init_area = 10
s = 2

# Desired altitude and heading
alt_d = 4
q1.yaw_d = -np.pi
q2.yaw_d =  np.pi/2
q3.yaw_d =  0

RA1 = range_agent.uwb_agent( ID=0 )
RA2 = range_agent.uwb_agent( ID=1 )
RA3 = range_agent.uwb_agent( ID=2 )

for t in time:
    RA1.handle_range_msg(Id=RA2.id, range=get_dist(q1.xyz[0:2], q2.xyz[0:2]))
    RA1.handle_range_msg(Id=RA3.id, range=get_dist(q1.xyz[0:2], q3.xyz[0:2]))
    RA1.handle_other_msg(Id1=RA2.id, Id2=RA3.id, range=get_dist(q2.xyz[0:2], q3.xyz[0:2]))

    RA2.handle_range_msg(Id=RA3.id, range=get_dist(q2.xyz[0:2], q3.xyz[0:2]))
    RA2.handle_range_msg(Id=RA1.id, range=get_dist(q2.xyz[0:2], q1.xyz[0:2]))
    RA2.handle_other_msg(Id1=RA1.id, Id2=RA3.id, range=get_dist(q1.xyz[0:2], q3.xyz[0:2]))

    RA3.handle_range_msg(Id=RA1.id, range=get_dist(q3.xyz[0:2], q1.xyz[0:2]))
    RA3.handle_range_msg(Id=RA2.id, range=get_dist(q3.xyz[0:2], q2.xyz[0:2]))
    RA3.handle_other_msg(Id1=RA1.id, Id2=RA2.id, range=get_dist(q1.xyz[0:2], q2.xyz[0:2]))

    A1,B1,C1 = RA1.define_triangle()
    A2,B2,C2 = RA2.define_triangle()
    A3,B3,C3 = RA3.define_triangle()

    X1 = np.array([A1[0], -A1[1], B1[0], -B1[1], C1[0], -C1[1]])
    X2 = np.array([A2[0], -A2[1], B2[0], -B2[1], C2[0], -C2[1]])
    X3 = np.array([A3[0], -A3[1], B3[0], -B3[1], C3[0], -C3[1]])

    rotmat1 = getRotMat(q1.xyz[0:2], q2.xyz[0:2], q3.xyz[0:2], X1[0:2], X1[2:4], X1[4:6])
    rotmat2 = getRotMat(q1.xyz[0:2], q2.xyz[0:2], q3.xyz[0:2], X2[2:4], X2[4:6], X2[0:2])
    rotmat3 = getRotMat(q1.xyz[0:2], q2.xyz[0:2], q3.xyz[0:2], X3[4:6], X3[0:2], X3[2:4])

    U = RA1.calc_u_acc(q1.v_ned, q2.v_ned, q3.v_ned)

    u1 = rotmat1.dot(U[0:2])
    u2 = rotmat1.dot(U[2:4])
    u3 = rotmat1.dot(U[4:6])

    U = np.array([u1[0], u1[1], u2[0], u2[1], u3[0], u3[1]])

    #Using lyapunov, input 2D acc, and desired alt
    q1.set_v_2D_alt_lya(U[0:2], -alt_d)
    q2.set_v_2D_alt_lya(U[2:4], -alt_d)
    q3.set_v_2D_alt_lya(U[4:6], -alt_d)

    q1.step(dt)
    q2.step(dt)
    q3.step(dt)

    # Animation
    if it%frames == 0:

        pl.figure(0)
        axis3d.cla()
        ani.draw3d(axis3d, q1.xyz, q1.Rot_bn(), quadcolor[0])
        ani.draw3d(axis3d, q2.xyz, q2.Rot_bn(), quadcolor[1])
        ani.draw3d(axis3d, q3.xyz, q3.Rot_bn(), quadcolor[2])
        axis3d.set_xlim(-15, 15)
        axis3d.set_ylim(-15, 15)
        axis3d.set_zlim(0, 15)
        axis3d.set_xlabel('South [m]')
        axis3d.set_ylabel('East [m]')
        axis3d.set_zlabel('Up [m]')
        axis3d.set_title("Time %.3f s" %t)
        pl.pause(0.001)
        pl.draw()
        #namepic = '%i'%it
        #digits = len(str(it))
        #for j in range(0, 5-digits):
        #    namepic = '0' + namepic
        #pl.savefig("./images/%s.png"%namepic)


    # Log
    q1_log.xyz_h[it, :] = q1.xyz
    q1_log.att_h[it, :] = q1.att
    q1_log.w_h[it, :] = q1.w
    q1_log.v_ned_h[it, :] = q1.v_ned

    q2_log.xyz_h[it, :] = q2.xyz
    q2_log.att_h[it, :] = q2.att
    q2_log.w_h[it, :] = q2.w
    q2_log.v_ned_h[it, :] = q2.v_ned

    q3_log.xyz_h[it, :] = q3.xyz
    q3_log.att_h[it, :] = q3.att
    q3_log.w_h[it, :] = q3.w
    q3_log.v_ned_h[it, :] = q3.v_ned

    Ed_log[it, :] = np.array([get_dist_clean(q1.xyz[0:2],q2.xyz[0:2])-15,
                                get_dist_clean(q1.xyz[0:2],q3.xyz[0:2])-15,
                                get_dist_clean(q3.xyz[0:2],q2.xyz[0:2])-15])

    it+=1

    # Stop if crash
    if (q1.crashed == 1 or q2.crashed == 1 or q3.crashed == 1):
        break

pl.figure(1)
pl.title("2D Position [m]")
pl.plot(q1_log.xyz_h[:, 0], q1_log.xyz_h[:, 1], label="q1", color=quadcolor[0])
pl.plot(q2_log.xyz_h[:, 0], q2_log.xyz_h[:, 1], label="q2", color=quadcolor[1])
pl.plot(q3_log.xyz_h[:, 0], q3_log.xyz_h[:, 1], label="q3", color=quadcolor[2])
pl.xlabel("East")
pl.ylabel("South")
pl.legend()

pl.figure(2)
pl.plot(time, q1_log.att_h[:, 2], label="yaw q1")
pl.plot(time, q2_log.att_h[:, 2], label="yaw q2")
pl.plot(time, q3_log.att_h[:, 2], label="yaw q3")
pl.xlabel("Time [s]")
pl.ylabel("Yaw [rad]")
pl.grid()
pl.legend()

pl.figure(3)
pl.plot(time, -q1_log.xyz_h[:, 2], label="$q_1$")
pl.plot(time, -q2_log.xyz_h[:, 2], label="$q_2$")
pl.plot(time, -q3_log.xyz_h[:, 2], label="$q_3$")
pl.xlabel("Time [s]")
pl.ylabel("Altitude [m]")
pl.grid()
pl.legend(loc=2)

pl.figure(4)
pl.plot(time, Ed_log[:, 0], label="$e_1$")
pl.plot(time, Ed_log[:, 1], label="$e_2$")
pl.plot(time, Ed_log[:, 2], label="$e_3$")
pl.xlabel("Time [s]")
pl.ylabel("Formation distance error [m]")
pl.grid()
pl.legend()

pl.pause(0)
