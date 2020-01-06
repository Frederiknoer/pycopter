from scipy import linalg as la
import matplotlib.pyplot as pl
import numpy as np
from shapely.geometry import Point, LineString, Polygon
import math

import quadrotor as quad
import formation_distance as form
import quadlog
import animation as ani
import uwb_agent as range_agent


def get_dist(p1, p2):
    p1 = Point(p1[0], p1[1])
    p2 = Point(p2[0], p2[1])
    dist = math.sqrt( (p2.x - p1.x)**2 + (p2.y - p1.y)**2 )
    return dist

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
xyz1_0 = np.array([1.0, 1.2, 0.0])
xyz2_0 = np.array([1.2, 2.0, 0.0])
xyz3_0 = np.array([-1.1, 2.6, 0.0])
v_ned_0 = np.array([0.0, 0.0, 0.0])
w_0 = np.array([0.0, 0.0, 0.0, 0.0])

# Setting quads
q1 = quad.quadrotor(1, m, l, J, CDl, CDr, kt, km, kw, \
        att_0, pqr_0, xyz1_0, v_ned_0, w_0)

q2 = quad.quadrotor(2, m, l, J, CDl, CDr, kt, km, kw, \
        att_0, pqr_0, xyz2_0, v_ned_0, w_0)

q3 = quad.quadrotor(3, m, l, J, CDl, CDr, kt, km, kw, \
        att_0, pqr_0, xyz3_0, v_ned_0, w_0)

# Formation Control
# Shape
side = 8
Btriang = np.array([[1, 0, -1],[-1, 1, 0],[0, -1, 1]])
dtriang = np.array([side, side, side])

# Motion
mu = 0e-2*np.array([1, 1, 1])
tilde_mu = 0e-2*np.array([1, 1, 1])

fc = form.formation_distance(2, 1, dtriang, mu, tilde_mu, Btriang, 5e-2, 5e-1)

# Simulation parameters
tf = 300
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

RA1 = range_agent.uwb_agent(ID=0, pos=Point( q1.xyz[0], q1.xyz[1] ))
RA2 = range_agent.uwb_agent(ID=1, pos=Point( q2.xyz[0], q2.xyz[1] ))
RA3 = range_agent.uwb_agent(ID=2, pos=Point( q3.xyz[0], q3.xyz[1] ))

for t in time:

    # Simulation
    #X = np.append(q1.xyz[0:2], np.append(q2.xyz[0:2], q3.xyz[0:2]))
    V = np.append(q1.v_ned[0:2], np.append(q2.v_ned[0:2], q3.v_ned[0:2]))

    #NEW:
    RA1.update_pos(q1.xyz[0:2])
    RA2.update_pos(q2.xyz[0:2])
    RA3.update_pos(q3.xyz[0:2])

    RA1.handle_range_msg(Id=RA2.id, nb_pos=RA2.pos)
    RA1.handle_range_msg(Id=RA3.id, nb_pos=RA3.pos)

    RA1.handle_other_msg(Id1=RA2.id, Id2=RA3.id, range=get_dist(RA2.pos, RA3.pos))

    A,B,C = RA1.define_triangle()


    #fc = form.formation_distance(2, 1, dtriang, mu, tilde_mu, RA1.get_B(), 5e-2, 5e-1)

    X = [A.x, A.y, B.x, B.y, C.x, C.y]
    '''
    print("Ground Truth: ")
    print(q1.xyz[0:2])
    print(q2.xyz[0:2])
    print(A.x, A.y)
    print(B.x, B.y)
    '''
    #print(fc)
    #q2x = ((abs(A.x-B.x)) - (side))/5 + ((abs(C.x-B.x)) - (side))/5
    #q2y = ((abs(A.y-B.y)) - (side))/5 + ((abs(C.y-B.y)) - (side))/5
    #q3x = ((abs(A.x-C.x)) - (side))/5 + ((abs(B.x-C.x)) - (side))/5
    #q3y = ((abs(A.y-C.y)) - (side))/5 + ((abs(B.y-C.y)) - (side))/5

    #Set This U (Input) from another class
    U = RA1.calc_u_acc()

    #print("U: ", U)


    #Using lyapunov, input 2D acc, and desired alt
    q1.set_a_2D_alt_lya(U[0:2], -alt_d)
    q2.set_a_2D_alt_lya(U[2:4], -alt_d)
    q3.set_a_2D_alt_lya(U[4:6], -alt_d)

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

        pl.figure(1)
        pl.clf()
        ani.draw2d(1, X, fc, quadcolor)
        ani.draw_edges(1, X, fc, -1)
        pl.xlabel('South [m]')
        pl.ylabel('West [m]')
        pl.title('2D Map')
        pl.xlim(-s*init_area, s*init_area)
        pl.ylim(-s*init_area, s*init_area)
        pl.grid()
        pl.pause(0.001)
        pl.draw()


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

    Ed_log[it, :] = fc.Ed

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
