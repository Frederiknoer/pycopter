#!/usr/bin/env python
import math
import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

DEBUG = False

class KalmanFilterPos:
    def __init__(self):
        self.x_pos = 0
        self.y_pos = 0
        self.speed = 0.0
        self.dt = 5e-2
        r=12

        self.KF = KalmanFilter(dim_x=4, dim_z=4)
        #State matrix:
        self.KF.x = np.array([  [0.0], #Pos_x
                                [0.0], #Vel_x
                                [0.0], #Pos_y
                                [0.0]])#Vel_y
        #Covariance Matrix:
        self.KF.P = np.array([  [100.0, 0.0, 0.0, 0.0],
                                [0.0, 100.0, 0.0, 0.0],
                                [0.0, 0.0, 100.0, 0.0],
                                [0.0, 0.0, 0.0, 100.0]])

        #Proces- and Measurements Noise
        self.KF.Q = Q_discrete_white_noise(dim=4, dt=self.dt, var=r**2)
        self.KF.R = np.array([  [r, 0.0, 0.0, 0.0],
                                [0.0, r, 0.0, 0.0],
                                [0.0, 0.0, r, 0.0],
                                [0.0, 0.0, 0.0, r]])

        #Measurement function
        self.KF.H = np.array([  [1.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0]])

        self.KF.F = np.array(([ [1.0, self.dt, 0.0, 0.0],
                                [0.0,   1.0,  0.0,  0.0],
                                [0.0, 0.0, 1.0, self.dt],
                                [0.0,   0.0,  0.0,  1.0]]))

    def Predict(self):
        self.KF.predict()
        self.x_pos = self.KF.x[0]
        self.y_pos = self.KF.x[2]

    def updateValues(self, new_x_pos, new_y_pos, x_vel, y_vel):
        self.KF.update([[new_x_pos], [x_vel], [new_y_pos], [y_vel]])
        #self.speed = math.sqrt(math.pow(float(self.KF.x[1]),2)
        #+ math.pow(float(self.KF.x[3]),2))


class uwb_agent:
    def __init__(self, ID):
        self.id = ID
        self.incidenceMatrix = np.array([])
        self.M = np.array([self.id]) #Modules
        self.N = np.array([]) #Neigbours
        self.E = np.array([0]) #
        self.P = np.array([]) #
        self.pairs = np.empty((0,3))
        self.errorMatrix = np.array([])
        self.des_dist = 15
        self.I = np.array([[1,0],[0,1]])
        self.poslist = np.array([])

        #Kalman stuff
        self.usingKalman = True
        self.kf_A = KalmanFilterPos()
        self.kf_B = KalmanFilterPos()
        self.kf_C = KalmanFilterPos()

    def get_B(self):
        return self.incidenceMatrix

    def calc_dist(self, p1, p2):
        return np.linalg.norm(p1 - p2)

    def update_incidenceMatrix(self):
        self.incidenceMatrix = np.array([])
        self.P = np.array([])
        self.incidenceMatrix = np.zeros((3,3), dtype=int)
        for i, pair in enumerate(self.pairs):
            col = np.zeros(3, dtype=int)
            m1 = int(pair[0])
            m2 = int(pair[1])
            col[m1] = 1
            col[m2] = -1
            self.incidenceMatrix[:,i] = col.T
            self.P = np.append(self.P, pair[2])


    def add_nb_module(self, Id, range):
        if not any(x == Id for x in self.N):
            self.N = np.append(self.N, Id)
            self.E = np.append(self.E, range)
            self.M = np.append(self.M, Id)
            self.pairs = np.append(self.pairs, np.array([[self.id, Id, range]]),axis=0)
        else:
            self.E[Id] = range
            for pair in self.pairs:
                if any(x == Id for x in pair) and any(x == self.id for x in pair):
                    pair[2] = range

    def add_pair(self, Id1, Id2, range):
        pair_present = False
        for i, pair in enumerate(self.pairs):
            if (pair[0] == Id1 and pair[1] == Id2) or (pair[1] == Id1 and pair[0] == Id2):
                self.pairs[i][2] = range
                pair_present = True

        if not pair_present:
            self.pairs = np.append(self.pairs, np.array([[Id1, Id2, range]]),axis=0)


    def handle_range_msg(self, Id, range):
        self.add_nb_module(Id, range)
        self.update_incidenceMatrix()

    def handle_other_msg(self, Id1, Id2, range):
        self.add_pair(Id1, Id2, range)
        self.update_incidenceMatrix()

    def clean_cos(self, cos_angle):
        return min(1,max(cos_angle,-1))

    def define_triangle(self):
        c,b,a = self.P[0:3]

        angle_a = math.acos(self.clean_cos( (b**2 + c**2 - a**2) / (2 * b * c) ))
        angle_b = math.acos(self.clean_cos( (a**2 + c**2 - b**2) / (2 * a * c) ))
        angle_c = math.acos(self.clean_cos( (a**2 + b**2 - c**2) / (2 * a * b) ))

        A = np.array([0.0, 0.0])
        B = np.array([c, 0.0])
        if self.id == 0:
            C = np.array([b*math.cos(angle_a), b*math.sin(angle_a)])
        elif self.id == 1:
            C = np.array([b*math.cos(angle_a), b*math.sin(angle_a)])
        elif self.id == 2:
            C = np.array([b*math.cos(angle_a), b*math.sin(angle_a)])

        self.poslist = A,B,C
        return A, B, C #, angle_a, angle_b, angle_c

    def kalman_triangle_update(self, vel_A, vel_B, vel_C):
        self.kf_A.updateValues(self.poslist[0][0], self.poslist[0][1], vel_A[0], -vel_A[1])
        self.kf_B.updateValues(self.poslist[1][0], self.poslist[1][1], vel_B[0], -vel_B[1])
        self.kf_C.updateValues(self.poslist[2][0], self.poslist[2][1], vel_C[0], -vel_C[1])

    def kalman_triangle_predict(self):
        self.kf_A.Predict()
        self.kf_B.Predict()
        self.kf_C.Predict()
        A = np.array([self.kf_A.x_pos, self.kf_A.y_pos])
        B = np.array([self.kf_B.x_pos, self.kf_B.y_pos])
        C = np.array([self.kf_C.x_pos, self.kf_C.y_pos])
        self.poslist = A,B,C
        return A, B, C

    def calcErrorMatrix(self,v1,v2,v3):
        arrSize = self.M.size

        self.errorMatrix = np.zeros((arrSize, arrSize))

        if self.usingKalman:
            self.kalman_triangle_update(v1, v2, v3)
            self.kalman_triangle_predict()


        poslist = self.poslist
        print("ID: ",self.id,"  Poslist: ",poslist)

        for i in range(arrSize):
            for j in range(arrSize):
                curDis = self.calc_dist(poslist[i], poslist[j])
                if curDis == 0:
                    self.errorMatrix[i][j] = 0.0
                else:
                    self.errorMatrix[i][j] = curDis - self.des_dist
        print("Error Matrix: ")
        print(self.errorMatrix)

    def calc_u_acc(self,v1,v2,v3):
        self.calcErrorMatrix(v1,v2,v3)

        U = np.array([])
        K = 0.01
        E = self.errorMatrix

        for i in range(self.M.size):
            u_x = 0
            u_y = 0
            for k in range(self.M.size):
                if k != i:
                    x_dif = self.poslist[i][0] - self.poslist[k][0]
                    y_dif = self.poslist[i][1] - self.poslist[k][1]
                    xy_mag = self.calc_dist(self.poslist[i],self.poslist[k])
                    unitvec = (self.poslist[i] - self.poslist[k]) / xy_mag

                    u_x += K * E[i][k] * unitvec[0]
                    u_y += K * E[i][k] * unitvec[1]

            U = np.append(U, [u_x, u_y])
        print("U: ")
        print(U)
        return U


    def run():
        pass
