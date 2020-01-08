#!/usr/bin/env python
from shapely.geometry import Point, LineString, Polygon
import math
import numpy as np
import matplotlib.pyplot as plt

DEBUG = False

class uwb_agent:
    def __init__(self, ID):
        self.id = ID
        #self.pos = pos
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

    def update_pos(self, new_pos):
        self.pos = new_pos

    def get_B(self):
        return self.incidenceMatrix

    def calc_dist(self, p1, p2):
        dist = np.linalg.norm(p1 - p2)
        return dist

    def get_distance(self, remote_pos):
        return self.calc_dist(self.pos, remote_pos)

    def update_incidenceMatrix(self):
        self.incidenceMatrix = np.array([])
        self.P = np.array([])
        rows = len(self.M)
        cols = self.pairs.shape[0]
        self.incidenceMatrix = np.zeros((rows,cols), dtype=int)
        for i, pair in enumerate(self.pairs):
            col = np.zeros(rows, dtype=int)
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
        #range = self.get_distance(nb_pos)
        self.add_nb_module(Id, range)
        self.update_incidenceMatrix()

    def handle_other_msg(self, Id1, Id2, range):
        self.add_pair(Id1, Id2, range)
        self.update_incidenceMatrix()

    def define_triangle(self):
        c,b,a = self.P[0:3]
        print("a:",a," b:",b," c:",c)
        angle_a = math.acos( (b**2 + c**2 - a**2) / (2 * b * c) )
        angle_b = math.acos( (a**2 + c**2 - b**2) / (2 * a * c) )
        angle_c = math.acos( (a**2 + b**2 - c**2) / (2 * a * b) )

        A = np.array([0.0, 0.0])
        B = np.array([c, 0.0])
        C = np.array([b*math.cos(angle_a), b*math.sin(angle_a)])

        self.poslist = A,B,C
        return A, B, C #, angle_a, angle_b, angle_c

    def calcErrorMatrix(self):
        arrSize = self.M.size

        self.errorMatrix = np.zeros((arrSize, arrSize))
        poslist = self.poslist

        for i in range(arrSize):
            for j in range(arrSize):
                curDis = self.calc_dist(poslist[i], poslist[j])
                if curDis == 0:
                    self.errorMatrix[i][j] = 0.0
                else:
                    self.errorMatrix[i][j] = curDis - self.des_dist

    def calc_u_acc(self):
        self.calcErrorMatrix()

        U = np.array([])
        K = 0.02
        u = 0
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
