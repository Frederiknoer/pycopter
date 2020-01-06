#!/usr/bin/env python
from shapely.geometry import Point, LineString, Polygon
import math
import numpy as np
import matplotlib.pyplot as plt

DEBUG = False

class uwb_agent:
    def __init__(self, ID, pos):
        self.id = ID
        self.pos = pos
        self.incidenceMatrix = np.array([])
        self.M = [self.id] #Modules
        self.N = [] #Neigbours
        self.E = [0] #
        self.P = [] #
        self.pairs = []
        self.errorMatrix = []
        self.des_dist = 8
        self.I = np.array([[1,0],[0,1]])
        self.poslist = []

    def update_pos(self, new_pos):
        self.pos = new_pos

    def get_B(self):
        return self.incidenceMatrix

    def calc_dist(self, p1, p2):
        #p1 = Point(p1[0], p1[1])
        #p2 = Point(p2[0], p2[1])
        dist = math.sqrt( (p2.x - p1.x)**2 + (p2.y - p1.y)**2 )
        return dist

    def get_distance(self, remote_pos):
        p1 = Point(self.pos)
        p2 = Point(remote_pos)
        dist = math.sqrt( (p2.x - p1.x)**2 + (p2.y - p1.y)**2 )
        return dist

    def update_incidenceMatrix(self):
        self.incidenceMatrix = np.array([])
        self.P = []
        rows = len(self.M)
        cols = len(self.pairs)
        self.incidenceMatrix = np.zeros((rows,cols), dtype=int)
        for i, pair in enumerate(self.pairs):
            col = np.zeros(rows, dtype=int)
            m1 = pair[0]
            m2 = pair[1]
            col[m1] = 1
            col[m2] = -1
            self.incidenceMatrix[:,i] = col.T
            self.P.append(pair[2])


    def add_nb_module(self, Id, range):
        if not any(x == Id for x in self.N):
            self.N.append(Id)
            self.E.append(range)
            self.M.append(Id)
            self.pairs.append([self.id, Id, range])
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
            self.pairs.append([Id1, Id2, range])


    def handle_range_msg(self, Id, nb_pos):
        range = self.get_distance(nb_pos)
        self.add_nb_module(Id, range)
        self.update_incidenceMatrix()

    def handle_other_msg(self, Id1, Id2, range):
        self.add_pair(Id1, Id2, range)
        self.update_incidenceMatrix()

    def define_triangle(self):
        a,b,c = self.P[0:3]
        angle_a = math.acos( (b**2 + c**2 - a**2) / (2 * b * c) )
        angle_b = math.acos( (a**2 + c**2 - b**2) / (2 * a * c) )
        angle_c = math.acos( (a**2 + b**2 - c**2) / (2 * a * b) )

        A = Point(0.0, 0.0)
        B = Point(c, 0.0)
        C = Point(b*math.cos(angle_a), b*math.sin(angle_a))

        self.poslist = A,B,C
        return A, B, C #, angle_a, angle_b, angle_c

    def calcErrorMatrix(self):
        arrSize = len(self.M)

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
        arrSize = len(self.M)
        self.calcErrorMatrix()
        #print(self.errorMatrix)

        U = []
        K = 0.0002
        u = 0
        E = self.errorMatrix

        for i in range(arrSize):
            u_x = 0
            u_y = 0
            for k in range(arrSize):
                if k != i:
                    if DEBUG:
                        print("i: ",i,"  k: ",k)
                        print(self.poslist[i])
                        print(self.poslist[k])

                    x_dif = self.poslist[i].x - self.poslist[k].x
                    y_dif = self.poslist[i].y - self.poslist[k].y
                    xy_mag = math.sqrt( (x_dif)**2 + (y_dif)**2 )

                    u_x += K * E[i][k] * (x_dif/xy_mag)
                    u_y += K * E[i][k] * (y_dif/xy_mag)

            U.append(u_x)
            U.append(u_y)

        if DEBUG:
            print("U: ")
            print(U)

        print ("A:", self.poslist[0].x,self.poslist[0].y, "  B: ", self.poslist[1].x,self.poslist[1].y, "  C: ", self.poslist[2].x,self.poslist[2].y)
        print ("Error Matrix: ")
        print (self.errorMatrix)
        print ("U: ")
        print (U)
        return U


    def run():
        pass
