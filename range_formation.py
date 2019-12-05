import numpy as np

class range_formation:
    def __init__(self, m, l, d, mu, tilde_mu, B, c_shape, c_vel):
        self.m = m # Dimension {2,3}
        self.l = l # Order of the Lyapunov function (1,2,...)
        self.d = d # Set of desired distances (linked to the set of edges)

        self.B = B # Incidence matrix
        self.agents, self.edges = self.B.shape


    def get_u_acc(self):
        pass
