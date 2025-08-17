#Code to visualize the different probability vectors on the lattice because hard otherwise.

import matplotlib.pyplot as plt
import majorization as mj

def plot_lattice(p, q, *kwargs, cone_angle=30): #2D rep of the cone on the lattice
    
    join = p + q
    # height of the points
    y_p = mj.entropy(p)
    y_q = mj.entropy(q)
    y_join = mj.entropy(join)
    
    dist_q = y_q - y_join
    dist_p = y_p - y_join
    
    