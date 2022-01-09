import math
import numpy as np
from keras import backend as K

def gcd(x,y):
    """Compute the greatest common divisor between two integers."""
    while(y):
        x, y = y, x % y
    return x

def monodromy(p,q):
    """Compute the anticlockwise monodromy matrix of a (p,q) 7-brane."""
    return np.array([[1+p*q,-p**2],[q**2,1-p*q]])

def anticlockwise_angle(x1,x2):
    """Compute the anticlockwise angle between x1 and x2."""
    x = np.array([x2[0]-x1[0]])
    y = np.array([x2[1]-x1[1]])
    
    if x2[1] == x1[1]:
        if x2[0] == x1[0]:
            return np.array([0])
        elif x2[0] > x1[0]:
            return np.array([360])
        else:
            return np.array([180])
        
    else:
        if x2[1] > x1[1]:
            return np.arctan2(y,x)*180/np.pi
        else:
            return 360 + np.arctan2(y,x)*180/np.pi

def anticlockwise_sort(var_list):
    """Order a set of (p,q) 7-branes anticlockwise."""
    angle1 = anticlockwise_angle((0,0),(var_list[0],var_list[3]))[0]
    angle2 = anticlockwise_angle((0,0),(var_list[1],var_list[4]))[0]
    angle3 = anticlockwise_angle((0,0),(var_list[2],var_list[5]))[0]
    
    length1 = math.sqrt(var_list[0]**2 + var_list[3]**2)
    length2 = math.sqrt(var_list[1]**2 + var_list[4]**2)
    length3 = math.sqrt(var_list[2]**2 + var_list[5]**2)

    angle_length_list = [(angle1,length1),(angle2,length2),(angle3,length3)]
    sorted_angle_length_list = sorted(angle_length_list)
    
    indices = []
    for j in range(len(sorted_angle_length_list)):
        indices.append(angle_length_list.index(sorted_angle_length_list[j]))

    p_list = [var_list[:3][j] for j in indices]
    q_list = [var_list[3:6][j] for j in indices]
    
    return p_list + q_list
