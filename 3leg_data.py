import numpy as np
import pandas as pd
from itertools import product
import sqlite3 as sql
import csv
from data_functions import gcd, anticlockwise_sort, monodromy

# generate sets of web variables p,q and m 
var_lists = []
for p1, p2, p3, q1, q2, q3 in product(range(-10,11), repeat=6):
    for m1, m2, m3 in product(range(1,11), repeat=3):
        # only record non trivial webs
        if not(m1 == 0 and m2 == 0 and m3 == 0):
            # check p charge conservation
            if p1*m1 + p2*m2 + p3*m3 == 0:
                # check q charge conservation
                if q1*m1 + q2*m2 + q3*m3 == 0:
                    # check coprimeness of 7-brane p,q charges
                    if (abs(gcd(p1,q1)) == 1 and abs(gcd(p2,q2)) == 1 and abs(gcd(p3,q3)) == 1):
                        var_lists.append([p1,p2,p3,q1,q2,q3,m1,m2,m3])    

# generate the list of variabels, web matrices, total monodromy traces, asymptotic charges and ranks
rank_list = []
trace_list = []
charge_list = []
p1_list, p2_list, p3_list = [], [], []
q1_list, q2_list, q3_list = [], [], []
m1_list, m2_list, m3_list = [], [], []
for i in range(len(var_lists)):
    
    # define the external leg p,q charges
    P1 = var_lists[i][0]*var_lists[i][6]
    P2 = var_lists[i][1]*var_lists[i][7]
    P3 = var_lists[i][2]*var_lists[i][8]
    Q1 = var_lists[i][3]*var_lists[i][6]
    Q2 = var_lists[i][4]*var_lists[i][7]
    Q3 = var_lists[i][5]*var_lists[i][8]
    
    # compute the SL2Z invariant quantity I
    term1 = abs(P1*Q2-P2*Q1 + P1*Q3-P3*Q1 + P2*Q3-P3*Q2)
    term2 = gcd(P1,Q1)**2 + gcd(P2,Q2)**2 + gcd(P3,Q3)**2
    I = term1 - term2
    
    # only record webs which satisfy SUSY, that is I>=-2
    if I >= -2:
        
        # record the variabels
        p1_list.append(var_lists[i][0])
        p2_list.append(var_lists[i][1])
        p3_list.append(var_lists[i][2])
        q1_list.append(var_lists[i][3])
        q2_list.append(var_lists[i][4])
        q3_list.append(var_lists[i][5])
        m1_list.append(var_lists[i][6])
        m2_list.append(var_lists[i][7])
        m3_list.append(var_lists[i][8])
    
        # record the web rank
        rank_list.append((I+2)/2)
        
        # define a list for the 7-brane p,q charges ordered anticlockwise
        var_list = anticlockwise_sort(var_lists[i][:6])
    
        # compute the inidividual monodromies
        M1 = monodromy(var_list[0],var_list[3])
        M2 = monodromy(var_list[1],var_list[4])
        M3 = monodromy(var_list[2],var_list[5])
    
        # compute the total monodromy
        M = np.matmul(M2,M1)
        M = np.matmul(M3,M)
        
        # record the total monodromy trace
        trace_list.append(np.matrix.trace(M))
    
        # compute the asymptotic charge 
        c1 = var_list[0]*var_list[4] - var_list[1]*var_list[3]
        c2 = var_list[0]*var_list[5] - var_list[2]*var_list[3]
        c3 = var_list[1]*var_list[5] - var_list[2]*var_list[4]
        gcd1 = gcd(c1,c2)
        gcd2 = gcd(c3,gcd1)
        
        # record the asymptotic charge
        charge_list.append(abs(gcd2))

# define a pandas dataframe to store the data
df = pd.DataFrame({'p1':p1_list,'p2':p2_list,'p3':p3_list,
                   'q1':q1_list,'q2':q2_list,'q3':q3_list,
                   'm1':m1_list,'m2':m2_list,'m3':m3_list,
                   'total_monodromy_trace':trace_list,'asymptotic_charge':charge_list,
                   'rank':rank_list})

# open a connection to a new database and create a new table in that database for the 3 leg web data
conn = sql.connect('3leg_data.db')
df.to_sql('data', conn)

# create lists of equivalent webs
equiv_idx = []
for i in range(len(rank_list)):
    idx_list = []
    for j in range(len(rank_list)):
        if (rank_list[i] == rank_list[j] and 
            charge_list[i] == charge_list[j] and 
            trace_list[i] == trace_list[j]):
            idx_list.append(j)
    equiv_idx.append(idx_list)

equiv_groups = []
for i in equiv_idx:
    if i not in equiv_groups:
        equiv_groups.append(i)
   
# create a list of inequivalent web matrices
web_list = []
for i in range(len(equiv_groups)):
    web_list.append([[p1_list[equiv_groups[i][0]]*m1_list[equiv_groups[i][0]],
                 p2_list[equiv_groups[i][0]]*m2_list[equiv_groups[i][0]],
                 p3_list[equiv_groups[i][0]]*m3_list[equiv_groups[i][0]]],
                [q1_list[equiv_groups[i][0]]*m1_list[equiv_groups[i][0]],
                 q2_list[equiv_groups[i][0]]*m2_list[equiv_groups[i][0]],
                 q3_list[equiv_groups[i][0]]*m3_list[equiv_groups[i][0]]]])
    
# export the results to files
webs_file = open('3leg_data.csv', 'w')
datawriter = csv.writer(webs_file)
datawriter.writerows(web_list)
webs_file.close()
