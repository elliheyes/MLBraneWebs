import csv
import numpy as np
import pandas as pd
import sqlite3 as sql
from itertools import product
from data_functions import gcd, anticlockwise_sort, monodromy

# generate sets of web variables p,q and m 
var_lists = []
for p1, p2, p3, q1, q2, q3 in product(range(-3,4),repeat=6):
    for m1, m2, m3 in product(range(1,4),repeat=3):
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
    # define a list for the 7-brane p,q charges and 5-brane multiplicities m
    var_list = var_lists[i]
    
    # define the external leg p,q charges
    P1 = var_list[0]*var_list[6]
    P2 = var_list[1]*var_list[7]
    P3 = var_list[2]*var_list[8]
    Q1 = var_list[3]*var_list[6]
    Q2 = var_list[4]*var_list[7]
    Q3 = var_list[5]*var_list[8]
    
    # compute the SL2Z invariant quantity I
    term1 = abs(P1*Q2-P2*Q1 + P1*Q3-P3*Q1 + P2*Q3-P3*Q2)
    term2 = gcd(P1,Q1)**2 + gcd(P2,Q2)**2 + gcd(P3,Q3)**2
    I = term1 - term2
    
    # only record webs which satisfy SUSY, that is I>=-2
    if I >= -2:
        
        # record the variabels
        p1_list.append(var_list[0])
        p2_list.append(var_list[1])
        p3_list.append(var_list[2])
        q1_list.append(var_list[3])
        q2_list.append(var_list[4])
        q3_list.append(var_list[5])
        m1_list.append(var_list[6])
        m2_list.append(var_list[7])
        m3_list.append(var_list[8])
    
        # record the web rank
        rank_list.append((I+2)/2)
        
        # order anticlockwise
        var_list = anticlockwise_sort(var_list)
    
        # compute the inidividual monodromies
        M1 = monodromy(var_list[0],var_list[3])
        M2 = monodromy(var_list[1],var_list[4])
        M3 = monodromy(var_list[2],var_list[5])
    
        # compute the total monodromy
        M = np.matmul(M2,M1)
        Mtot = np.matmul(M3,M)
        
        # record the total monodromy trace
        trace_list.append(np.matrix.trace(Mtot))
    
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

# create lists of equivalent webs
equiv_idx = []
for i in range(len(df)):
    idx_list = []
    for j in range(len(df)):
        # check if the asymptotic charge, total monodromy trace and ranks are equal
        if (df['asymptotic_charge'][i] == df['asymptotic_charge'][j] and 
            df['total_monodromy_trace'][i] == df['total_monodromy_trace'][j] and
            df['rank'][i] == df['rank'][j]):
            idx_list.append(j)
    equiv_idx.append(idx_list)
    
equiv_groups = []
for i in equiv_idx:
    if i not in equiv_groups:
        equiv_groups.append(i)
        
# create lists of web matrices and labels from the 10 classes
P1, P2, P3 = [], [], []
Q1, Q2, Q3 = [], [], []
labels = []
count1, count2, count3, count4, count5 = 0, 0, 0, 0, 0
count6, count7, count8, count9, count10 = 0, 0, 0, 0, 0
count11, count12, count13, count14 = 0, 0, 0, 0
non_equiv_webs = []
for i in range(len(df)):
    if i in equiv_groups[0] and count1 < 48:
        P1.append(df['p1'][i]*df['m1'][i])
        P2.append(df['p2'][i]*df['m2'][i])
        P3.append(df['p3'][i]*df['m3'][i])
        Q1.append(df['q1'][i]*df['m1'][i])
        Q2.append(df['q2'][i]*df['m2'][i])
        Q3.append(df['q3'][i]*df['m3'][i])
        labels.append(0)
        if count1 == 0:
            non_equiv_webs.append([df['p1'][i],df['p2'][i],df['p3'][i],
                           df['q1'][i],df['q2'][i],df['q3'][i],
                           df['m1'][i],df['m2'][i],df['m3'][i]])
        count1 += 1
    elif i in equiv_groups[1] and count2 < 48:
        P1.append(df['p1'][i]*df['m1'][i])
        P2.append(df['p2'][i]*df['m2'][i])
        P3.append(df['p3'][i]*df['m3'][i])
        Q1.append(df['q1'][i]*df['m1'][i])
        Q2.append(df['q2'][i]*df['m2'][i])
        Q3.append(df['q3'][i]*df['m3'][i])
        labels.append(1)
        if count2 == 0:
            non_equiv_webs.append([df['p1'][i],df['p2'][i],df['p3'][i],
                           df['q1'][i],df['q2'][i],df['q3'][i],
                           df['m1'][i],df['m2'][i],df['m3'][i]])
        count2 += 1
    elif i in equiv_groups[2] and count3 < 48:
        P1.append(df['p1'][i]*df['m1'][i])
        P2.append(df['p2'][i]*df['m2'][i])
        P3.append(df['p3'][i]*df['m3'][i])
        Q1.append(df['q1'][i]*df['m1'][i])
        Q2.append(df['q2'][i]*df['m2'][i])
        Q3.append(df['q3'][i]*df['m3'][i])
        labels.append(2)
        if count3 == 0:
            non_equiv_webs.append([df['p1'][i],df['p2'][i],df['p3'][i],
                           df['q1'][i],df['q2'][i],df['q3'][i],
                           df['m1'][i],df['m2'][i],df['m3'][i]])
        count3 += 1
    elif i in equiv_groups[3] and count4 < 48:
        P1.append(df['p1'][i]*df['m1'][i])
        P2.append(df['p2'][i]*df['m2'][i])
        P3.append(df['p3'][i]*df['m3'][i])
        Q1.append(df['q1'][i]*df['m1'][i])
        Q2.append(df['q2'][i]*df['m2'][i])
        Q3.append(df['q3'][i]*df['m3'][i])
        labels.append(3)
        if count4 == 0:
            non_equiv_webs.append([df['p1'][i],df['p2'][i],df['p3'][i],
                           df['q1'][i],df['q2'][i],df['q3'][i],
                           df['m1'][i],df['m2'][i],df['m3'][i]])
        count4 += 1
    elif i in equiv_groups[4] and count5 < 48:
        P1.append(df['p1'][i]*df['m1'][i])
        P2.append(df['p2'][i]*df['m2'][i])
        P3.append(df['p3'][i]*df['m3'][i])
        Q1.append(df['q1'][i]*df['m1'][i])
        Q2.append(df['q2'][i]*df['m2'][i])
        Q3.append(df['q3'][i]*df['m3'][i])
        labels.append(4)
        if count5 == 0:
            non_equiv_webs.append([df['p1'][i],df['p2'][i],df['p3'][i],
                           df['q1'][i],df['q2'][i],df['q3'][i],
                           df['m1'][i],df['m2'][i],df['m3'][i]])
        count5 += 1
    elif i in equiv_groups[5] and count6 < 48:
        P1.append(df['p1'][i]*df['m1'][i])
        P2.append(df['p2'][i]*df['m2'][i])
        P3.append(df['p3'][i]*df['m3'][i])
        Q1.append(df['q1'][i]*df['m1'][i])
        Q2.append(df['q2'][i]*df['m2'][i])
        Q3.append(df['q3'][i]*df['m3'][i])
        labels.append(5)
        if count6 == 0:
            non_equiv_webs.append([df['p1'][i],df['p2'][i],df['p3'][i],
                           df['q1'][i],df['q2'][i],df['q3'][i],
                           df['m1'][i],df['m2'][i],df['m3'][i]])
        count6 += 1
    elif i in equiv_groups[6] and count7 < 48:
        P1.append(df['p1'][i]*df['m1'][i])
        P2.append(df['p2'][i]*df['m2'][i])
        P3.append(df['p3'][i]*df['m3'][i])
        Q1.append(df['q1'][i]*df['m1'][i])
        Q2.append(df['q2'][i]*df['m2'][i])
        Q3.append(df['q3'][i]*df['m3'][i])
        labels.append(6)
        if count7 == 0:
            non_equiv_webs.append([df['p1'][i],df['p2'][i],df['p3'][i],
                           df['q1'][i],df['q2'][i],df['q3'][i],
                           df['m1'][i],df['m2'][i],df['m3'][i]])
        count7 += 1
    elif i in equiv_groups[7] and count8 < 48:
        P1.append(df['p1'][i]*df['m1'][i])
        P2.append(df['p2'][i]*df['m2'][i])
        P3.append(df['p3'][i]*df['m3'][i])
        Q1.append(df['q1'][i]*df['m1'][i])
        Q2.append(df['q2'][i]*df['m2'][i])
        Q3.append(df['q3'][i]*df['m3'][i])
        labels.append(7)
        if count8 == 0:
            non_equiv_webs.append([df['p1'][i],df['p2'][i],df['p3'][i],
                           df['q1'][i],df['q2'][i],df['q3'][i],
                           df['m1'][i],df['m2'][i],df['m3'][i]])
        count8 += 1
    elif i in equiv_groups[8] and count9 < 48:
        P1.append(df['p1'][i]*df['m1'][i])
        P2.append(df['p2'][i]*df['m2'][i])
        P3.append(df['p3'][i]*df['m3'][i])
        Q1.append(df['q1'][i]*df['m1'][i])
        Q2.append(df['q2'][i]*df['m2'][i])
        Q3.append(df['q3'][i]*df['m3'][i])
        labels.append(8)
        if count9 == 0:
            non_equiv_webs.append([df['p1'][i],df['p2'][i],df['p3'][i],
                           df['q1'][i],df['q2'][i],df['q3'][i],
                           df['m1'][i],df['m2'][i],df['m3'][i]])
        count9 += 1
    elif i in equiv_groups[9] and count10 < 48:
        P1.append(df['p1'][i]*df['m1'][i])
        P2.append(df['p2'][i]*df['m2'][i])
        P3.append(df['p3'][i]*df['m3'][i])
        Q1.append(df['q1'][i]*df['m1'][i])
        Q2.append(df['q2'][i]*df['m2'][i])
        Q3.append(df['q3'][i]*df['m3'][i])
        labels.append(9)
        if count10 == 0:
            non_equiv_webs.append([df['p1'][i],df['p2'][i],df['p3'][i],
                           df['q1'][i],df['q2'][i],df['q3'][i],
                           df['m1'][i],df['m2'][i],df['m3'][i]])
        count10 += 1
    elif i in equiv_groups[10] and count11 < 48:
        P1.append(df['p1'][i]*df['m1'][i])
        P2.append(df['p2'][i]*df['m2'][i])
        P3.append(df['p3'][i]*df['m3'][i])
        Q1.append(df['q1'][i]*df['m1'][i])
        Q2.append(df['q2'][i]*df['m2'][i])
        Q3.append(df['q3'][i]*df['m3'][i])
        labels.append(10)
        if count11 == 0:
            non_equiv_webs.append([df['p1'][i],df['p2'][i],df['p3'][i],
                           df['q1'][i],df['q2'][i],df['q3'][i],
                           df['m1'][i],df['m2'][i],df['m3'][i]])
        count11 += 1
    elif i in equiv_groups[11] and count12 < 48:
        P1.append(df['p1'][i]*df['m1'][i])
        P2.append(df['p2'][i]*df['m2'][i])
        P3.append(df['p3'][i]*df['m3'][i])
        Q1.append(df['q1'][i]*df['m1'][i])
        Q2.append(df['q2'][i]*df['m2'][i])
        Q3.append(df['q3'][i]*df['m3'][i])
        labels.append(11)
        if count12 == 0:
            non_equiv_webs.append([df['p1'][i],df['p2'][i],df['p3'][i],
                           df['q1'][i],df['q2'][i],df['q3'][i],
                           df['m1'][i],df['m2'][i],df['m3'][i]])
        count12 += 1
    elif i in equiv_groups[12] and count13 < 48:
        P1.append(df['p1'][i]*df['m1'][i])
        P2.append(df['p2'][i]*df['m2'][i])
        P3.append(df['p3'][i]*df['m3'][i])
        Q1.append(df['q1'][i]*df['m1'][i])
        Q2.append(df['q2'][i]*df['m2'][i])
        Q3.append(df['q3'][i]*df['m3'][i])
        labels.append(12)
        if count13 == 0:
            non_equiv_webs.append([df['p1'][i],df['p2'][i],df['p3'][i],
                           df['q1'][i],df['q2'][i],df['q3'][i],
                           df['m1'][i],df['m2'][i],df['m3'][i]])
        count13 += 1
    elif i in equiv_groups[13] and count14 < 48:
        P1.append(df['p1'][i]*df['m1'][i])
        P2.append(df['p2'][i]*df['m2'][i])
        P3.append(df['p3'][i]*df['m3'][i])
        Q1.append(df['q1'][i]*df['m1'][i])
        Q2.append(df['q2'][i]*df['m2'][i])
        Q3.append(df['q3'][i]*df['m3'][i])
        labels.append(13)
        if count14 == 0:
            non_equiv_webs.append([df['p1'][i],df['p2'][i],df['p3'][i],
                           df['q1'][i],df['q2'][i],df['q3'][i],
                           df['m1'][i],df['m2'][i],df['m3'][i]])
        count14 += 1
    
# create a datafrane to store the results
df2 = pd.DataFrame({'P1':P1,'P2':P2,'P3':P3,'Q1':Q1,'Q2':Q2,'Q3':Q3,'label':labels})

# open a connection to a new database and create a new table in that database for the 3 leg web data
conn = sql.connect('3leg_data_X.db')
df2.to_sql('data', conn)

# save non-equivalent webs to a file
writefile = open('3leg_data_Y_I.csv', 'w')
writer = csv.writer(writefile)
writer.writerows(non_equiv_webs)
writefile.close()
