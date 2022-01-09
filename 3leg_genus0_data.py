import csv
from itertools import product
from data_functions import gcd

# generate sets of web variables p,q and m 
var_list = []
for p, q, l, r in product(range(-10,11), repeat=4):
        # only record non trivial webs
        if not(p == l and q == r):
            # check coprimeness of 7-brane p,q charges
            if (gcd(p,q) == 1 and gcd(l,r) == 1):
                var_list.append([p,q,l,r])    

# export the results to files
webs_file = open('3leg_genus0_data.csv', 'w')
datawriter = csv.writer(webs_file)
datawriter.writerows(var_list)
webs_file.close()


