import itertools
import sqlite3 as sql
import pandas as pd

# import data
with sql.connect('3leg_data.db') as db: 
    c = db.cursor()
    df = pd.read_sql_query("SELECT * FROM data", db)
    headings = df.columns.values
    data = df.values
del(c,df)

# define a dataframe
df = pd.DataFrame(data = data, columns = headings)

# create a list of all possible pairs of webs from the 3 leg web data 
pairs_list = list(itertools.combinations(range(len(df)),2))

# create data lists
p1_1, p2_1, p3_1 = [], [], []
p1_2, p2_2, p3_2 = [], [], []
q1_1, q2_1, q3_1 = [], [], []
q1_2, q2_2, q3_2 = [], [], []
m1_1, m2_1, m3_1 = [], [], []
m1_2, m2_2, m3_2 = [], [], []
charge_1, charge_2 = [], []
trace_1, trace_2 = [], []
rank_1, rank_2 = [], []
label_1, label_2 = [], []
predictions = []
for i in range(len(pairs_list)):
    
    index1 = pairs_list[i][0]
    index2 = pairs_list[i][1]
    
    label_1.append(index1)
    label_2.append(index2)
    
    p1_1.append(df['p1'][index1])
    p1_2.append(df['p1'][index2])
    p2_1.append(df['p2'][index1])
    p2_2.append(df['p2'][index2])
    p3_1.append(df['p3'][index1])
    p3_2.append(df['p3'][index2])
    q1_1.append(df['q1'][index1])
    q1_2.append(df['q1'][index2])
    q2_1.append(df['q2'][index1])
    q2_2.append(df['q2'][index2])
    q3_1.append(df['q3'][index1])
    q3_2.append(df['q3'][index2])
    m1_1.append(df['m1'][index1])
    m1_2.append(df['m1'][index2])
    m2_1.append(df['m2'][index1])
    m2_2.append(df['m2'][index2])
    m3_1.append(df['m3'][index1])
    m3_2.append(df['m3'][index2])
    
    trace_1.append(df['total_monodromy_trace'][index1])
    trace_2.append(df['total_monodromy_trace'][index2])
    charge_1.append(df['asymptotic_charge'][index1])
    charge_2.append(df['asymptotic_charge'][index2])
    rank_1.append(df['rank'][index1])
    rank_2.append(df['rank'][index2])
    
    if (df['total_monodromy_trace'][index1] == df['total_monodromy_trace'][index2] and 
        df['asymptotic_charge'][index1] == df['asymptotic_charge'][index2] and 
        df['rank'][index1] == df['rank'][index2]):
        predictions.append(1)
    else:
        predictions.append(0)

# define a pandas dataframe to store the results
results_df = pd.DataFrame({'prediction':predictions,
                           
                           'label_1':label_1,
                           'p1_1':p1_1,'p2_1':p2_1,'p3_1':p3_1,
                           'q1_1':q1_1,'q2_1':q2_1,'q3_1':q3_1,
                           'm1_1':m1_1,'m2_1':m2_1,'m3_1':m3_1,
                           'total_monodromy_trace_1':trace_1,
                           'asymptotic_charge_1':charge_1,
                           'rank_1':rank_1,
                           
                           'label_2':label_2,
                           'p1_2':p1_2,'p2_2':p2_2,'p3_2':p3_2,
                           'q1_2':q1_2,'q2_2':q2_2,'q3_2':q3_2,
                           'm1_2':m1_2,'m2_2':m2_2,'m3_2':m3_2,
                           'total_monodromy_trace_2':trace_2,
                           'asymptotic_charge_2':charge_2,
                           'rank_2':rank_2})

# open a connection to a new database and create a new table in that database for the results
conn = sql.connect('naive_results.db')
results_df.to_sql('data', conn)
