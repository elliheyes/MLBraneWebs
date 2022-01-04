import csv
import networkx as nx
import sqlite3 as sql
import pandas as pd
from collections import Counter
from networkx.algorithms.community import greedy_modularity_communities

# choose method: 'naive' or 'NN'
method = 'naive'

# import data
with sql.connect(method+'_results.db') as db: 
    c = db.cursor()
    df = pd.read_sql_query("SELECT * FROM data", db)
    headings = df.columns.values
    data = df.values
del(c,df)

# define a dataframe
df = pd.DataFrame(data = data, columns = headings)

# define the list of web labels
labels = list(range(int(df.iloc[-1]['label_2']+1)))

# construct the lists web matrices, total monodromy traces, asymptotic charges and ranks
web = [df['p1_1'][0]*df['m1_1'][0]]+[df['p2_1'][0]*df['m2_1'][0]]+[df['p3_1'][0]*df['m3_1'][0]]+[df['q1_1'][0]*df['m1_1'][0]]+[df['q2_1'][0]*df['m2_1'][0]]+[df['q3_1'][0]*df['m3_1'][0]]
webs = [web]

ranks = [df['rank_1'][0]]
traces = [df['total_monodromy_trace_1'][0]]
charges = [df['asymptotic_charge_1'][0]]

index = 0

for i in range(1,len(labels)-1):
    index = index + len(labels) - i
    
    web = [df['p1_1'][index]*df['m1_1'][index]]+[df['p2_1'][index]*df['m2_1'][index]]+[df['p3_1'][index]*df['m3_1'][index]]+[df['q1_1'][index]*df['m1_1'][index]]+[df['q2_1'][index]*df['m2_1'][index]]+[df['q3_1'][index]*df['m3_1'][index]]
    webs.append(web)

    ranks.append(df['rank_1'][index])
    traces.append(df['total_monodromy_trace_1'][index])
    charges.append(df['asymptotic_charge_1'][index])
    
web = [df.iloc[-1]['p1_2']*df.iloc[-1]['m1_2']]+[df.iloc[-1]['p2_2']*df.iloc[-1]['m2_2']]+[df.iloc[-1]['p3_2']*df.iloc[-1]['m3_2']]+[df.iloc[-1]['q1_2']*df.iloc[-1]['m1_2']]+[df.iloc[-1]['q2_2']*df.iloc[-1]['m2_2']]+[df.iloc[-1]['q3_2']*df.iloc[-1]['m3_2']]
webs.append(web)

ranks.append(df.iloc[-1]['rank_2'])
traces.append(df.iloc[-1]['total_monodromy_trace_2'])
charges.append(df.iloc[-1]['asymptotic_charge_2'])

# create the graph with webs as nodes and edges if the two webs are predicted equivalent
G = nx.Graph()
G.add_nodes_from(labels)
for i in range(len(df)):
    if df.iloc[i]['prediction'] == 1:
        G.add_edge(df.iloc[i]['label_1'],df.iloc[i]['label_2'])
        
# group the webs by greedy modularity maximisation
c = list(greedy_modularity_communities(G))
clustering = []
for i in range(len(c)):
    clustering.append(sorted(c[i]))
    
# get the web matrices, ranks, total monodromy traces and asymptotic charges of the webs in each cluster
rank_list = [[ranks[clustering[i][j]] for j in range(len(clustering[i]))] for i in range(len(clustering))]
web_list = [[webs[clustering[i][j]] for j in range(len(clustering[i]))] for i in range(len(clustering))]
trace_list = [[traces[clustering[i][j]] for j in range(len(clustering[i]))] for i in range(len(clustering))]
charge_list = [[charges[clustering[i][j]] for j in range(len(clustering[i]))] for i in range(len(clustering))]

# create a counter for the cluster ranks
top_rank_list = []
for i in range(len(rank_list)):
    c = Counter(rank_list[i])
    top_rank_list.append(c.most_common()[0][0])
rank_counter = Counter(top_rank_list)

# export the results
writefile = open(method+'_grouped_ranks.csv', 'w')
writer = csv.writer(writefile)
writer.writerows(rank_list)
writefile.close()

writefile = open(mehthod+'_grouped_webs.csv', 'w')
writer = csv.writer(writefile)
writer.writerows(web_list)
writefile.close()

writefile = open(method+'_grouped_traces.csv', 'w')
writer = csv.writer(writefile)
writer.writerows(trace_list)
writefile.close()

writefile = open(method+'_grouped_charges.csv', 'w')
writer = csv.writer(writefile)
writer.writerows(charge_list)
writefile.close()

writefile = open(method+'_grouping_rank_counter.csv', 'w')
writer = csv.writer(writefile)
colnames = ['rank','count']
writer.writerow(colnames)
for rank, count in sorted(rank_counter.items()):
    values = [rank, count]
    writer.writerow(values)
writefile.close()
