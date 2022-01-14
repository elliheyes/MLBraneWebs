import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import rand_score
import sqlite3 as sql
import pandas as pd
import seaborn as sns; sns.set()
from network_functions import embedding_model, complete_model

# create file path
dbfile = '3leg_data.db'

# import data
with sql.connect(dbfile) as db: 
    c = db.cursor()
    df = pd.read_sql_query("SELECT * FROM data", db)
    headings = df.columns.values
    data = df.values
del(c,df)

# define a dataframe
df = pd.DataFrame(data = data, columns = headings)

# shuffle data 
df = df.sample(frac=1).reset_index(drop=True)

# create lists of equivalent webs
idx = []
for i in range(len(df)):
    print('Loop 1: ',i/len(df))
    idx_list = []
    for j in range(len(df)):
        if (df['total_monodromy_trace'][i] == df['total_monodromy_trace'][j] and 
            df['asymptotic_charge'][i] == df['asymptotic_charge'][j] and 
            df['rank'][i] == df['rank'][j]):
            idx_list.append(j)
    idx.append(idx_list)

equiv_groups = []
for i in idx:
    if i not in equiv_groups:
        equiv_groups.append(i)
   
# create a list of web matrices and corresponding labels
webs = []
labels = []
for i in range(len(df)):
    print('Loop 2: ',i/len(df))
    webs.append([[df['p1'][i]*df['m1'][i],df['p2'][i]*df['m2'][i],df['p3'][i]*df['m3'][i]],
                [df['q1'][i]*df['m1'][i],df['q2'][i]*df['m2'][i],df['q3'][i]*df['m3'][i]]])
    for j in range(len(equiv_groups)):
        if i in equiv_groups[j]:
            labels.append(j)

# build the model
base_model = embedding_model()
model = complete_model(base_model)

# load the saved trained weights.
model.load_weights('model.hdf5')

# use the trained model to generate embeddings for the webs
embeddings = base_model.predict(np.array(webs).reshape(-1,2,3,1))

# fit kmeans clustering to the embeddings
kmeans = KMeans(n_clusters=len(equiv_groups), random_state=0).fit(np.array(embeddings))

# get kmeans labels
kmeans_labels = kmeans.labels_

# determine rand score of kmeans labels 
print('Rand Score: ', rand_score(labels, kmeans_labels))

# append pandas dataframe with the correct labels and kmeans labels
df['label'] = labels
df['kmeans_label'] = kmeans_labels

# open a connection to a new database and create a new table in that database for the results
conn = sql.connect('SNN_results.db')
df.to_sql('data', conn)
