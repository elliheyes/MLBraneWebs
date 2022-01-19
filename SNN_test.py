import numpy as np
import sqlite3 as sql
import pandas as pd
from network_functions import embedding_model, complete_model
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import rand_score
import itertools
from keras import backend as K
from sklearn.metrics import matthews_corrcoef, accuracy_score

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
    webs.append([[df['p1'][i]*df['m1'][i],df['p2'][i]*df['m2'][i],df['p3'][i]*df['m3'][i]],
                [df['q1'][i]*df['m1'][i],df['q2'][i]*df['m2'][i],df['q3'][i]*df['m3'][i]]])
    for j in range(len(equiv_groups)):
        if i in equiv_groups[j]:
            labels.append(j)

# append the dataframe with the class labels 
df['label'] = labels

# build model
base_model = embedding_model()
model = complete_model(base_model)

# load the saved weights.
model.load_weights('model.hdf5')

# use the trained model to generate embeddings for the webs
embeddings = base_model.predict(np.array(webs).reshape(-1,2,3,1))

# fit kmeans clustering to the embeddings
kmeans = KMeans(n_clusters=len(equiv_groups), random_state=0).fit(np.array(embeddings))

# get kmeans labels
kmeans_labels = kmeans.labels_

# determine rand score of kmeans labels 
print('Rand Score: ', rand_score(labels, kmeans_labels))

# append the dataframe with the kmeans labels
df['kmeans_label'] = kmeans_labels

# open a connection to a new database and create a new table in that database for the results
conn = sql.connect('SNN_kmeans.db')
df.to_sql('data', conn)

# generate a list of all pairs of webs
pairs_list = list(itertools.combinations(range(len(df)), 2))

# determine equivalence predictions of pairs based on their embeddings
limit = 0.01
predictions = []
truth = []
for i in range(len(pairs_list)):
    index1 = pairs_list[i][0]
    index2 = pairs_list[i][1]
    
    embedding1 = embeddings[index1]
    embedding2 = embeddings[index2]
    
    distance = K.sum(K.square(embedding1-embedding2))
    
    if distance < limit:
        predictions.append(1)
    else:
        predictions.append(0)
        
    if (df['total_monodromy_trace'][index1] == df['total_monodromy_trace'][index2] and 
        df['asymptotic_charge'][index1] == df['asymptotic_charge'][index2] and
        df['rank'][index1] == df['rank'][index2]):
        truth.append(1)
    else:
        truth.append(0)

# determine the accuracy and MCC score of predictions
print('Accuracy: ', accuracy_score(truth,predictions))
print('MCC: ', matthews_corrcoef(truth,predictions))
