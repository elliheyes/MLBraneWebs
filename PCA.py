import sqlite3 as sql 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns


### Data ###

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

# save web matrices, classes and ranks to lists
webs, ranks, classes = [], [], []
for i in range(len(df)):
    webs.append(np.array([df['p1'][i]*df['m1'][i]]+[df['p2'][i]*df['m2'][i]]+[df['p3'][i]*df['m3'][i]]+
                [df['q1'][i]*df['m1'][i]]+[df['q2'][i]*df['m2'][i]]+[df['q3'][i]*df['m3'][i]]))
    ranks.append(df['rank'][i])
    for j in range(len(equiv_groups)):
        if i in equiv_groups[j]:
            classes.append(j)

# scale the data 
scaler = StandardScaler()
webs_scaled = scaler.fit_transform(webs)


### PCA ###

# create a pca instance and fit to the scaled data 
pca = PCA(n_components=6)
principal_components = pca.fit_transform(webs_scaled)

# save the pca components to a dataframe
PCA_components = pd.DataFrame(principal_components)

# plot the first two pca components of the data colour coded to the corresponding web class
palette = np.array(sns.color_palette("hls", len(equiv_groups)))
scatter = plt.scatter(PCA_components[0], PCA_components[1], c=palette[np.array(classes).astype(np.int)])
plt.xlabel('PCA 1') 
plt.ylabel('PCA 2')

# look at one example class

ex_class = 0

# extract the webs that belong to the class
idxs_ex = []
webs_ex = []
for i in range(len(classes)):
    if classes[i] == ex_class:
        webs_ex.append(webs[i])
        idxs_ex.append(i)
        rank = ranks[i]
        
# scale the data 
scaler = StandardScaler()
webs_ex_scaled = scaler.fit_transform(webs_ex)

# create and fit a new pca instance 
pca_ex = PCA(n_components=6)
principal_components_ex = pca_ex.fit_transform(webs_ex_scaled)

# save the pca components to a dataframe
PCA_components_ex_1 = PCA_components.iloc[idxs_ex]
PCA_components_ex_2 = pd.DataFrame(principal_components_ex)

# plot the first two pca components of the example class web data using the original fitted pca 
scatter = plt.scatter(PCA_components_ex_1[0], PCA_components_ex_1[1])
plt.title(f"Class {ex_class}"
          "\n"
          f"Rank: {rank}"
          "\n"
          f"Web: [{webs[equiv_groups[ex_class][0]][:3]},{webs[equiv_groups[ex_class][0]][3:]}]")
plt.xlabel('PCA 1') 
plt.ylabel('PCA 2')

# plot the first two pca components of the example class web data using the new fitted pca
scatter = plt.scatter(PCA_components_ex_2[0], PCA_components_ex_2[0])
plt.title(f"Class {ex_class}"
          "\n"
          f"Rank: {rank}"
          "\n"
          f"Web: [{webs[equiv_groups[ex_class][0]][:3]},{webs[equiv_groups[ex_class][0]][3:]}]")
plt.xlabel('PCA 1') 
plt.ylabel('PCA 2')

