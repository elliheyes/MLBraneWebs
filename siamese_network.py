from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import sqlite3 as sql
import pandas as pd
import itertools
from collections import Counter
from sklearn.model_selection import train_test_split
import seaborn as sns; sns.set()
from network_functions import generate_triplets, embedding_model, complete_model

# import data
with sql.connect('3leg_data_3.db') as db: 
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

# split the data into train and test sets and reshape the input data
X_train, X_test, y_train, y_test = train_test_split(webs, labels, test_size=0.2, random_state=42)

X_train = np.array(X_train).reshape(np.array(X_train).shape[0], 2, 3, 1)
X_test = np.array(X_test).reshape(np.array(X_test).shape[0], 2, 3, 1)

# create instances of both the test and train batch generators and the complete model
train_generator = generate_triplets(X=X_train, y=y_train, N_label=len(equiv_groups))
test_generator = generate_triplets(X=X_test, y=y_test, N_label=len(equiv_groups))
batch = next(train_generator)

base_model = embedding_model()
model = complete_model(base_model)

# fit the model using triplet webs provided by the train batch generator
history = model.fit(train_generator, 
                    validation_data=test_generator, 
                    epochs=100, 
                    verbose=2,steps_per_epoch=20, 
                    validation_steps=30)

# save the trained weights.
model.save_weights('model.hdf5')

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training and Validation Losses',size = 20)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

# using the newly trained model compute the embeddings for all webs
embeddings = base_model.predict(np.array(webs).reshape(-1,2,3,1))

# fit kmeans clustering to the embeddings
kmeans = KMeans(n_clusters=len(equiv_groups), random_state=0).fit(np.array(embeddings))

# get kmeans labels
kmeans_labels = kmeans.labels_

# determine accuracy of kmeans labels
rand_score(labels, kmeans_labels)

# TSNE to use dimensionality reduction to visulaise the resultant embeddings
tsne = TSNE()
tsne_embeds = tsne.fit_transform(embeddings)

# create a scatter plot of all the the embeddings labeled according to their true class
def scatter(x, labels, subtitle=None):
    palette = np.array(sns.color_palette("hls", len(equiv_groups)))
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    ax.scatter(x[:,0], x[:,1], lw=0,alpha = 0.5, s=40, c=palette[np.array(labels).astype(np.int)] )
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')
scatter(tsne_embeds, labels)

# create a scatter plot of all the the embeddings labeled according to their kmeans class
def scatter(x, labels, subtitle=None):
    palette = np.array(sns.color_palette("hls", len(equiv_groups)))
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    ax.scatter(x[:,0], x[:,1], lw=0,alpha = 0.5, s=40, c=palette[np.array(labels).astype(np.int)] )
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')
scatter(tsne_embeds, kmeans_labels)
