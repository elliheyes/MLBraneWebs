from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import sqlite3 as sql
import pandas as pd
import itertools
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
trm = base_model.predict(np.array(webs).reshape(-1,2,3,1))

# TSNE to use dimensionality reduction to visulaise the resultant embeddings
tsne = TSNE()
tsne_embeds = tsne.fit_transform(trm)

# create a scatter plot of all the the embeddings 
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

# create a list of all possible pairs of webs 
pairs_list = list(itertools.combinations(range(len(df)), 2))

# use the trained model to predict the equivalence of the web pairs
label_1, label_2 = [], []
p1_1, p2_1, p3_1 = [], [], []
p1_2, p2_2, p3_2 = [], [], []
q1_1, q2_1, q3_1 = [], [], []
q1_2, q2_2, q3_2 = [], [], []
m1_1, m2_1, m3_1 = [], [], []
m1_2, m2_2, m3_2 = [], [], []
rank_1, rank_2 = [], []
trace_1, trace_2 = [], []
charge_1, charge_2 = [], []
truth = []
predictions = []

limit = ...

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
    rank_1.append(df['rank'][index1])
    rank_2.append(df['rank'][index2])
    trace_1.append(df['total_monodromy_trace'][index1])
    trace_2.append(df['total_monodromy_trace'][index2])
    charge_1.append(df['asymptotic_charge'][index1])
    charge_2.append(df['asymptotic_charge'][index2])
    
    embedding1 = base_model.predict(np.array(webs[index1]).reshape(-1,2,3,1))
    embedding2 = base_model.predict(np.array(webs[index2]).reshape(-1,2,3,1))
    
    dist = K.sum(K.square(embedding1-embedding2),axis=1)
    
    if dist < limit:
        predictions.append(1)
    else:
        predictions.append(0)
    
    if (df['total_monodromy_trace'][index1] == df['total_monodromy_trace']labels[index2] and
        df['asymptotic_charge'][index1] == df['asymptotic_charge']labels[index2] and
        df['rank'][index1] == df['rank']labels[index2]):
        truth.append(1)
    else:
        truth.append(0)

# determine the accuracy of predictions
fp, fn, tp, tn = 0, 0, 0, 0
for i in range(len(predictions)):
    if predictions[i] == 1:
        if truth[i] == 1:
            tp += 1
        else:
            fp += 1
    else:
        if truth[i] == 1:
            fn += 1
        else:
            tn += 1
            
accuracy = (tp+tn)/len(predictions)
print('Accuracy: ', accuracy)

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
conn = sql.connect('NN_results.db')
results_df.to_sql('data', conn)
