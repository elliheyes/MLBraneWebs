from network_functions import generate_triplets, embedding_model, complete_model
import numpy as np
import sqlite3 as sql
import pandas as pd

#%% Data

# create file path: choose '3leg_data_Y.db' or '3leg_data_X.db'
dbfile = '3leg_data_Y.db'

# import data
with sql.connect(dbfile) as db: 
    c = db.cursor()
    df = pd.read_sql_query("SELECT * FROM data", db)
    headings = df.columns.values
    data = df.values
del(c,df)

# define a dataframe
df = pd.DataFrame(data = data, columns = headings)

# get lists of webs and labels
webs, labels = [], []
for i in range(len(df)):
    labels.append(df['label'][i])
    webs.append([[df['P1'][i],df['P2'][i],df['P3'][i]],[df['Q1'][i],df['Q2'][i],df['Q3'][i]]]) 
              
# split the data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(webs, labels, dataY, test_size=0.2)

# reshape the input data
X_train = np.array(X_train).reshape(np.array(X_train).shape[0], 2, 3, 1)
X_test = np.array(X_test).reshape(np.array(X_test).shape[0], 2, 3, 1)

# create instances of both the test and train batch generators 
train_generator = generate_triplets(X=X_train, y=y_train, N_label=2)
test_generator = generate_triplets(X=X_test, y=y_test, N_label=2)
batch = next(train_generator)

# build the complete model
base_model = embedding_model()
model = complete_model(base_model)

# fit the model using triplet webs provided by the train batch generator
history = model.fit(train_generator, 
                    validation_data=test_generator, 
                    epochs=10, 
                    steps_per_epoch=1000, 
                    validation_steps=1000,
                    verbose=2)

# use the trained model to generate embeddings for the webs in the test set
test_embeddings = base_model.predict(np.array(X_test).reshape(-1,2,3,1))

# determine equivalence predictions of pairs based on the distance between their embeddings 
predictions = []
truth = []
for i in range(len(pairs_list)):
    index1 = pairs_list[i][0]
    index2 = pairs_list[i][1]

    embedding1 = tes_embeddings[index1]
    embedding2 = test_embeddings[index2]
    
    distance = K.sum(K.square(embedding1-embedding2))
    
    if distance < 1:
        predictions.append(1)
    else:
        predictions.append(0)
        
    if (labels[index1] == labels[index2]):
        truth.append(1)
    else:
        truth.append(0)
        
# determine the accuracy and MCC score of predictions
accuracy = accuracy_score(truth,predictions)
mcc = matthews_corrcoef(truth,predictions)
print('Accuracy: ', accuracy)
print('MCC: ', mcc)

# plot embeddings with classes colour coded
all_embeddings = base_model.predict(np.array(webs).reshape(-1,2,3,1))
tsne = TSNE(n_components=2).fit_transform(all_embeddings)
tsne_x = [tsne[i][0] for i in range(len(tsne))]
tsne_y = [tsne[i][1] for i in range(len(tsne))]
palette = np.array(sns.color_palette("hls", len(np.unique(labels))))
f = plt.figure(figsize=(8, 8))
ax = plt.subplot(aspect='equal')
sc = ax.scatter(tsne_x, tsne_y, lw=0, alpha = 0.5, s=40, c=palette[labels])
plt.xlim(-25, 25)
plt.ylim(-25, 25)
ax.axis('off')
ax.axis('tight')

# fit kmeans clustering to the web embeddings
kmeans = KMeans(n_clusters=len(np.unique(labels))).fit(all_embeddings)
kmeans_labels = kmeans.labels_

# get rand index score of kmeans clustering
kmeans_rand_score = rand_score(labels, kmeans_labels)
print('K-Means Rand Score: ',kmeans_rand_score)
    
