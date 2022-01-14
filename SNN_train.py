import numpy as np
import matplotlib.pyplot as plt
import sqlite3 as sql
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns; sns.set()
from network_functions import generate_triplets, embedding_model, complete_model

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

# split the data into train and test sets and reshape the input data
X_train, X_test, y_train, y_test = train_test_split(webs, labels, test_size=0.2, random_state=42)

X_train = np.array(X_train).reshape(np.array(X_train).shape[0], 2, 3, 1)
X_test = np.array(X_test).reshape(np.array(X_test).shape[0], 2, 3, 1)

# create instances of both the test and train batch generators
train_generator = generate_triplets(X=X_train, y=y_train, N_label=len(equiv_groups))
test_generator = generate_triplets(X=X_test, y=y_test, N_label=len(equiv_groups))
batch = next(train_generator)

# build the model
base_model = embedding_model()
model = complete_model(base_model)

# fit the model using triplet webs provided by the train batch generator
history = model.fit(train_generator, 
                    validation_data=test_generator, 
                    epochs=100, 
                    verbose=2,steps_per_epoch=100, 
                    validation_steps=100)

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

