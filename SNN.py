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

# use the trained model to generate embeddings for the webs
embeddings = base_model.predict(np.array(X_test).reshape(-1,2,3,1))
    
