import numpy as np
import sqlite3 as sql
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler
import itertools
from math import floor
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from helpful_functions import matthews_correlation_coefficient

# import data
with sql.connect('3leg_data.db') as db: 
    c = db.cursor()
    df = pd.read_sql_query("SELECT * FROM data", db)
    headings = df.columns.values
    data = df.values
del(c,df)

# define a dataframe
df = pd.DataFrame(data = data, columns = headings)
    
# create a list of all possible pairs of webs 
pairs_list = list(itertools.combinations(range(len(df)), 2))

# combine the data
data = []
for i in range(len(pairs_list)):
    index1 = pairs_list[i][0]
    index2 = pairs_list[i][1]
    data.append([df['p1'][index1]*df['m1'][index1]]+[df['p2'][index1]*df['m2'][index1]]+[df['p3'][index1]*df['m3'][index1]]+
                [df['q1'][index1]*df['m1'][index1]]+[df['q2'][index1]*df['m2'][index1]]+[df['q3'][index1]*df['m3'][index1]]+
                [df['total_monodromy_trace'][index1]]+[df['asymptotic_charge'][index1]]+[df['rank'][index1]]+
                [df['p1'][index2]*df['m1'][index2]]+[df['p2'][index2]*df['m2'][index2]]+[df['p3'][index2]*df['m3'][index2]]+
                [df['q1'][index2]*df['m1'][index2]]+[df['q2'][index2]*df['m2'][index2]]+[df['q3'][index2]*df['m3'][index2]]+
                [df['total_monodromy_trace'][index2]]+[df['asymptotic_charge'][index2]]+[df['rank'][index2]])

# fit a Min Max scaler to the data
scaler = MinMaxScaler()
scaler.fit(data)

# shuffle data
df = df.sample(frac=1).reset_index(drop=True)

# extract the first half of the data
p1 = df['p1'][:floor(len(df)/2)]
p2 = df['p2'][:floor(len(df)/2)]
p3 = df['p3'][:floor(len(df)/2)]
q1 = df['q1'][:floor(len(df)/2)]
q2 = df['q2'][:floor(len(df)/2)]
q3 = df['q3'][:floor(len(df)/2)]
m1 = df['m1'][:floor(len(df)/2)]
m2 = df['m2'][:floor(len(df)/2)]
m3 = df['m3'][:floor(len(df)/2)]
traces = df['total_monodromy_trace'][:floor(len(df)/2)]
charges = df['asymptotic_charge'][:floor(len(df)/2)]
ranks = df['rank'][:floor(len(df)/2)]

# create a list of all possible pairs of webs in the first half of the data
pairs_list = list(itertools.combinations(range(floor(len(df)/2)), 2))

# define total dataset size
size = 10000

# define count variables
yes = 0
no = 0
i = 0

# create data lists with an equal number of equivalent and non equivalent webs
dataX, dataY = [], []
while True:
    index1 = pairs_list[i][0]
    index2 = pairs_list[i][1]
    
    if (traces[index1] == traces[index2] and charges[index1] == charges[index2] and ranks[index1] == ranks[index2]):
        if yes < floor(size/2):
            dataY.append(1)
            dataX.append([p1[index1]*m1[index1]]+[p2[index1]*m2[index1]]+[p3[index1]*m3[index1]]+
                           [q1[index1]*m1[index1]]+[q2[index1]*m2[index1]]+[q3[index1]*m3[index1]]+
                           [traces[index1]]+[charges[index1]]+[ranks[index1]]+
                           [p1[index2]*m1[index2]]+[p2[index2]*m2[index2]]+[p3[index2]*m3[index2]]+
                           [q1[index2]*m1[index2]]+[q2[index2]*m2[index2]]+[q3[index2]*m3[index2]]+
                           [traces[index2]]+[charges[index2]]+[ranks[index2]])
        yes += 1
                    
    else:
        if no < floor(size/2):
            dataY.append(0)
            dataX.append([p1[index1]*m1[index1]]+[p2[index1]*m2[index1]]+[p3[index1]*m3[index1]]+
                           [q1[index1]*m1[index1]]+[q2[index1]*m2[index1]]+[q3[index1]*m3[index1]]+
                           [traces[index1]]+[charges[index1]]+[ranks[index1]]+
                           [p1[index2]*m1[index2]]+[p2[index2]*m2[index2]]+[p3[index2]*m3[index2]]+
                           [q1[index2]*m1[index2]]+[q2[index2]*m2[index2]]+[q3[index2]*m3[index2]]+
                           [traces[index2]]+[charges[index2]]+[ranks[index2]])
        no += 1
    
    i += 1
    
    if len(dataY) >= size:
        break
    
# rescale the data using the fitted scaler
dataX_scaled = scaler.transform(dataX)

# split the data into train and test sets
trainX, testX, trainY, testY = train_test_split(dataX_scaled, dataY, test_size=0.2)

# build the binary classification model
input_layer = Input(shape=(18,))
x = Dense(50, activation="relu")(input_layer)
x = Dense(100)(x)
x = Dense(50)(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=input_layer, outputs=x)
   
# compile the model
model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['binary_accuracy', matthews_correlation_coefficient])
   
# fit the model to the data
history = model.fit(x=np.array(trainX), y=np.array(trainY), validation_data=(np.array(testX), np.array(testY)), epochs=100, batch_size=16)

# extract the second half of the data
p1 = df['p1'][floor(len(df)/2):].reset_index(drop=True)
p2 = df['p2'][floor(len(df)/2):].reset_index(drop=True)
p3 = df['p3'][floor(len(df)/2):].reset_index(drop=True)
q1 = df['q1'][floor(len(df)/2):].reset_index(drop=True)
q2 = df['q2'][floor(len(df)/2):].reset_index(drop=True)
q3 = df['q3'][floor(len(df)/2):].reset_index(drop=True)
m1 = df['m1'][floor(len(df)/2):].reset_index(drop=True)
m2 = df['m2'][floor(len(df)/2):].reset_index(drop=True)
m3 = df['m3'][floor(len(df)/2):].reset_index(drop=True)
traces = df['total_monodromy_trace'][floor(len(df)/2):].reset_index(drop=True)
charges = df['asymptotic_charge'][floor(len(df)/2):].reset_index(drop=True)
ranks = df['rank'][floor(len(df)/2):].reset_index(drop=True)

# define second half data lists
dataX, dataY = [], []
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

# append the second half data lists
for i in range(len(pairs_list)):
    
    index1 = pairs_list[i][0]
    index2 = pairs_list[i][1]
    
    label_1.append(index1)
    label_2.append(index2)
    
    p1_1.append(p1[index1])
    p1_2.append(p1[index2])
    p2_1.append(p2[index1])
    p2_2.append(p2[index2])
    p3_1.append(p3[index1])
    p3_2.append(p3[index2])
    q1_1.append(q1[index1])
    q1_2.append(q1[index2])
    q2_1.append(q2[index1])
    q2_2.append(q2[index2])
    q3_1.append(q3[index1])
    q3_2.append(q3[index2])
    m1_1.append(m1[index1])
    m1_2.append(m1[index2])
    m2_1.append(m2[index1])
    m2_2.append(m2[index2])
    m3_1.append(m3[index1])
    m3_2.append(m3[index2])
    trace_1.append(traces[index1])
    trace_2.append(traces[index2])
    charge_1.append(charges[index1])
    charge_2.append(charges[index2])
    rank_1.append(ranks[index1])
    rank_2.append(ranks[index2])
    
    dataX.append([p1[index1]*m1[index1]]+[p2[index1]*m2[index1]]+[p3[index1]*m3[index1]]+
                 [q1[index1]*m1[index1]]+[q2[index1]*m2[index1]]+[q3[index1]*m3[index1]]+
                 [traces[index1]]+[charges[index1]]+[ranks[index1]]+
                 [p1[index2]*m1[index2]]+[p2[index2]*m2[index2]]+[p3[index2]*m3[index2]]+
                 [q1[index2]*m1[index2]]+[q2[index2]*m2[index2]]+[q3[index2]*m3[index2]]+
                 [traces[index2]]+[charges[index2]]+[ranks[index2]])
        
    if (traces[index1] == traces[index2] and charges[index1] == charges[index2] and ranks[index1] == ranks[index2]):
        dataY.append(1)
    else:
        dataY.append(0)

# rescale the data using the fitted scaler
dataX_scaled = scaler.transform(dataX)

# use the trained network to predict whether or not the web pairs from the second half of the data are equivalent 
predictions = np.round(np.array(model.predict(np.array(dataX_scaled))))
predictions = [predictions[i][0] for i in range(len(predictions))]

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
