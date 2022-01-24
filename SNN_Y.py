import numpy as np
import csv
import seaborn as sns; sns.set()
from network_functions import generate_triplets, embedding_model, complete_model
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import rand_score
from sklearn.metrics import matthews_corrcoef, accuracy_score, confusion_matrix
import itertools
from keras import backend as K
from math import floor
import statistics as stat
from sklearn.preprocessing import StandardScaler

#%% Data

# import data
web_data = []
with open('/Users/elliheyes/Documents/PhD/Data/Brane_Webs/SL2Z_HW_relabel_W.csv','r') as file:
    csvreader = csv.reader(file)
    for row in csvreader:
        web_data.append(row)

# format data
for i in range(len(web_data)):
    for j in range(len(web_data[i])):
        web_data[i][j] = web_data[i][j].replace('{', '')
        web_data[i][j] = web_data[i][j].replace('}', '')
        web_data[i][j] = web_data[i][j].replace(' ', '')
        web_data[i][j] = web_data[i][j].split (",")
        for k in range(len(web_data[i][j])):
            web_data[i][j][k] = int(web_data[i][j][k])

# save web matrices and class labels to lists
webs, labels = [], []
for i in range(len(web_data)):
    for j in range(len(web_data[i])):
        webs.append(web_data[i][j])
        labels.append(i)  
        
# scale data
scaler = StandardScaler()
webs = scaler.fit_transform(webs)
              
# zip data together 
data = [[webs[index],labels[index]] for index in range(len(webs))]

# define the number of data points in each validation split
s = int(floor(len(webs)/5)) 

# define data lists, each with 5 sublists 
training_inputs, training_outputs, testing_inputs, testing_outputs = [], [], [], []
for i in range(5):
    training_inputs.append([datapoint[0] for datapoint in data[:i*s]]+[datapoint[0] for datapoint in data[(i+1)*s:]])
    training_outputs.append([datapoint[1] for datapoint in data[:i*s]]+[datapoint[1] for datapoint in data[(i+1)*s:]])
    testing_inputs.append([datapoint[0] for datapoint in data[i*s:(i+1)*s]])
    testing_outputs.append([datapoint[1] for datapoint in data[i*s:(i+1)*s]])
    
#%% Model

# train and test the model 5 times recording the performance scores 
accuracies = []
mccs = []
for i in range(5):
    print(i)
    
    # define training and testing data
    X_train = training_inputs[i]
    y_train = training_outputs[i]
    X_test = testing_inputs[i]
    y_test = testing_outputs[i]
    
    # reshape input data
    X_train = np.array(X_train).reshape(np.array(X_train).shape[0], 2, 3, 1)
    X_test = np.array(X_test).reshape(np.array(X_test).shape[0], 2, 3, 1)

    # create instances of both the test and train batch generators and the complete model
    train_generator = generate_triplets(X=X_train, y=y_train, N_label=len(np.unique(y_train)))
    test_generator = generate_triplets(X=X_test, y=y_test, N_label=len(np.unique(y_test)))
    batch = next(train_generator)
    
    base_model = embedding_model()
    model = complete_model(base_model)

    # fit the model using triplet webs provided by the train batch generator
    history = model.fit(train_generator, 
                        validation_data=test_generator, 
                        epochs=100, 
                        steps_per_epoch=1000, 
                        validation_steps=1000,
                        verbose=2)

    # use the trained model to generate embeddings for the webs
    embeddings = base_model.predict(np.array(X_test).reshape(-1,2,3,1))
    
    # generate a list of all pairs of webs
    pairs_list = list(itertools.combinations(range(len(X_test)), 2))

    # determine equivalence predictions of pairs based on the distance between their embeddings 
    predictions = []
    truth = []
    for i in range(len(pairs_list)):
        index1 = pairs_list[i][0]
        index2 = pairs_list[i][1]

        embedding1 = embeddings[index1]
        embedding2 = embeddings[index2]
    
        distance = K.sum(K.square(embedding1-embedding2))
    
        if distance < 1:
            predictions.append(1)
        else:
            predictions.append(0)
        
        if (y_test[index1] == y_test[index2]):
            truth.append(1)
        else:
            truth.append(0)
    
    # determine the accuracy and MCC score of predictions
    accuracies.append(accuracy_score(truth,predictions))
    mccs.append(matthews_corrcoef(truth,predictions))

# save the trained weights.
model.save_weights('model_y.hdf5')

#%% Results

# compute the average accuracy and MCC score as well as the standard deviations
av_acc = np.mean(accuracies)
av_mcc = np.mean(mccs)
std_acc = stat.stdev(accuracies)
std_mcc = stat.stdev(mccs)

print(f'Accuracy: {av_acc} +/- {std_acc}')
print(f'MCC: {av_mcc} +/- {std_mcc}')

# get the confusion matrix
tn, fp, fn, tp = confusion_matrix(truth, predictions).ravel()
print('TN: ',tn)
print('FP: ',fp)
print('TP: ',tp)
print('FN: ',fn)

# fit kmeans clustering to the embeddings
kmeans = KMeans(n_clusters=len(np.unique(y_test))).fit(np.array(embeddings))

# get kmeans labels
kmeans_labels = kmeans.labels_

# determine rand score of kmeans labels 
rand_score = rand_score(y_test, kmeans_labels)
print('Rand Score: ', rand_score)
