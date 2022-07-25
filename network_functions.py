from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Convolution2D, Input, Lambda
from tensorflow.keras import backend as K
import numpy as np

def get_web(label, X, y):
    """Choose a web from the training or test data with the
    given label."""
    idx = np.random.randint(len(y))
    while y[idx] != label:
        idx = np.random.randint(len(y))
    return X[idx]
    
def get_triplet(X, y, N_label):
    """Choose a triplet (anchor, positive, negative) of webs
    such that anchor and positive have the same label and
    anchor and negative have different labels."""
    n = a = np.random.randint(N_label)
    while n == a:
        n = np.random.randint(N_label)
    a, p = get_web(a, X, y), get_web(a, X, y)
    n = get_web(n, X, y)
    return a, p, n

def generate_triplets(X, y, N_label, batch_size=32):
    """Generate an un-ending stream of triplets for training or test."""
    while True:
        list_a = []
        list_p = []
        list_n = []

        for i in range(batch_size):
            a, p, n = get_triplet(X, y, N_label)
            list_a.append(a)
            list_p.append(p)
            list_n.append(n)
            
        A = np.array(list_a, dtype='float32')
        P = np.array(list_p, dtype='float32')
        N = np.array(list_n, dtype='float32')

        label = np.ones(batch_size)
        yield [A, P, N], label
        
def identity_loss(y_true, y_pred):
    return K.mean(y_pred)

def triplet_loss(x, alpha = 5):
    """Compute the triplet loss."""
    anchor,positive,negative = x
    
    pos_dist = K.sum(K.square(anchor-positive),axis=1)
    neg_dist = K.sum(K.square(anchor-negative),axis=1)
  
    basic_loss = pos_dist-neg_dist+alpha
    loss = K.maximum(basic_loss,0.0)
    return loss

def embedding_model():
    """Build the convolution model."""
    model = Sequential()
    model.add(Convolution2D(8, (2, 2), input_shape=(2,3,1),activation='relu'))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10))
    return model

def complete_model(base_model): 
    """Build the complete model with three embedding models
    and minimise the loss between their output embeddings."""
    input_1 = Input((2, 3, 1))
    input_2 = Input((2, 3, 1))
    input_3 = Input((2, 3, 1))
        
    A = base_model(input_1)
    P = base_model(input_2)
    N = base_model(input_3)
   
    loss = Lambda(triplet_loss)([A, P, N]) 
    model = Model(inputs=[input_1, input_2, input_3], outputs=loss)
    model.compile(loss=identity_loss, optimizer=Adam(0.0001))
    
    return model
