'''TDA: Persistent Homology for braneweb data'''
#TDA of embeddings - prerun the SNN.py script to import chosen web data (X or Y) and train the SNN for embedding
#Import additional libraries
from ripser import ripser          #...github tda library available at: https://github.com/scikit-tda/ripser.py.git
from persim import plot_diagrams   #...additional library for persistence diagram plotting
import matplotlib.pyplot as plt 

#%% Web raw-data TDA
#Slice the webdata
webdata = df.values[:,1:-1]

#Compute persistent homology
persistence_diagrams_raw = ripser(webdata,maxdim=1)['dgms'] #...maxdim p => compute cohomology for classes up to H_p

#%% #Plot persistence diagram
fig = plt.figure()
axes = plt.axes()
plot_diagrams(persistence_diagrams_raw, lifetime=False, show=True, ax=axes) 
fig.tight_layout()
#fig.savefig('./WebDataPersistenceLifetime.pdf')

#%% Web embedding TDA
#Generate embedding for all webs using the trained SNN sub-network (train on X or Y data for respective embedding)
full_embeddings = base_model.predict(np.array(webs).reshape(-1,2,3,1)) 

#Compute persistent homology
persistence_diagrams_embed = ripser(full_embeddings,maxdim=1)['dgms'] #...maxdim p => compute cohomology for classes up to H_p

#%% #Plot persistence diagram
fig = plt.figure()
axes = plt.axes()
plot_diagrams(persistence_diagrams_embed, lifetime=False, show=True, ax=axes) 
fig.tight_layout()
#fig.savefig('./EmbeddingsPersistenceLifetime.pdf')

