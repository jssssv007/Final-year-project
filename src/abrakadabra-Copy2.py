#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pandas')
get_ipython().system('pip install -U scikit-learn')
get_ipython().system('pip install plotly')
get_ipython().system('pip install modularity')
get_ipython().system('pip install python-louvain')
get_ipython().system('pip install nltk')


# In[2]:


import nltk


# In[3]:


from tqdm import *
f1 =open("output1.txt", "r").read()
arr = []
for i in tqdm(f1.split("\n\n")):
    arr.append(i.split("\n"))
arr = arr[:-1]
brr=arr[:]


# In[4]:


for i in brr:
    if(i[2]==''):
        print(i)


# In[5]:


brr[0]


# In[6]:


for i in range(len(brr)):
    for j in range(5, 8):
        brr[i][j] = brr[i][j].split()
        brr[i][j] = brr[i][j][0] + ":" + brr[i][j][1]

d = {}
for i in range(len(brr)):
    d[brr[i][0]] = [brr[i][5], brr[i][6], brr[i][7], brr[i][2]]
for i in d:
    d[i][0] = d[i][0].split(":")
    d[i][1] = d[i][1].split(":")
    d[i][2] = d[i][2].split(":")
    k = d[i][-1].split(" ")
    d[i] = {
        d[i][0][0]: float(d[i][0][1]),
        d[i][1][0]: float(d[i][1][1]),
        d[i][2][0]: float(d[i][2][1]),
        k[0]: " ".join(k[i] for i in range(1, len(k))),
    }


# In[7]:


# g={}
# for i in d:
#     g[i] = {
#         "attr_sum": (5*d[i]['#hi'] + d[i]['#pi'] + d[i]['#upi'])//3,
#         "#a": d[i]["#a"],
#     }
# k = sorted(g.items(), key=lambda x: x[1]["attr_sum"], reverse=True)
# x=k[:10000]


# In[8]:


import cmath
g={}
for i in d:
    g[i] = {
        "attr_sum": ((5 * cmath.rect(math.exp(math.log(abs(d[i]["#hi"]))), cmath.phase(d[i]["#hi"])) + 
                  cmath.rect(math.exp(math.log(abs(d[i]["#upi"]))), cmath.phase(d[i]["#upi"])) + 
                  cmath.rect(math.exp(math.log(abs(d[i]["#pi"]))), cmath.phase(d[i]["#pi"]))).real )// 3,
        "#a": d[i]["#a"],
    }
# k = sorted(g.items(), key=lambda x: x[1]["attr_sum"], reverse=True)
# x=k[:10000]


# In[9]:


k = list(g.items())


# In[10]:


len(k)


# In[11]:


g['#index 8']


# In[12]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re 
affili={}
cs={}
for author in tqdm(g):
    
    affiliations_text = g[author]['#a'] # extract the affiliation text
    affiliations_list = re.split(';(?=\w)', affiliations_text) # split affiliations using semicolon delimiter
    # remove any HTML tags and email addresses from affiliations
    affiliations_list = [re.sub('<[^<]+?>|rfc822', '', affiliation).strip() for affiliation in affiliations_list]
    affili[author]=affiliations_list
# affili=sorted(affili,reverse=True)

po={}
for ind,i in tqdm(enumerate(affili)):
    try:
        vectorizer = CountVectorizer()
    
    # Convert the authors array to a matrix of word counts
        count_matrix = vectorizer.fit_transform(affili[i])
        
    # Compute pairwise cosine similarity between the rows of the count matrix
        cosine_sim = cosine_similarity(count_matrix)
        cs[i]=sum(cosine_sim)
        
    except:
        po[ind]=affili[i]
        
        
    
# k=0
# for i,j in zip(x,cs):
#     i[1]['#a']=np.round(cs[k],1)
#     k+=1


# In[48]:


cs


# In[14]:


po


# In[15]:


k[10602]


# In[16]:


xyyx={}
j=0
for ind,i in tqdm(enumerate(k)):
    if(ind in po):
        continue
    else:
        xyyx[j]=i
        j+=1


# In[17]:


xyyx[10602][1],xyyx[10602]


# In[18]:


len(k),len(g),len(xyyx)


# In[19]:


# at_dic={}
# for i in x:
#     at_dic[i[0]]={"attr_sum":i[1]['attr_sum']}
ffk={}
for ind,i in tqdm(enumerate(xyyx)):
    ffk[xyyx[ind][0]]=xyyx[ind][1]


# In[20]:


ffk['#index 10']['#a']


# In[21]:


len(ffk)


# In[22]:


author


# In[23]:


len(affili)


# In[24]:


# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# # import numpy as np
# # import re 
# # affili1=[]
# # cs={}
# # for author in tqdm(k):
    
# #     affiliations_text = k[author]['#a'] # extract the affiliation text
# #     affiliations_list = re.split(';(?=\w)', affiliations_text) # split affiliations using semicolon delimiter
# #     # remove any HTML tags and email addresses from affiliations
# #     affiliations_list = [re.sub('<[^<]+?>|rfc822', '', affiliation).strip() for affiliation in affiliations_list]
# #     affili1.append(affiliations_list)
# # # affili=sorted(affili,reverse=True)
# cs1={}
# poo={}
# for ind,i in tqdm(enumerate(affili)):
#     try:
#         vectorizer = CountVectorizer()
    
#     # Convert the authors array to a matrix of word counts
#         count_matrix = vectorizer.fit_transform(affili[i])
        
#     # Compute pairwise cosine similarity between the rows of the count matrix
#         cosine_sim = cosine_similarity(count_matrix)
#         cs1[ind]=sum(cosine_sim)
        
#     except:
#         poo[ind]=i
        
        
    
# # k=0
# # for i,j in zip(x,cs):
# #     i[1]['#a']=np.round(cs[k],1)
# #     k+=1


# In[25]:


xyyx[0]


# In[26]:


kk=0
for i,j in zip(ffk,cs):
    print(i)
    ffk[i]['#a']=np.round(cs[j],1)
    kk+=1


# In[27]:


import itertools
from itertools import *
t=dict(itertools.islice(dict(sorted(ffk.items(), key=lambda x: x[1]["#a"], reverse=True)).items(),10000))


# In[28]:


t


# In[29]:


# your_dict={}
# c=1
# for i in at_dic:
#     your_dict[f"key{c}"]=at_dic[i]
#     c+=1
# #print(your_dict)


# In[30]:


your_dict1={}
c1=1
for i in t:
    your_dict1[c1]={'#similarity_score' : t[i]['#a']}
    c1+=1
print(your_dict1)


# In[31]:


# import pandas as pd
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score
# import matplotlib.pyplot as plt

# # Create a DataFrame from the dictionary
# data = pd.DataFrame.from_dict(your_dict1, orient='index')

# # Define the range of K values to test
# k_range = trange(2, len(data))

# # Calculate the silhouette scores for each value of K
# silhouette_scores = []
# for k in k_range:
#     kmeans = KMeans(n_clusters=k, random_state=42,n_init='auto')
#     labels = kmeans.fit_predict(data)
#     silhouette_scores.append(silhouette_score(data, labels))

# # Plot the results
# plt.plot(k_range, silhouette_scores)
# plt.xlabel('Number of clusters')
# plt.ylabel('Silhouette score')
# plt.title('Silhouette analysis')
# plt.show()

# # Find the best number of clusters
# best_k = k_range[silhouette_scores.index(max(silhouette_scores))]
# print('The best number of clusters is', best_k)


# In[32]:


# x[0]


# In[33]:


your_dict1[14]


# In[34]:


len(your_dict1)


# In[ ]:





# In[35]:


affili['#index 8']


# In[44]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objs as go
import plotly.offline as pyo

# Convert data to DataFrame

df = pd.DataFrame.from_dict(your_dict1, orient='index')

# Perform k-means clustering
kmeans = KMeans(n_clusters=7, random_state=0,n_init='auto').fit(df)

# Add cluster labels to DataFrame
df['cluster'] = kmeans.labels_

# Plot clusters using mpl_toolkits
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df.index.values.astype(str), df['#similarity_score'], df['cluster'], c=kmeans.labels_.astype(float))
ax.set_xlabel('Data points')
ax.set_ylabel('Similarity Score')
ax.set_zlabel('Cluster')
plt.show()

# Plot clusters using Plotly
df.to_csv('kmeans_clustering_results.csv')


# In[64]:


from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
import pandas as pd

# Load the clustering results
df = pd.read_csv('kmeans_clustering_results.csv')

# Extract the data points and cluster labels
X = df.drop('cluster', axis=1)
labels = df['cluster']

# Compute the silhouette score
score = silhouette_score(X, labels)
ch_score = calinski_harabasz_score(X, labels)
print('Silhouette score:', score)
print('CH:', ch_score)


# In[47]:


df


# In[42]:


df['#similarity_score']


# In[43]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import plotly.graph_objs as go
import plotly.offline as pyo

# Convert data to DataFrame
df = pd.DataFrame.from_dict(your_dict1, orient='index')

# Perform k-means clustering
kmeans = KMeans(n_clusters=3, random_state=0).fit(df)

# Add cluster labels to DataFrame
df['cluster'] = kmeans.labels_

# Plot clusters using Plotly
fig = go.Figure(data=[go.Scatter3d(
    x=df.index.values.astype(str),
    y=df['#similarity_score'],
    z=df['cluster'],
    mode='markers',
    marker=dict(
        size=5,
        color=kmeans.labels_.astype(float),
        colorscale='Viridis',
        opacity=0.8
    )
)])
fig.update_layout(
    title='KMeans Clustering Results',
    scene=dict(
        xaxis_title='Data points',
        yaxis_title='Similarity Score',
        zaxis_title='Cluster'
    )
)
pyo.iplot(fig)

# Save results to CSV file
#df.to_csv('kmeans_clustering_results.csv', index=False)


# In[ ]:


ls


# In[ ]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import plotly.graph_objs as go
import plotly.offline as pyo
df2=pd.read_csv('kmeans_clustering_results.csv')

# Perform k-means clustering
kmeans = KMeans(n_clusters=3, random_state=0).fit(df2)

# Add cluster labels to DataFrame
df2['cluster'] = kmeans.labels_


# Plot clusters using Plotly
fig = go.Figure(data=[go.Scatter3d(
    x=df2.index.values.astype(str),
    y=df2['#similarity_score'],
    z=df2['cluster'],
    mode='markers',
    marker=dict(
        size=5,
        color=kmeans.labels_.astype(float),
        colorscale='Viridis',
        opacity=0.8
    )
)])
fig.update_layout(
    title='KMeans Clustering Results',
    scene=dict(
        xaxis_title='Data points',
        yaxis_title='Similarity Score',
        zaxis_title='Cluster'
    )
)
pyo.iplot(fig)


# In[ ]:


from sklearn.cluster import KMeans

# Convert the dictionary values to a list
similarity_scores = []
for key, value in your_dict1.items():
    similarity_scores.append(value['#similarity_score'])

# convert similarity scores to matrix
X = np.array(similarity_scores).reshape(-1, 1)
# Determine the optimal k value using the elbow method
elbow = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k).fit(X)
    elbow.append(kmeans.inertia_)

# Visualize the elbow plot
import matplotlib.pyplot as plt
plt.plot(range(1, 10), elbow)
plt.title('Elbow Plot')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# Based on the elbow plot, select the optimal k value and perform clustering
k = 3 # Replace this with the optimal k value determined from the elbow plot
kmeans = KMeans(n_clusters=k).fit(X)

# Print the cluster labels
labels = kmeans.labels_
for i in range(k):
    print(f"Cluster {i}: {[your_dict1[k]['#similarity_score'] for k, v in enumerate(labels) if v == i]}")


# In[ ]:


# import networkx as nx
# import community

# # Create a weighted graph from the similarity scores
# G = nx.Graph()
# for node, data in tqdm(your_dict1.items()):
#     G.add_node(node)
#     for neighbor, neighbor_data in your_dict1.items():
#         if node != neighbor:
#             similarity_score = data['#similarity_score'] + neighbor_data['#similarity_score']
#             G.add_edge(node, neighbor, weight=similarity_score)

# # Apply the Louvain method
# partition = community.best_partition(G, weight='weight')

# # Print the partition
# print(partition)


# In[49]:


len(t)


# In[50]:


t


# In[63]:


dict(itertools.islice(t.items(),136))


# In[53]:





# In[56]:


l[:136]


# In[58]:


l[0][1]


# In[ ]:


bcd=[]
for i in l:
    bcd.append()

