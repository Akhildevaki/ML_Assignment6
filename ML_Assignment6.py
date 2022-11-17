#!/usr/bin/env python
# coding: utf-8

# In[9]:


### importing required libraries
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from sklearn import metrics

from sklearn.preprocessing import StandardScaler, normalize

from sklearn.cluster import AgglomerativeClustering


# In[10]:


### reading data from csv file
df = pd.read_csv('C:\\Users\\Rams\Downloads\\CC GENERAL.csv')


# In[11]:


df.describe()


# In[12]:


df.info()


# In[13]:


df.head()


# In[14]:


### Removing Categorical column from the data
df.drop(['TENURE'], axis=1, inplace=True)

### Removing unnecessary column from the data
df.drop(['CUST_ID'], axis=1, inplace=True)


# In[15]:


### finding null values in data
df.isna().sum()


# In[16]:


### filling null values
df['MINIMUM_PAYMENTS'].fillna(df['MINIMUM_PAYMENTS'].mean(), inplace=True)
df['CREDIT_LIMIT'].fillna(df['CREDIT_LIMIT'].mean(), inplace=True)


# ### Scaling of Features using Standard Scaler funtion

# In[17]:


scaler = StandardScaler()
scaled_df = scaler.fit_transform(df)


# In[18]:


scaled_df


# ### Normalizing of features using normalize funtion

# In[12]:


normalized_df = normalize(scaled_df)


# In[13]:


normalized_df


# ### Using Principal Component Analysis with 2 components

# In[14]:


pca = PCA(2)
x_pca = pca.fit_transform(normalized_df)
df2 = pd.DataFrame(data=x_pca)
df2


# In[15]:


k_clusters=[]
silhoutte_Scores=[]


# ### Applying Agglomerative Clustering using 2 clusters

# In[16]:


nclusters = 2
ac = AgglomerativeClustering(n_clusters=nclusters)
y_cluster_kmeans = ac.fit_predict(df2)
score = metrics.silhouette_score(df2, y_cluster_kmeans)
print("silhoutte score : {0} , with {1} clusters".format(score,nclusters))
labels = ac.labels_
k_clusters.append(nclusters)
silhoutte_Scores.append(score)
plt.title("Scatter plot for Agglomerative clustering using {}".format(nclusters))
plt.scatter(df2[0], df2[1], c=ac.labels_)
plt.show()


# ### Applying Agglomerative Clustering using 3 clusters

# In[17]:


nclusters = 3
ac = AgglomerativeClustering(n_clusters=nclusters)
y_cluster_kmeans = ac.fit_predict(df2)
score = metrics.silhouette_score(df2, y_cluster_kmeans)
print("silhoutte score : {0} , with {1} clusters".format(score,nclusters))
labels = ac.labels_
k_clusters.append(nclusters)
silhoutte_Scores.append(score)
plt.title("Scatter plot for Agglomerative clustering using {}".format(nclusters))
plt.scatter(df2[0], df2[1], c=ac.labels_)
plt.show()


# ### Applying Agglomerative Clustering using 4 clusters

# In[18]:


nclusters = 4
ac = AgglomerativeClustering(n_clusters=nclusters)
y_cluster_kmeans = ac.fit_predict(df2)
score = metrics.silhouette_score(df2, y_cluster_kmeans)
print("silhoutte score : {0} , with {1} clusters".format(score,nclusters))
labels = ac.labels_
k_clusters.append(nclusters)
silhoutte_Scores.append(score)
plt.title("Scatter plot for Agglomerative clustering using {}".format(nclusters))
plt.scatter(df2[0], df2[1], c=ac.labels_)
plt.show()


# ### Applying Agglomerative Clustering using 5 clusters

# In[19]:


nclusters = 5
ac = AgglomerativeClustering(n_clusters=nclusters)
y_cluster_kmeans = ac.fit_predict(df2)
score = metrics.silhouette_score(df2, y_cluster_kmeans)
print("silhoutte score : {0} , with {1} clusters".format(score,nclusters))
labels = ac.labels_
k_clusters.append(nclusters)
silhoutte_Scores.append(score)
plt.title("Scatter plot for Agglomerative clustering using {}".format(nclusters))
plt.scatter(df2[0], df2[1], c=ac.labels_)
plt.show()


# In[20]:


score


# ## Bar Graph for the silhoutte scores with different number of clusters

# In[21]:


plt.title("Silhoutte scores with different number of clusters")
plt.bar(x=k_clusters,height=silhoutte_Scores)
plt.xlabel('K clusters')
plt.ylabel('silhouttee score')
plt.xticks(k_clusters)
plt.show()

