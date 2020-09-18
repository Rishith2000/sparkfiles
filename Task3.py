#!/usr/bin/env python
# coding: utf-8

# Bandaru Rishith Reddy

# Task#3- To Explore Unsupervised Machine Learning

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')



# In[18]:


df=pd.read_csv('Iris.csv')

df.head()


# In[19]:


df.shape


# 

# In[20]:


df.describe()


# In[21]:


df['Species'].value_counts()


# In[32]:


x=df.iloc[:,1:5].values
x


# Finding the optimal number of clusters using elbow method

# In[33]:


from sklearn.cluster import KMeans
Wcss=[]
for i in range(1,11):
  Kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
  Kmeans.fit(x)
  Wcss.append(Kmeans.inertia_)
plt.plot(range(1,11),Wcss)
plt.title("The Elbow Method Graph")
plt.xlabel("Number of Clusters")
plt.ylabel("Wcss")
plt.show()


# From the above graph,we can see the eldow point is at 3.So,The Number of clusters here will be 3 

# In[34]:


Kmeans=KMeans(n_clusters=3,init='k-means++',random_state=0)
yp=Kmeans.fit_predict(x)
yp


# Visualizing the clusters

# In[31]:


plt.scatter(x[yp==0,0],x[yp==0,1],s=100,c='red', label = 'Iris-setosa')
plt.scatter(x[yp==1,0],x[yp==1,1],s=100,c='blue',label='Iris-versicolor')
plt.scatter(x[yp==2,0],x[yp==2,1],s=100,c='green',label='Iris-virginica ')
plt.scatter(Kmeans.cluster_centers_[:, 0],Kmeans.cluster_centers_[:,1], 
            s=100,c='yellow', label='Centroids')
plt.xlabel('SepalLength in Cm')
plt.ylabel('Sepalwidth in Cm')
plt.legend()



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




