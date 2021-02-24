
#!/usr/bin/python

import sys
#Your code here
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def loadData(fileDj):
    #Your code here
    data=pd.read_csv(fileDj,sep=" ",header=None)
    data=pd.DataFrame.as_matrix(data)
    return data

## K-means functions 

def getInitialCentroids(X, k):
    initialCentroids = {}
    #Your code here
    for i in range(k):
      ind=np.random.randint(0,len(X)-1)
      cx1=X[ind,0]
      cx2=X[ind,1]
      initcen=(cx1,cx2)
      initialCentroids[i+1]=[initcen]
    return initialCentroids

def getDistance(pt1,pt2):
    dist = 0
    #Your code here
    dist=np.linalg.norm(np.asarray(pt1)-np.asarray(pt2))
    return dist

def allocatePoints(X,clusters):
    #Your code here
    points=[]
    for i in range(len(X)):
      x1=X[i,0]
      x2=X[i,1]
      points.append((x1,x2))
    center=list(clusters.values())
    center_label=list(clusters.keys())
    for i in range(len(center)):
      clusters.update({i+1:[]})
    for point in points:
      distance=[]
      for i in range(len(center_label)):
        dist=getDistance(point,center[i])
        distance.append(dist)
      result=np.argmin(distance) 
      label=center_label[result]
      clusters[label].append(point)
    return clusters

def updateCentroids(clusters):
    #Your code here
    lists=[]
    for i in range(len(clusters.keys())):
      mylist=clusters[i+1]
      lists.append(mylist)
    points=[]
    for i in range(len(clusters.keys())):
      points=points+clusters[i+1]
    center=[]
    for m in range(len(lists)):
      centr=tuple(np.mean(lists[m],axis=0))
      center.append(centr)
    for i in range(len(center)):
      clusters.update({i+1:[]})
    for point in points:
      distance=[]
      for i in range(len(center)):
        dist=getDistance(point,center[i])
        distance.append(dist)
      label=np.argmin(distance)+1
      clusters[label].append(point)
    return clusters

def visualizeClusters(clusters):
    #Your code here
    labels=list(clusters.keys())
    colors=['red','blue','green','yellow','black','orange']
    for i in range(len(labels)):
      mylist=clusters[labels[i]]
      plt.scatter(x=np.asarray(mylist)[:,0],y=np.asarray(mylist)[:,1],color=colors[i])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    
def getCentroids(clusters):
    centroids=[]
    for item in clusters:
      centroids.append(tuple(np.mean(clusters[item],axis=0)))
    return centroids

def kmeans(X, k, maxIter=1000):
    clusters = getInitialCentroids(X,k)
    clusters = allocatePoints(X,clusters)
    clusters = updateCentroids(clusters)
    centroids=getCentroids(clusters)
    for i in range(maxIter):
      new_clusters=updateCentroids(clusters)
      new_centroids=getCentroids(clusters)
      if new_centroids==centroids:
        clusters=new_clusters
        break
      else:
        centroids=new_centroids
    return clusters


def kneeFinding(X,kList):
    #Your code here
    losses=[]
    for k in kList:
      loss=0
      clusters = kmeans(X, k, maxIter=1000)
      labels=list(clusters.keys())
      for i in range(len(labels)):
        centroid=tuple(np.mean(clusters[labels[i]],axis=0))
        mylist=clusters[labels[i]]
        for point in mylist:
          dist=getDistance(point,centroid)
          dist2=dist**2
          loss+=dist2
      losses.append(loss)
      ks=[1,2,3,4,5,6]
    plt.plot(ks,losses)
    plt.show()
          

def purity(X, clusters):
    purities = []
    #Your code here
    labels=list(clusters.keys())
    df1=pd.DataFrame(np.asarray(clusters[1]))
    df1['label']=1
    df2=pd.DataFrame(np.asarray(clusters[2]))
    df2['label']=2
    df3=df1.append(df2)
    df3.columns=['x','y','label']
    df3=df3.sort_values(by=['x','y'])
    df3=df3.reset_index()
    df3['index']=df3.index
    df4=pd.DataFrame(X)
    df4.columns=['x','y','class']
    df4=df4.sort_values(by=['x','y'])
    df4=df4.reset_index()
    df4['index']=df4.index
    df5=pd.merge(df3,df4,how='inner',on=['index'])
    for m in range(len(labels)):
      ct1=0
      ct2=0
      for i in range(len(df5)):
        if df5['label'][i]==df5['class'][i]==m+1:
          ct1+=1
        else:
          ct2+=1
      purities.append(max(ct1,ct2)/(ct1+ct2))
    return purities


def main():
    #######dataset path
    datadir = sys.argv[1]
    #datadir='C:/Users/annah/Documents/Python/cs4501-001'
    pathDataset1 = datadir+'/humanData.txt'
    pathDataset2 = datadir+'/audioData.txt'
    dataset1 = loadData(pathDataset1)
    dataset2 = loadData(pathDataset2)

    #Q4
    kneeFinding(dataset1,range(1,7))

    #Q5
    clusters = kmeans(dataset1, 2, maxIter=1000)
    purity(dataset1,clusters)


if __name__ == "__main__":
    main()