# Starting code for UVA CS 4501 Machine Learning- KNN

__author__ = 'Yehan Huang'
import numpy as np
#more importants
from sklearn.neighbors import KNeighborsClassifier
## the only purpose of the above import is in case that you want to double-check your knn result with sklearn knn
import pandas as pd
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt

#file is just a filename, this method read in file contents
# Att: there are many ways to read in one reference dataset, 
# e.g., this template reads in the whole file and put it into one numpy array. 
# (But in HW1, our template actually read the file into two numpy array, one for Xval, the other for Yval. 
# Both ways are correct.) 
def read_csv(file):
    #your code
    # please shuffle your data after the read.
    reviews = pd.read_csv(file, sep='\t')
    data=np.asarray(reviews)
    np.random.seed(37)
    np.random.shuffle(data)
    return data

#data is the full training numpy array
#currenti is the current iteration of cross validation
#kfold is the total number of cross validation folds
#we assume the data has been shuffled before using this function
# Att: there are many ways to CV 
# the following way decouples the evaluation step and data splitting 
def fold(data, currenti, kfold):
    foldSize = len(data)//kfold
    start = currenti*foldSize
    end = start + foldSize
    testI  = list(range(start, end))
    trainI = list(range(0, start)) + list(range(end, len(data)))
    training = data[trainI]
    testing = data[testI]
    return (training, testing)


#training is the numpy array of training data 
#(you run through each testing point and classify based on the training points)
#testing is a numpy array, use this method to predict 1 or 0 for each of the testing points
#k is the number of neighboring points to take into account when predicting the label
def classify(training, testing, k):
  results=[]
  for i in range(0,len(testing)):
    dists=[]
    knn=[]
    zero=0
    one=0
    for j in range(0,len(training)):
      dist=np.linalg.norm(testing[i,:-1]-training[j,:-1])
      dists.append((dist,training[j,-1]))
    dists.sort()
    knn=dists[:k]
    for m in range(0,k):
      if int(knn[m][1])==1:
        one+=1
      else:
        zero+=1
    if one>zero:
      results.append(1)
    else:
      results.append(0)
  return results

#predictions is a numpy array of 1s and 0s for the class prediction
#labels is a numpy array of 1s and 0s for the true class label
def calc_accuracy(predictions, labels):
  x=1-np.sum(np.absolute(predictions-labels))/len(labels)
  return x

def findBestK(data, kfold):
    ##--- please revise the following code to try a range of K values and return the best K 
    ##please draw a bar graph to show CV-accuracy vs. K
    ks=[3,5,7,9,11,13]
    acc=[]
    for j in range(0,len(ks)):
        sum = 0 
        for i in range(0, kfold):
            training, testing = fold(data, i, kfold)
            predictions = classify(training, testing, ks[j])
            labels = testing[:,-1]
            sum += calc_accuracy(predictions, labels)
        accuracy = sum / kfold
        print("k="+str(ks[j])+", accuracy is "+str(accuracy))
        acc.append(accuracy)
    kbest=ks[np.argmin(acc)]
    accbest=np.min(acc)
    #plot
    klabel=["3","5","7","9","11","13"]
    accseries = pd.Series.from_array(acc)
    acclabel=[str(round(acc[i],3)) for i in range(len(acc))]
    plt.figure(figsize=(6, 4))
    ax = accseries.plot(kind='bar')
    ax.set_title('Accuracy vs K values')
    ax.set_xlabel('K')
    ax.set_ylabel('Accuracy')
    ax.set_xticklabels(klabel)
    rects = ax.patches
    for rect, label in zip(rects, acclabel):
      height = rect.get_height()
      ax.text(rect.get_x() + rect.get_width() / 2, height + 0.001, label, ha='center', va='bottom')
    plt.show()
    print("kest="+str(kbest)+", accuracy is "+str(accbest))
    return kbest


if __name__ == "__main__":
    filename = "Movie_Review_Data.txt"
    data = np.asarray(read_csv(filename))
    kfold = 4
    findBestK(data, kfold)

