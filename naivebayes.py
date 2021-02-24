#!/usr/bin/python

import sys
import os
import numpy as np
import pandas as pd
import math
import glob
from sklearn.naive_bayes import MultinomialNB

###############################################################################


def transfer(fileDj, vocabulary):
  file=open(fileDj,"r").readlines()
  for line in file:
    for word in line.strip().split():
      word.replace("loving","love")
      word.replace("loved","love")
      word.replace("loves","love")
      if word in vocabulary:
        vocabulary[word] += 1
      else:
        vocabulary['UNK'] += 1
  BOWDj=[vocabulary['love'],vocabulary['wonderful'],vocabulary['best'],vocabulary['great'],
         vocabulary['superb'],vocabulary['still'],vocabulary['beautiful'],vocabulary['bad'],
         vocabulary['worst'],vocabulary['stupid'],vocabulary['waste'],vocabulary['boring'],
         vocabulary['?'],vocabulary['!'],vocabulary['UNK']]
  return BOWDj


def loadData(Path):
  Xtrain=[]
  Xtest=[]
  ytrain=[]
  ytest=[]
  train_pos=os.listdir(Path+"/training_set/pos")
  train_neg=os.listdir(Path+"/training_set/neg")
  test_pos=os.listdir(Path+"/test_set/pos")
  test_neg=os.listdir(Path+"/test_set/neg")
  for file in train_pos:
    file_name=Path+"/training_set/pos/"+file
    ytrain.append("pos")
    vocabulary={'love':0,'wonderful':0,'best':0,'great':0,'superb':0,'still':0,
                  'beautiful':0,'bad':0,'worst':0,'stupid':0,'waste':0,'boring':0,
                  '?':0,'!':0,'UNK':0}
    BOWDj=transfer(file_name, vocabulary)
    Xtrain.append(BOWDj)
  for file in train_neg:
    file_name=Path+"/training_set/neg/"+file
    ytrain.append("neg")
    vocabulary={'love':0,'wonderful':0,'best':0,'great':0,'superb':0,'still':0,
                  'beautiful':0,'bad':0,'worst':0,'stupid':0,'waste':0,'boring':0,
                  '?':0,'!':0,'UNK':0}
    BOWDj=transfer(file_name, vocabulary)
    Xtrain.append(BOWDj)
  for file in test_pos:
    file_name=Path+"/test_set/pos/"+file
    ytest.append("pos")
    vocabulary={'love':0,'wonderful':0,'best':0,'great':0,'superb':0,'still':0,
                  'beautiful':0,'bad':0,'worst':0,'stupid':0,'waste':0,'boring':0,
                  '?':0,'!':0,'UNK':0}
    BOWDj=transfer(file_name, vocabulary)
    Xtest.append(BOWDj)
  for file in test_neg:
    file_name=Path+"/test_set/neg/"+file
    ytest.append("neg")
    vocabulary={'love':0,'wonderful':0,'best':0,'great':0,'superb':0,'still':0,
                  'beautiful':0,'bad':0,'worst':0,'stupid':0,'waste':0,'boring':0,
                  '?':0,'!':0,'UNK':0}
    BOWDj=transfer(file_name, vocabulary)
    Xtest.append(BOWDj)
  return Xtrain, Xtest, ytrain, ytest


def naiveBayesMulFeature_train(Xtrain, ytrain):
  features=['love','wonderful','best','great','superb','still','beautiful','bad','worst',
            'stupid','waste','boring','?','!','UNK']
  Xtrain=pd.DataFrame(Xtrain,columns=features)
  ytrain=pd.DataFrame(ytrain,columns=['class'])
  train=pd.concat([Xtrain,ytrain],axis=1)
  wordi_p=train.loc[train['class']=='pos','love':'UNK'].sum()
  wordi_n=train.loc[train['class']=='neg','love':'UNK'].sum()
  word_pos = sum(wordi_p)
  word_neg = sum(wordi_n)
  alpha=1
  thetaPos=[]
  thetaNeg=[]
  v=15    
  for i in range(v):
    theta=(wordi_p[i]+alpha)/(word_pos+alpha*v)
    thetaPos.append(theta)
  for j in range(v):
    theta=(wordi_n[j]+alpha)/(word_neg+alpha*v)
    thetaNeg.append(theta)
  return thetaPos, thetaNeg


def naiveBayesMulFeature_test(Xtest, ytest,thetaPos, thetaNeg):
  logthetaPos=[math.log(i) for i in thetaPos]
  logthetaNeg=[math.log(i) for i in thetaNeg]
  yPredict=[]
  ct=0 
  for i in range(len(Xtest)):
    p_pos=math.log(0.5)
    p_neg=math.log(0.5)
    p_pos+=np.array(Xtest[i]).dot(logthetaPos)
    p_neg+=np.array(Xtest[i]).dot(logthetaNeg)
    if p_pos>p_neg:
      classlabel="pos"
    else:
      classlabel="neg"
    yPredict.append(classlabel)
    if classlabel==ytest[i]:
      ct+=1 
  Accuracy=ct/len(ytest)
  return yPredict, Accuracy


def naiveBayesMulFeature_sk_MNBC(Xtrain, ytrain, Xtest, ytest):
  ct=0
  clf=MultinomialNB()
  clf.fit(Xtrain,ytrain)
  yPredict=clf.predict(Xtest)
  for i in range(len(ytest)):
    if yPredict[i]==ytest[i]:
      ct+=1
  Accuracy=ct/len(ytest)
  return Accuracy



#def naiveBayesMulFeature_testDirectOne(path,thetaPos, thetaNeg):
#   return yPredict


def naiveBayesMulFeature_testDirect(path,thetaPos, thetaNeg):
    yPredict = []

    return yPredict, Accuracy



def naiveBayesBernFeature_train(Xtrain, ytrain):
  features=['love','wonderful','best','great','superb','still','beautiful','bad','worst',
            'stupid','waste','boring','?','!','UNK']
  Xtrain=pd.DataFrame(Xtrain,columns=features)
  Xtrain=Xtrain.astype(bool)
  ytrain=pd.DataFrame(ytrain,columns=['class'])
  train=pd.concat([Xtrain,ytrain],axis=1)
  pos=train[train['class']=='pos']
  neg=train[train['class']=='neg']
  pos_true=pos.sum(axis=0)[:-1]
  neg_true=neg.sum(axis=0)[:-1]
  thetaPosTrue=[]
  thetaNegTrue=[]
  for x in pos_true:
    thetaPosTrue.append((x+1)/(len(pos)+2))
  for x in neg_true:
    thetaNegTrue.append((x+1)/(len(neg)+2))
  return thetaPosTrue, thetaNegTrue

    
def naiveBayesBernFeature_test(Xtest, ytest, thetaPosTrue, thetaNegTrue):
  yPredict=[]
  Xtest=np.array(Xtest).astype(bool)
  logthetaPos=[math.log(i) for i in thetaPos]
  logthetaNeg=[math.log(i) for i in thetaNeg]
  ct=0 
  for i in range(len(Xtest)):
    p_pos=math.log(0.5)
    p_neg=math.log(0.5)
    p_pos+=Xtest[i].dot(logthetaPos)
    p_neg+=Xtest[i].dot(logthetaNeg)
    if p_pos>p_neg:
      classlabel="pos"
    else:
      classlabel="neg"
    yPredict.append(classlabel)
    if classlabel==ytest[i]:
      ct+=1 
  Accuracy=ct/len(ytest)
  return yPredict, Accuracy


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python naiveBayes.py dataSetPath testSetPath")
        sys.exit()

    print("--------------------")
    textDataSetsDirectoryFullPath = sys.argv[1]
    testFileDirectoryFullPath = sys.argv[2]


    Xtrain, Xtest, ytrain, ytest = loadData(textDataSetsDirectoryFullPath)


    thetaPos, thetaNeg = naiveBayesMulFeature_train(Xtrain, ytrain)
    print("thetaPos =", thetaPos)
    print("thetaNeg =", thetaNeg)
    print("--------------------")

    yPredict, Accuracy = naiveBayesMulFeature_test(Xtest, ytest, thetaPos, thetaNeg)
    print("MNBC classification accuracy ="+str(Accuracy))

    Accuracy_sk = naiveBayesMulFeature_sk_MNBC(Xtrain, ytrain, Xtest, ytest)
    print("Sklearn MultinomialNB accuracy ="+str(Accuracy_sk))

 #   yPredict, Accuracy = naiveBayesMulFeature_testDirect(testFileDirectoryFullPath, thetaPos, thetaNeg)
 #   print "Directly MNBC tesing accuracy =", Accuracy
 #   print "--------------------"

    thetaPosTrue, thetaNegTrue = naiveBayesBernFeature_train(Xtrain, ytrain)
    print("thetaPosTrue =", thetaPosTrue)
    print ("thetaNegTrue =", thetaNegTrue)
    print ("--------------------")

    yPredict, Accuracy = naiveBayesBernFeature_test(Xtest, ytest, thetaPosTrue, thetaNegTrue)
    print("BNBC classification accuracy =", Accuracy)
    print("--------------------")


