import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def kfold(x,y, currenti, kfold):
  foldSize = len(y)//kfold
  start = currenti*foldSize
  end = start + foldSize
  testI  = list(range(start, end))
  trainI = list(range(0, start)) + list(range(end, len(y)))
  trainingx = x.iloc[trainI,:].reset_index(drop=True)
  trainingy= x.iloc[trainI].reset_index(drop=True)
  testingx = x.iloc[testI,:].reset_index(drop=True)
  testingy = y.iloc[testI].reset_index(drop=True)
  return (trainingx, trainingy, testingx, testingy)

def split_data(datax, datay):
    random.seed(3)
    testind=random.sample(range(len(datax)), len(datax)//5)
    testx=datax.iloc[testind[:len(testind)//2],:].reset_index(drop=True)
    testy=datay.iloc[testind[:len(testind)//2]].reset_index(drop=True)
    valx=datax.iloc[testind[len(testind)//2:],:].reset_index(drop=True)
    valy=datay.iloc[testind[len(testind)//2:]].reset_index(drop=True)
    trainx=datax.loc[~datax.index.isin(datay.iloc[testind].index)].reset_index(drop=True)
    trainy=datay.loc[~datay.index.isin(datay.iloc[testind].index)].reset_index(drop=True)
    x=pd.concat([trainx,valx]).sample(frac=1).reset_index(drop=True)
    y=pd.concat([trainy,valy]).sample(frac=1).reset_index(drop=True)
    return trainx, trainy, valx, valy, testx, testy, x, y

def linear_reg(trainx, trainy, valx, valy):
    reg = LinearRegression(normalize=True).fit(trainx, trainy)
    pred = reg.predict(valx)
    R_sq = reg.score(valx,valy)
    feature_imp = pd.Series(reg.coef_,index=trainx.columns).sort_values(ascending=False)#
    return R_sq,feature_imp

def logistic_reg(trainx, trainy, valx, valy):
    clf = LogisticRegression(random_state=0).fit(trainx, trainy)
    pred=clf.predict(valx)
    con_mat = confusion_matrix(valy, pred)
    acc=sum(valy==pred)/len(valy)
    return con_mat, acc

def svm(trainx, trainy, valx, valy):
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(trainx, trainy)
    pred=clf.predict(valx)
    con_mat = confusion_matrix(valy, pred)
    acc=sum(valy==pred)/len(valy)
    return con_mat, acc

def random_forest(trainx, trainy, valx, valy, n=100):
    clf=RandomForestClassifier(n_estimators=n)
    clf.fit(trainx,trainy)
    pred=clf.predict(valx)
    con_mat = confusion_matrix(valy, pred)
    acc=sum(valy==pred)/len(valy)
    feature_imp = pd.Series(clf.feature_importances_,index=datax.columns).\
                  sort_values(ascending=False)
    return con_mat, acc, feature_imp

def qda(trainx, trainy, valx, valy):
    clf = QuadraticDiscriminantAnalysis()
    clf.fit(trainx,trainy)
    pred=clf.predict(valx)
    con_mat = confusion_matrix(valy, pred)
    acc=sum(valy==pred)/len(valy)
    return con_mat, acc

    

if __name__ == '__main__':
    
    cov=pd.read_csv('C:/Users/Xmy/Desktop/Project/tract_covariates.csv')

    data1=cov[cov['popdensity2000'].notnull()]
    data2=data1[data1['popdensity2010'].notnull()]
    data2['pop2000pct']=data2['popdensity2000']/sum(data2['popdensity2000'])
    data2['pop2010pct']=data2['popdensity2010']/sum(data2['popdensity2010'])
    data2['poppctchange']=data2['pop2010pct']-data2['pop2000pct']
    data2['popchange']=data2['popdensity2010']-data2['popdensity2000']
    
    data=data2.fillna(data2.mean())
    data.dropna(axis=1,inplace=True)
    datax=data.iloc[:,5:].drop(columns=['popdensity2000','popdensity2010',\
                   'pop2000pct', 'pop2010pct', 'poppctchange', 'popchange'])
    
    datay=data['popchange']
    trainx, trainy, valx, valy, testx, testy, x, y=split_data(datax,datay)
    
    #linear regression
    R_sq_lin, feature_imp=linear_reg(trainx, trainy, valx, valy) 
    print('linear regression R^2: ', R_sq_lin)#0.60
    print('linear regression feature importance: ', feature_imp)
    
    #create binary labels
    datay=data['poppctchange']
    datay=(data['poppctchange']>0).astype(int)
    trainx, trainy, valx, valy, testx, testy, x, y=split_data(datax,datay)
    
    #check for na or inf
    np.isfinite(datax).all()
    np.sum(datax.isna())
    
    #standardization
    scaler = StandardScaler()
    scaler.fit(trainx)
    trainx = scaler.transform(trainx)
    valx = scaler.transform(valx)
    testx = scaler.transform(testx)

    
    #logistic regression
    con_mat, acc_log=logistic_reg(trainx, trainy, testx, testy)
    print('logreg cf_mat: \n', con_mat)
    print('logreg accuracy: ', acc_log) #0.6671249828131445
    
    #SVM
    con_mat, acc_svm=svm(trainx, trainy, testx, testy)
    print('SVM cf_mat: \n', con_mat)
    print('SVM accuracy: ', acc_svm) #0.7124003299422601
    
    #Random Forest
    for n in [50, 100, 200, 500]:
        con_mat,acc_rand,feature=random_forest(trainx, trainy, valx, valy, n)
        print(n, acc_rand)
        #n=200 yields the best result
        
    con_mat,acc_rand,feature_imp=random_forest(trainx, trainy, testx, testy, 200)
    print('Rf cf_mat: \n', con_mat)
    print('Rf accuracy: ', acc_rand) #0.7300976213391998

    #plot feature of importance
    feature_imp = feature_imp[::-1]
    plt.figure(figsize=(10,10))
    plt.title("Feature Importances")
    plt.barh(range(testx.shape[1]), feature_imp,color="r", align="center")
    plt.yticks(range(testx.shape[1]), feature_imp.index)
    plt.show()


    #QDA
    con_mat, acc_qda=qda(trainx, trainy, testx, testy)
    print('QDA cf_mat: \n', con_mat)
    print('QDA accuracy: ', acc_qda) #0.6562628901416196




