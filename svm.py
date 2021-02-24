# Starting code for UVA CS 4501 ML- SVM

import numpy as np
from sklearn.svm import SVC
import random
import pandas as pd
from sklearn.preprocessing import scale

def fold(x,y, currenti, kfold):
    foldSize = len(y)//kfold
    start = currenti*foldSize
    end = start + foldSize
    testI  = list(range(start, end))
    trainI = list(range(0, start)) + list(range(end, len(y)))
    trainingx = x[trainI]
    trainingy=y[trainI]
    testingx = x[testI]
    testingy = y[testI]
    return (trainingx, trainingy, testingx, testingy)

# Attention: You're not allowed to use the model_selection module in sklearn.
#            You're expected to implement it with your own code.
# from sklearn.model_selection import GridSearchCV

class SvmIncomeClassifier:
    def __init__(self):
        random.seed(0)

    def load_data(self, csv_fpath):
        # 1. Data pre-processing.
        # Hint: Feel free to use some existing libraries for easier data pre-processing.
        data=pd.read_csv(csv_fpath,sep=',',header=None, names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss','hours-per-week', 'native-country','label'])
        #replace missing values with the mode of that column
        df1=data.copy()
        df1['workclass']=df1['workclass'].replace([' ?'],df1['workclass'].mode())
        df1['education']=df1['education'].replace([' ?'],df1['education'].mode())
        df1['marital-status']=df1['marital-status'].replace([' ?'],df1['marital-status'].mode())
        df1['occupation']=df1['occupation'].replace([' ?'],df1['occupation'].mode())
        df1['relationship']=df1['relationship'].replace([' ?'],df1['relationship'].mode())
        df1['race']=df1['race'].replace([' ?'],df1['race'].mode())
        df1['sex']=df1['sex'].replace([' ?'],df1['sex'].mode())
        df1['native-country']=df1['native-country'].replace([' ?'],df1['native-country'].mode())
        df1['native-country']=df1['native-country'].replace([' Holand-Netherlands'],df1['native-country'].mode())
        #create dummy variables
        df2=pd.get_dummies(df1,columns=['workclass','education', 'marital-status', 'occupation', 'relationship','race', 'sex', 'native-country'],prefix=['workclass','education', 'marital-status', 'occupation', 'relationship','race', 'sex', 'native-country'])
        df2['label']=np.where(df2['label'].str.contains('<='), 0, 1)
        df3=np.asarray(df2)
        np.random.shuffle(df3)
        X=np.delete(df3,6,axis=1)
        x=scale(X, axis=0, with_mean=True, with_std=True, copy=True )
        y=df3[:,6]
        pass
        return x, y   

    def train_and_select_model(self, training_csv):
        x, y = self.load_data(training_csv)
        # 2. Select the best model with cross validation.
        # Attention: Write your own hyper-parameter candidates.
        kfold=3
        train_acc=[]
        test_acc=[]
        models=[]
        c_record=[]
        g_record=[]
        d_record=[]
        #basic linear kernel 
        lin_train_sum = 0 
        lin_test_sum = 0 
        for i in range(kfold):
          x_train,y_train,x_test,y_test=fold(x,y,i,kfold)
          linear=SVC(kernel='linear',C=1.0)
          linear.fit(x_train,y_train)
          predicty_train=linear.predict(x_train)
          acc_train=1-np.sum(np.absolute(predicty_train-y_train))/len(y_train)
          lin_train_sum+=acc_train
          predicty_test=linear.predict(x_test)
          acc_test=1-np.sum(np.absolute(predicty_test-y_test))/len(y_test)
          lin_test_sum+=acc_test  
        train_acc_lin=lin_train_sum/kfold
        test_acc_lin=lin_test_sum/kfold
        train_acc.append(train_acc_lin)
        test_acc.append(test_acc_lin)
        models.append('linear')
        c_record.append(1)
        g_record.append(0)
        d_record.append(0)
        print("linear kernel CV train accuracy is "+str(train_acc_lin)+", C=1")
        print("linear kernel CV test accuracy is "+str(test_acc_lin)+", C=1")
        #RBF kernel
        cs=[0.1,1,10]
        gammas=[0.01,0.1,1]
        for c in range(len(cs)):
          for g in range(len(gammas)):
            rbf_train_sum = 0 
            rbf_test_sum = 0 
            for k in range(kfold):
              x_train,y_train,x_test,y_test=fold(x,y,k,kfold)
              rbf=SVC(kernel='rbf',C=cs[c],gamma=gammas[g])
              rbf.fit(x_train,y_train)
              predicty_train=rbf.predict(x_train)
              acc_train=1-np.sum(np.absolute(predicty_train-y_train))/len(y_train)
              rbf_train_sum+=acc_train
              predicty_test=rbf.predict(x_test)
              acc_test=1-np.sum(np.absolute(predicty_test-y_test))/len(y_test)
              rbf_test_sum+=acc_test  
            train_acc_rbf=rbf_train_sum/kfold
            test_acc_rbf=rbf_test_sum/kfold
            train_acc.append(train_acc_rbf)
            test_acc.append(test_acc_rbf)
            models.append('rbf')
            c_record.append(cs[c])
            g_record.append(gammas[g])
            d_record.append(0)
            print("rbf kernel CV train accuracy is "+str(train_acc_rbf)+", C="+str(cs[c])+", gamma="+str(gammas[g]))
            print("rbf kernel CV test accuracy is "+str(test_acc_rbf)+", C="+str(cs[c])+", gamma="+str(gammas[g]))
        #polynomial kernel
        degrees=[3,5,7]
        for d in range(len(degrees)):
              poly_train_sum = 0 
              poly_test_sum = 0 
              for k in range(kfold):
                x_train,y_train,x_test,y_test=fold(x,y,k,kfold)
                poly=SVC(kernel='poly',C=1,gamma=0.01,degree=degrees[d])
                poly.fit(x_train,y_train)
                predicty_train=poly.predict(x_train)
                acc_train=1-np.sum(np.absolute(predicty_train-y_train))/len(y_train)
                poly_train_sum+=acc_train
                predicty_test=poly.predict(x_test)
                acc_test=1-np.sum(np.absolute(predicty_test-y_test))/len(y_test)
                poly_test_sum+=acc_test  
              train_acc_poly=poly_train_sum/kfold
              test_acc_poly=poly_test_sum/kfold
              train_acc.append(train_acc_poly)
              test_acc.append(test_acc_poly)
              models.append('poly')
              c_record.append(1)
              g_record.append(0.01)
              d_record.append(degrees[d])
              print("poly kernel CV train accuracy is "+str(train_acc_poly)+", C=1, gamma=0.01, degree="+str(degrees[d]))
              print("poly kernel CV test accuracy is "+str(test_acc_poly)+", C=1, gamma=0.01, degree="+str(degrees[d]))
        #select best model based on test accuracy
        model_best=models[np.argmin(test_acc)]
        c_best=c_record[np.argmin(test_acc)]
        gamma_best=g_record[np.argmin(test_acc)]
        d_best=d_record[np.argmin(test_acc)]
        best_model=SVC(kernel=model_best,C=c_best,gamma=gamma_best,d=d_best)
        best_model.fit(x,y)
        best_score=np.min(test_acc)
        pass
        return best_model, best_score

    def predict(self, test_csv, trained_model):
        x_test, _ = self.load_data(test_csv)
        predictions = trained_model.predict(x_test)
        return predictions

    def output_results(self, predictions):
        # 3. Upload your Python code, the predictions.txt as well as a report to Collab.
        # Hint: Don't archive the files or change the file names for the automated grading.
        with open('predictions.txt', 'w') as f:
            for pred in predictions:
                if pred == 0:
                    f.write('<=50K\n')
                else:
                    f.write('>50K\n')

if __name__ == '__main__':
    training_csv = "salary.labeled.csv"
    testing_csv = "salary.2Predict.csv"
    clf = SvmIncomeClassifier()
    trained_model, cv_score = clf.train_and_select_model(training_csv)
    print("The best model was scored %.2f" % cv_score)
    predictions = clf.predict(testing_csv, trained_model)
    clf.output_results(predictions)


