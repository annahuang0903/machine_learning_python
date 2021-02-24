# Machine Learning HW2
# First Programming Task: Polynomial Regression

__author__ = 'Yehan Huang'

import numpy as np
import matplotlib.pylab as plt
# more imports 

def loadDataSet(filename):
    data=np.loadtxt(filename)
    x=data[:,0:2]
    y=data[:,2] 
    #plot data
    plt.scatter(x[:,1],y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    return x, y

def polyRegresTrain(X_train, Y_train, d):
    """Given the training data, learn a polynomial regression
    model of degree d.
    Output: a (d + 1)x1 vector containing the learned coefficients for the regression model.
    I.e., y_predicted = theta0 + theta1 * x + theta2 * x^2 + ... + thetad * x^d
    """
    if d==0:
      xVal=X_train[:,0]
      theta=np.mean(Y_train)
      MSETrainLoss=np.mean((theta-Y_train)**2)
      x_bin = np.linspace(0, 6.5, 2)
      y_val=[theta,theta]
      #plot line
      plt.plot(x_bin, y_val,color="red") 
      plt.scatter(X_train[:,1],Y_train)
      plt.xlabel('x')
      plt.ylabel('y')
      plt.show()
      return theta, MSETrainLoss
    elif d==1:
      xVal=X_train
      x_t=np.transpose(xVal) #transpose of x
      x_t_x=np.dot(x_t,xVal)   
      inv=np.linalg.inv(x_t_x) #inverse
      x_tt=np.dot(inv,x_t)
      theta=np.dot(x_tt,Y_train) #theta calculated using normal equation
      Y_predict=np.dot(xVal,theta)
      MSETrainLoss=np.mean((Y_predict-Y_train)**2)
      line=theta[0]+theta[1]*xVal
      #plot line
      plt.plot(xVal,line,color="red")    
      plt.scatter(xVal[:,1],Y_train)
      plt.xlabel('x')
      plt.ylabel('y')
      plt.show()
      return theta, MSETrainLoss
    else:
      xd=[]
      for i in range(d-1):
        xi=X_train[:,1]**(i+2)
        xd.append(xi)
      xdarray=np.asarray(xd)
      xdmatrix=np.asmatrix(xdarray)
      xdt=xdmatrix.transpose()
      xVal=np.hstack((X_train,xdt)) #create new x matrix of higher orders
      x_t=np.transpose(xVal) #transpose of x
      x_t_x=np.dot(x_t,xVal)   
      inv=np.linalg.inv(x_t_x) #inverse
      x_tt=np.dot(inv,x_t)
      theta=np.dot(x_tt,Y_train) #theta calculated using normal equation
      Y_predict=np.asarray(np.transpose(np.dot(xVal,np.transpose(theta)))).flatten()    
      MSETrainLoss=np.mean((Y_predict-Y_train)**2)
      #plot line of fit
      x_bin = np.linspace(0, 6.5, 1000)
      y_val=np.zeros(len(x_bin))
      for i in range(len(x_bin)):
        y=theta[0,0]
        for j in range(d):
          y+=theta[0,j+1]*(x_bin[i]**(j+1))
        y_val[i]=y
      plt.plot(x_bin, y_val,color="red") 
      plt.scatter(X_train[:,1],Y_train)
      plt.xlabel('x')
      plt.ylabel('y')
      plt.show()
      return theta, MSETrainLoss


def polyRegresTest(X_test, Y_test, degree, ptheta):
    if degree==0:
      xVal=X_test[:,0]
      theta=np.mean(Y_test)
      MSETestLoss=np.mean((theta-Y_test)**2)
      return MSETestLoss
    elif degree==1:
      xVal=X_test
      Y_predict=np.dot(xVal,ptheta)
      MSETestLoss=np.mean((Y_predict-Y_test)**2)
      return MSETestLoss
    else:
      xd=[]
      for i in range(degree-1):
        xi=X_test[:,1]**(i+2)
        xd.append(xi)
      xdarray=np.asarray(xd)
      xdmatrix=np.asmatrix(xdarray)
      xdt=xdmatrix.transpose()
      xVal=np.hstack((X_test,xdt))    
      Y_predict=np.asarray(np.transpose(np.dot(xVal,np.transpose(ptheta)))).flatten()    
      MSETestLoss=np.mean((Y_predict-Y_test)**2)
      return MSETestLoss


def trainAndValidate(X_train, Y_train, X_test, Y_test):
    """Iteratively call polyRegresTrain and polyRegresTest to find the best polynomial
    model for the dataset.
    Display the following plots:
        (1) Training MSE vs degree
        (2) Testing MSE vs degree
    Return pthetaBest, a vector containing the coefficients for the best polynomial model
    """
    # your code
    degrees = [0,1,2,3,4,5,6,7,8]
    Train_loss=[]
    Test_loss=[]
    for i in degrees:
      ptheta, MSETrainLoss = polyRegresTrain(X_train, Y_train, d=degrees[i])
      MSETestLoss = polyRegresTest(X_test, Y_test, degree=degrees[i], ptheta=ptheta)
      Train_loss.append(MSETrainLoss)
      Test_loss.append(MSETestLoss)
    #plot loss
    plt.plot(degrees, Train_loss)
    plt.xlabel('degree')
    plt.ylabel('MSETrainLoss')
    plt.show()
    plt.plot(degrees, Test_loss)
    plt.xlabel('degree')
    plt.ylabel('MSETestLoss')
    plt.show()
    #determine pthetaBest
    d=degrees[np.argmin(Test_loss)]
    pthetaBest, MSETrainLoss=polyRegresTrain(X_train, Y_train, d)
    #plot line of fit on validation
    x_bin = np.linspace(0, 6.5, 1000)
    y_val=np.zeros(len(x_bin))
    for i in range(len(x_bin)):
      y=pthetaBest[0,0]
      for j in range(d):
        y+=pthetaBest[0,j+1]*(x_bin[i]**(j+1))
      y_val[i]=y
    plt.plot(x_bin, y_val,color="red") 
    plt.scatter(X_test[:,1],Y_test)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    return pthetaBest


if __name__ == "__main__":
    ### Function 1: train polynomial regression model
    X_train, Y_train = loadDataSet('polyRegress_train.txt')
    degree = 3 # experiment with this value
    ptheta, MSETrainLoss = polyRegresTrain(X_train, Y_train, d=degree)

    ### Function 2: evaluate the polynomial model on the validation set
    X_test, Y_test = loadDataSet('polyRegress_validation.txt')
    MSETestLoss = polyRegresTest(X_test, Y_test, degree=degree, ptheta=ptheta)

    ### Function 3: use the previous two methods to find a good regression model
    pthetaBest = trainAndValidate(X_train, Y_train, X_test, Y_test)
