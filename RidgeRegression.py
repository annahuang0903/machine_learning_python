# Machine Learning HW2-Ridge

__author__ = 'Yehan Huang'

import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

def loadDataSet(filename):
    data=np.loadtxt(filename)
    x=data[:,0:3]
    y=data[:,3] 
    return x, y


def ridgeRegress(xVal, yVal, lambdaV, showFigure=True):
    x_t=np.transpose(xVal)
    x_t_x=np.dot(x_t,xVal)  
    lambdaI=np.dot(lambdaV,np.identity(len(xVal[0,:])))
    sums=x_t_x+lambdaI
    inv=np.linalg.inv(sums) 
    x_tt=np.dot(inv,x_t)
    beta=np.dot(x_tt,yVal)
    if showFigure:
      #scatter plot
      fig = plt.figure()
      ax = plt.axes(projection='3d')
      ax.scatter(xs=xVal[:,1], ys=xVal[:,2], zs=yVal)
      ax.set_xlabel('X1')
      ax.set_ylabel('X2')
      ax.set_zlabel('Y')
      #surface plot
      x = np.outer(np.linspace(-6, 6, 100), np.ones(100))
      y = x.copy().T
      z = beta[0]+beta[1]*x+beta[2]*y
      ax.plot_surface(x, y, z, cmap=plt.cm.jet, rstride=1, cstride=1, linewidth=0)
      ax.set_xlabel('X1')
      ax.set_ylabel('X2')
      ax.set_zlabel('Y')
    return beta


def cv(xVal, yVal):
    lambdas=[]
    mses=[]
    dataset=np.hstack((xVal,np.reshape(yVal,(200,1))))
    np.random.seed(37)
    np.random.shuffle(dataset)
    for i in range(1,51):
      lambdav=i*0.02
      lambdas.append(lambdav)
      mselist=[]
      for j in range(1,5):
        test=dataset[(j-1)*50:j*50,:]
        if j==1:
          train=dataset[50:,]
        elif j==2:
          train=np.vstack((dataset[:50],dataset[100:,]))
        elif j==3:
          train=np.vstack((dataset[:100],dataset[150:,]))
        else:
          train=dataset[:150,]
        trainx=train[:,0:3]
        trainy=train[:,3]
        testx=test[:,0:3]
        testy=test[:,3]
        beta=ridgeRegress(trainx,trainy,lambdaV=lambdav,showFigure=False)
        predicty=np.dot(testx,beta)
        mse=np.mean((predicty-testy)**2)
        mselist.append(mse)
      mseavg=np.mean(mselist)
      mses.append(mseavg)
    lambdaBest=lambdas[np.argmin(mses)]
    plt.plot(lambdas, mses) 
    plt.xlabel('lambda')
    plt.ylabel('testing MSE')
    plt.show()
    return lambdaBest



def standRegress(xVal, yVal):
    x_t=np.transpose(xVal) 
    x_t_x=np.dot(x_t,xVal)   
    inv=np.linalg.inv(x_t_x) 
    x_tt=np.dot(inv,x_t)
    theta=np.dot(x_tt,yVal) 
    plt.scatter(xVal[:,1], yVal)    
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show() 
    return theta


if __name__ == "__main__":
    xVal, yVal = loadDataSet('RRdata.txt')
    betaLR = ridgeRegress(xVal, yVal, lambdaV=0)
    print(betaLR)
    lambdaBest = cv(xVal, yVal)
    print(lambdaBest)
    betaRR = ridgeRegress(xVal, yVal, lambdaV=lambdaBest)
    print(betaRR)
    # depending on the data structure you use for xVal and yVal, the following line may need some change
    standRegress(xVal[:,:2],xVal[:,2])
    
    
    
