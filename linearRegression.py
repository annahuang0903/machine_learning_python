
# Machine Learning HW1
# Yehan Huang yh5sc

import matplotlib.pyplot as plt
import numpy as np

def loadDataSet(filename):
    data=np.loadtxt("Q2data.txt")
    x=data[:,0:2]
    y=data[:,2] 
    plt.scatter(x[:,1],y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    return x, y

def standRegresOpt1(xVal, yVal):
    x_t=np.transpose(xVal)
    x_t_x=np.dot(x_t,xVal)   
    inv=np.linalg.inv(x_t_x)
    x_tt=np.dot(inv,x_t)
    theta=np.dot(x_tt,yVal)
    line=theta[0]+theta[1]*xVal
    plt.plot(xVal,line)    
    plt.scatter(xVal[:,1],yVal)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    return theta
  
def standRegresOpt2(xVal, yVal):
    alpha=0.0001
    epoch=np.arange(0,10000)
    loss=np.zeros(10000,float)
    theta=np.zeros(2,float)
    x_t=np.transpose(xVal)
    for i in range(len(epoch)):
      x_theta=np.dot(xVal,theta)
      diff=yVal-x_theta
      loss[i]=sum(diff**2)/len(yVal)
      prod=np.dot(x_t,diff)
      theta=theta+alpha*prod
    line=theta[0]+theta[1]*xVal
    plt.plot(xVal,line)    
    plt.scatter(xVal[:,1],yVal)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    plt.plot(loss)
    plt.xlabel('epoch')
    plt.ylabel('training loss')
    plt.show()
    return theta

xVal, yVal = loadDataSet('Q2data.txt')

theta = standRegresOpt1(xVal, yVal)

theta2 = standRegresOpt2(xVal, yVal)

## If you implement one more optimizatoin 
#theta3 = standRegresOpt3(xVal, yVal)
