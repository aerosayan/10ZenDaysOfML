# LANG : Python 3.5
# FILE : 02-multi-linear-regression.py
# AUTH : Sayan Bhattacharjee
# EMAIL: aero.sayan@gmail.com
# DATE : 27/JULY/2018
# INFO : Multiple regression on diabetes dataset taken from sci-kit learn
from __future__ import print_function
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn import datasets

def linear_model(x,y,echo=False):
    """ Linear model for multi-linear regression """
    # Standard NumPy mehtod if np.matrix is not used
    #beta_hat = np.dot(np.dot(inv(np.dot(x.T,x)),x.T),y)
    # Standard MATLAB like syntax that can be used if np.matrix is used
    beta_xhat = inv(x.T * x)* x.T * y
    # Conditional printing
    if echo:
    #    print("DBG : beta_hat  : \n", beta_hat)
        print("DBG : beta_xhat : \n",beta_xhat)
        print("\n")
    return beta_xhat

if __name__ == "__main__":
    # Run some tests on matrix multiplication to demonstrate np.matrix's importance
    """
    print("INF : Test 1 \n----------------------")
    linear_model(np.matrix(np.arange(9).reshape(3,3)),np.matrix(np.eye(3)),echo=True )
    print("INF : Test 2 \n----------------------")
    linear_model(np.arange(9).reshape(3,3), np.eye(3),echo=True )
    print("----------------------")
    """
    # We load up data of the diabetes set for performing multiple regression
    diabetes = datasets.load_diabetes();
    # We specify which features we want
    indices  = (0,1)
    # We load the features for training and convert them to np.matrix
    xtrain  = np.matrix(diabetes.data[:-20, indices])
    # We add the bias column vector
    xtrain  = np.column_stack( ((np.ones(len(xtrain))),xtrain))
    # We load the targets for training and convert them to a np.matrix in column form
    ytrain  = np.matrix(diabetes.target[:-20]).T

    # We load the features for testing and convert them to np.matrix
    xtest   = np.matrix(diabetes.data[-20:, indices])
    # We add the bias column vector
    xtest   = np.column_stack( ((np.ones(len(xtest))),xtest))
    # We load the targets for testing
    ytest   = np.matrix(diabetes.target[-20:]).T

    # We print the shapes of our matrices to ensure our maths is correct
    print("The Multiple Regression has resulted in...")
    print("xtrain.shape \t\t:\t",xtrain.shape)
    print("ytrain.shape \t\t:\t",ytrain.shape)
    print("xtest.shape \t\t:\t" ,xtest.shape)
    print("ytest.shape \t\t:\t" ,ytest.shape)
    #print("xtrain : \n",xtrain)
    #print("ytrain : \n",ytrain)

    # we train the model and get the bias vector back
    beta_hat = linear_model(xtrain,ytrain)
    print("beta_hat.shape \t\t:\t",beta_hat.shape)
    #print("beta_hat : \n",beta_hat)


    # We plot the figures
    fig = plt.figure()
    # We activate 3d plotting
    ax = plt.axes(projection='3d')
    # We plot the training dataset
    ax.scatter3D(xtrain[:,1],xtrain[:,2],ytrain,c='k',marker='+')
    # We plot the test predictions
    ax.scatter3D(xtest[:,1],xtest[:,2],xtest*beta_hat,c='r',marker='o',s=40)
    # We plot the original test target to compare with prediction
    ax.scatter3D(xtest[:,1],xtest[:,2],ytest,c='b',marker='v',s=80)
    # We beautify the plot
    ax.set_title('Multiple regression')
    ax.set_xlabel('X_1')
    ax.set_ylabel('X_2')
    ax.set_zlabel('Y')
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    plt.show()
