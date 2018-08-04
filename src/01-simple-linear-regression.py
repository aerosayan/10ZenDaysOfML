# LANG : Python 3.5
# FILE : 01-simple-linear-regression.py
# AUTH : Sayan Bhattacharjee
# EMAIL: aero.sayan@gmail.com
# DATE : 27/JULY/2018
# INFO : How does hot soup sale change in winter based on temperature?
#      : Here, we do linear regression with ordinary least squares
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
def simple_linreg(n,x,y):
    """ Perform simple linear regression
    @param n : number of data points for calculation
    @param x : the x co-ordinate of the data
    @param y : the y co-ordinate of the data
    """
    x_bar = sum(x)/float(n)                                   # average of x
    y_bar = sum(y)/float(n)                                   # average of y

    m = sum((x-x_bar)*(y-y_bar))/ float(sum((x-x_bar)**2) )   # slope
    b = y_bar - m*x_bar                                       # y intercept

    print("The linear regression has resulted in ...")
    print("m(slope)       : ",m )
    print("b(y intercept) : ",b )
    return m,b

if __name__ == "__main__":
    # We get the data using which we want to perform linear regression
    n = 100                                               # no. of data points
    temp = np.linspace(3, 30, n)                          # temperature (deg C)
    noise = np.random.randint(-2,5,size = n)              # noise to simulate RL
    soup_sale = np.linspace(40, 22 , n, dtype = 'int') + noise # soup sale count

    m,b = simple_linreg(n,temp,soup_sale)

    plt.scatter(temp,soup_sale)
    plt.plot(temp,m*temp + b,'-r')
    plt.title("Temperature vs Soup Sale"
    " \n (linear regression with ordinary least squares)")
    plt.xlabel("Temp in degree Celcius")
    plt.ylabel("Hot soup bowls sold")
    plt.grid()
    plt.show()
