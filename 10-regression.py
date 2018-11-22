import numpy as np
from bokeh.plotting import figure, show, output_notebook
from bokeh.layouts import gridplot
from bokeh.io import push_notebook

#output_notebook()
import numpy as np

def local_regression(x0, X, Y, tau):
    # add bias term
    x0 = np.r_[1, x0] # Add one to avoid the loss in information 
    X = np.c_[np.ones(len(X)), X]   
       
    # fit model: normal equations with kernel
    xw = X.T * radial_kernel(x0, X, tau)   # XTranspose * W
    
    beta = np.linalg.pinv(xw @ X) @ xw @ Y   # @ Matrix Multiplication or Dot Product  
        
    # predict value
    return x0 @ beta    # @ Matrix Multiplication or Dot Product for prediction 

def radial_kernel(x0, X, tau):
    return np.exp(np.sum((X - x0) ** 2, axis=1) / (-2 * tau * tau))   # Weight or Radial Kernal Bias Function

n = 1000
# generate dataset
X = np.linspace(-3, 3, num=n)
print("The Data Set ( 10 Samples) X :\n",X[1:10])
Y = np.log(np.abs(X ** 2 - 1) + .5)
print("The Fitting Curve Data Set (10 Samples) Y  :\n",Y[1:10])
# jitter X
X += np.random.normal(scale=.1, size=n)
print("Normalised (10 Samples) X :\n",X[1:10])

domain = np.linspace(-3, 3, num=300)
print(" Xo Domain Space(10 Samples)  :\n",domain[1:10])

def plot_lwr(tau):
    # prediction through regression
    prediction = [local_regression(x0, X, Y, tau) for x0 in domain]
    plot = figure(plot_width=400, plot_height=400)
    plot.title.text='tau=%g' % tau
    plot.scatter(X, Y, alpha=.3)
    plot.line(domain, prediction, line_width=2, color='red')
    return plot
# Plotting the curves with different tau
show(gridplot([
    [plot_lwr(10.), plot_lwr(1.)],
    [plot_lwr(0.1), plot_lwr(0.01)]
]))
