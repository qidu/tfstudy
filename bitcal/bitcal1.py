import numpy as np

# sigmoid
# y = f(x)
# f(x) = 1/(1+e^-x)
# dy/dx = f'(x) = e^-x / (1+e^-x)^2 = 1/(1+e^-x) - 1/(1+e^-x)^2 = y(1-y)
# delta_y = dy * delta_x 
#
# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
    
# input dataset
X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])
    
# output dataset            
y = np.array([[0,0,1,1]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0 : matrix 3x1 elemets bewteen (-1,1)
syn0 = 2*np.random.random((3,1)) - 1

for iter in xrange(10000):
    #print syn0
    # forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0,syn0)) # [4x3] . [3x1] = [4x1], and then elemtns are normalized to (0,1)

    # how much did we miss?
    l1_error = y - l1

    # multiply how much we missed by the 
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1,True)

    # update weights : [3x4] . [4x1] = [3x1]
    syn0 += np.dot(l0.T,l1_delta)
    #print l1,"\n",l1_delta,"\n"

print "Output After Training:"
print l1
print y
