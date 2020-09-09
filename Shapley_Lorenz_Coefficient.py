'''
Calculates the Shapley Lorenz marginal contribution, based on the paper:
Shapley-Lorenz decompositions in eXplainable Artificial Intelleigence
by Paolo Giudici* and Emanuela Raffinetti**
*University of Pavia
**Università degli Studi di Milano
'''

# Packages to import
import numpy as np
import itertools
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import datasets
from scipy.special import binom,\
                        factorial

class ShapleyLorenzShare():
    '''
    Uses the Shapley approach to calculate Shapley Lorenz share coefficients

    Parameters:
    ---------------------------------------------------------
    model : method
        specifies the prediction model

    '''
    def __init__(self, model):
        self.model = model

# Combinatoric tool
    def powerset(self, iterable):
            s = list(iterable)
            return itertools.chain.from_iterable(itertools.combinations(s,r)\
                for r in range(len(s)+1))
            # s : iterable
            # r : length

    # Shapley Kernel
    def ShapleyKernel(self, M, s):
        return factorial(s)*factorial(M-s-1)/factorial(M)

    def ShapleyLorenz_val(self, X, y, M = None):

        '''
        Computes the Shapley Lorenz marginal contribution of
        a covariate X_k.

        Parameters:
        ---------------------------------------------------------
        X : matrix
            nxp matrix containing the model covariates
        y : vector
            n-vector containing the values to predict
        M : int (DEFAULT: M = X.shape[1])
            number of covariates to calculate the Shapley Lorenz
            marginal contributions for
        '''
        if M == None:
            M = X.shape[1]

        #LZ_temp = np.zeros((2**(M-1),1))
        LZ = np.zeros((M,1))

        # Loop over number of covariates
        for k in range(M):
            # Initialise
            V_base = np.zeros((X.shape[0],M,2**(M-1)))
            V_k = np.zeros((X.shape[0],M,2**(M-1)))
            y_base = np.zeros((y.shape[0],2**(M-1)))
            y_k = np.zeros((y.shape[0],2**(M-1)))
            kernel = np.zeros((2**(M-1),1))

            s_all = list(range(M))
            s_base = s_all.copy()
            s_base.pop(k)
            k = [k, ]
            # loop over possible all possibel (2**(M-1)) covariate
            # combinations
            for i,s in enumerate(self.powerset(s_base)):
                s = list(s)
                s_k = k+s

                # yHat including covariate k
                V_k[:,s_k,i] = X[:,s_k]
                y_k[:,i] = self.model.fit(V_k[:,s_k,i],y)\
                    .predict(V_k[:,s_k,i]).reshape(len(y))
                y_k[:,i] = np.sort(y_k[:,i],0)

                # yHat baseline, (w/o covariate k)
                if len(s) == 0:
                    for j in range(X.shape[1]):
                        perm_indx = np.random.randint(0,X.shape[0],X.shape[0])
                        V_base[:,j,0] = X[perm_indx,j]
                        y_base[:,0] = self.model.fit(V_base[:,:,0],y)\
                            .predict(V_base[:,:,0]).reshape(len(y))
                        y_base[:,0] = np.sort(y_base[:,0],0)
                else:
                    V_base[:,s,i] = X[:,s]
                    y_base[:,i] = self.model.fit(V_base[:,s,i],y)\
                        .predict(V_base[:,s,i]).reshape(len(y))
                    y_base[:,i] = np.sort(y_base[:,i],0)

                    # Compute Kernel
                    kernel[i,0] = self.ShapleyKernel(M,len(s))

            # Compute Lorenz Zenoid values
            Lor_val_temp = np.zeros((X.shape[0],2**(M-1)))
            for j in range(len(y)):
                Lor_val_temp[j,:] = j*(y_k[j,:]-y_base[j,:])
            Lor_val = ((2/(len(y)**2))*np.mean(y))*(Lor_val_temp.sum(0))
            Lor_val = Lor_val.reshape((1,8))

            LZ[k,0] = np.dot(Lor_val,kernel)
            
        return LZ


# Tests

# Simulation data
#X = np.random.normal(0.5,4,(20,4))
#beta = np.array([[0.5,0.8,3,0.1]])
#y = 100 + np.dot(X,beta.T)+np.random.normal(0,1,(X.shape[0],1))

# Linear Regressor method
#lin_reg = LinearRegression()

#LZ_contributionsII = ShapleyLorenzShare(lin_reg).ShapleyLorenz_val(X,y)
#print(LZ_contributionsII)