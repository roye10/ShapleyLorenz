'''
Calculates the Shapley Lorenz marginal contribution, based on the paper:
"Shapley-Lorenz decompositions in eXplainable Artificial Intelleigence"
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

    def ShapleyLorenz_val(self, X, y, class_prob = False, M = None, row = None, pred_out = None, show_last_y = False):

        '''
        Computes the Shapley Lorenz marginal contribution of
        a covariate X_k.

        Parameters:
        ---------------------------------------------------------
        X : matrix
            nxp matrix containing the model covariates
        y : vector
            n-vector containing the values to predict
        class_prob : boolean (DEFAULT: False)
            if False --> regression problem
            if True --> classification problem
        pred_out : str (DEFAULT: 'predict')
            Need to specify if class_prob = True
            prediction output to use. Available options:
            'predict' --> and 1/0 in classification caes
            'predict_proba' --> outputs float64 class probabilities (ONLY FOR CLASSIFICATION PROBLEMS)
        M : int (DEFAULT: X.shape[1])
            number of covariates to calculate the Shapley Lorenz
            marginal contributions for
        row : int (DEFAULT: None)
            observation(s) to explain
        show_last_y : boolean (DEFAULT: False)
            if True --> y_k and y_base will be shown for last feature
            if False --> only Lorenz Zonoid values are returned
        '''

        # Transform to array
        X = np.array(X)
        y = np.array(y)

        # Conditions
        if M == None:
            M = X.shape[1]
        elif M > 10:
            raise Warning('For features larger than 10, runtime is prohibitively long, due to problem to solve becoming NP hard\
                            a value less than 10 is suggested')
        if class_prob == True and pred_out == None:
            raise ValueError('Need to specify if class_prob = True')

        if X.shape[0] == 1:
            raise ValueError('Need to specify an appropriate number of observations. n >= 100 is suggested')
        if len(X.shape) == 1:
            raise ValueError('Need to specify an appropriate number of features. p has to be >= 1')
        else:
            n = X.shape[0]

        #LZ_temp = np.zeros((2**(M-1),1))
        LZ = np.zeros((M,1))
        if pred_out == 'predict_proba' or pred_out == 'predict_log_proba':
            LZ0 = np.zeros((M,1))
            LZ1 = np.zeros((M,1))

        # Loop over number of covariates
        for k in range(M):
            # Initialise
            V_base = np.zeros((n,M,2**(M-1)))
            V_k = np.zeros((n,M,2**(M-1)))
            if class_prob == False and (class_prob == True and pred == 'predict'):
                y_base = np.zeros((n,2**(M-1)))
                y_k = np.zeros((n,2**(M-1)))
            elif class_prob == True and (pred_out == 'predict_proba' or 'predict_loga_proba'):
                y_base = np.zeros((n,2,2**(M-1)))
                y_b0 = np.zeros((n,2**(M-1)))
                y_b1 = np.zeros((n,2**(M-1)))
                y_k = np.zeros((n,2,2**(M-1)))
                y_k0 = np.zeros((n,2**(M-1)))
                y_k1 = np.zeros((n,2**(M-1)))

                val, num = np.unique(y, return_counts = True)
                y_class_mu = np.zeros(len(val))
                for i in range(val.shape[0]):
                    y_class_mu[i] = num[i]/n

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

                V_k[:,s_k,i] = X[:,s_k]
                V_base[:,s,i] = X[:,s]

                # Compute Kernel
                kernel[i,0] = self.ShapleyKernel(M,len(s))

                # yHat including covariate k
                # For specified rows
                if row != None:
                    if class_prob == False or (class_prob == True and pred_out == 'predict'):
                        if len(row) == 1:
                            y_k[:,i] = self.model.fit(V_k[:,s_k,i],y)\
                                .predict(V_k[row,s_k,i].reshape(1,-1))
                        elif len(row) > 1:
                            y_k[:,i] = self.model.fit(V_k[:,s_k,i],y)\
                                .predict(V_k[row,s_k,i]).reshape(len(row))
                            y_k[:,i] = np.sort(y_k[:,i],0)
                    elif class_prob == True:
                        if pred_out == 'predict_proba' or pred_out == 'predict_log_proba':
                            if len(row) == 1:
                                y_k[:,:,i] = getattr(self.model.fit(V_k[:,s_k,i],y)\
                                    ,pred_out)(V_k[row,s_k,i].reshape(1,-1))
                                y_k0[:,i] = y_k[:,0,i]
                                y_k1[:,i] = y_k[:,1,i]
                            elif len(row) > 1:
                                y_k[:,:,i] = getattr(self.model.fit(V_k[:,s_k,i],y)\
                                    ,pred_out)(V_k[row,s_k,i]).reshape(len(row),2)
                                y_k0[:,i] = np.sort(y_k[:,0,i],0)
                                y_k1[:,i] = np.sort(y_k[:,1,i],0)
                        else:
                            raise ValueError\
                                ('No valid method. Valid methods are: predict, predict_proba')
                    else:
                        raise ValueError('Unrecognised option')

                elif row == None:
                    if class_prob == False or (class_prob == True and pred_out == 'predict'):
                        y_k[:,i] = self.model.fit(V_k[:,s_k,i],y)\
                            .predict(V_k[:,s_k,i]).reshape(n)
                        y_k[:,i] = np.sort(y_k[:,i],0)
                    if class_prob == True:
                        if pred_out == 'predict_proba' or pred_out == 'predict_log_proba':
                            y_k[:,:,i] = getattr(self.model.fit(V_k[:,s_k,i],y)\
                                ,pred_out)(V_k[:,s_k,i]).reshape(n,2)
                            y_k0[:,i] = y_k[:,0,i]
                            y_k1[:,i] = y_k[:,1,i]

                # yHat baseline, (w/o covariate k)
                if row != None:
                    if len(s) == 0:
                        s = np.arange(M)
                        if i != 0:
                            ValueError('s is empty for i not equal 0')
                        # Shuffle rows for each feature
                        for j in range(X.shape[1]):
                                perm_indx = np.random.randint(0,n,n)
                                V_base[:,j,i] = X[perm_indx,j]
                    else:
                        V_base[:,s,i] = X[:,s]

                    if class_prob == False or (class_prob == True and pred_out == 'predict'):
                        if len(row) == 1:
                            y_base[:,i] = self.model.fit(V_base[:,s,i],y)\
                                .predict(V_base[row,:,i].reshape(1,-1))
                        elif len(row) > 1:
                            y_base[:,i] = self.model.fit(V_base[:,s,i],y)\
                                .predict(V_base[row,:,i]).reshape(len(row))
                            y_base[:,i] = np.sort(y_base[:,i],0)
                    elif class_prob == True:
                        if pred_out == 'predict_proba' or pred_out == 'predict_log_proba':
                            if len(row) == 1:
                                y_base[:,:,i] = getattr(self.model.fit(V_base[:,s,i],y)\
                                    ,pred_out)(V_k[row,:,i].reshape(1,-1))
                                y_b0[:,i] = y_base[:,0,i]
                                y_b1[:,i] = y_base[:,1,i]
                        if len(row) > 1:
                            y_base[:,:,i] = getattr(self.model.fit(V_base[:,s,i],y)\
                                    ,pred_out)(V_k[row,:,i]).reshape(len(row),2)
                            y_b0[:,i] = np.sort(y_base[:,0,i],0)
                            y_b1[:,i] = np.sort(y_base[:,1,i],0)

                else:
                    if len(s) == 0:
                        s = np.arange(M)
                        if i != 0:
                            ValueError('s is empty for i not equal 0')
                        # Shuffle rows for each feature
                        for j in range(X.shape[1]):
                                perm_indx = np.random.randint(0,n,n)
                                V_base[:,j,i] = X[perm_indx,j]
                    else:
                        V_base[:,s,i] = X[:,s]
                        
                    if class_prob == False or (class_prob == True and pred_out == 'predict'):
                        y_base[:,i] = self.model.fit(V_base[:,:,i],y)\
                            .predict(V_base[:,:,i]).reshape(n)
                        y_base[:,i] = np.sort(y_base[:,i],0)
                    elif class_prob == True:
                        if pred_out == 'predict_proba' or pred_out == 'predict_log_proba':
                            y_base[:,:,i] = getattr(self.model.fit(V_base[:,s,i],y)\
                                ,pred_out)(V_base[:,s,i]).reshape(n,2)
                            y_b0[:,i] = np.sort(y_base[:,0,i],0)
                            y_b1[:,i] = np.sort(y_base[:,1,i],0)

            # Compute Lorenz Zenoid values
            Lor_val_temp = np.zeros((n,2**(M-1)))
            Lor_val_temp0 = np.zeros((n,2**(M-1)))
            Lor_val_temp1 = np.zeros((n,2**(M-1)))

            if class_prob == False or (class_prob == True and pred_out == 'predict'):
                for j in range(n):
                    Lor_val_temp[j,:] = j*(y_k[j,:]-y_base[j,:])
                Lor_val = ((2/(n**2))*np.mean(y))*(Lor_val_temp.sum(0))
                Lor_val = Lor_val.reshape((1,2**(M-1)))

                LZ[k,0] = np.dot(Lor_val,kernel)
            elif class_prob == True and (pred_out == 'predict_proba' or pred_out == 'predict_log_proba'):
                for j in range(n):
                    Lor_val_temp0[j,:] = j*(y_k0[j,:]-y_b0[j,:])
                    Lor_val_temp1[j,:] = j*(y_k1[j,:]-y_b1[j,:])
                Lor_val0 = ((2/(n**2))*y_class_mu[0])*(Lor_val_temp0.sum(0))
                Lor_val1 = ((2/(n**2))*y_class_mu[1])*(Lor_val_temp1.sum(0))
                Lor_val0 = Lor_val0.reshape((1,2**(M-1)))
                Lor_val1 = Lor_val1.reshape((1,2**(M-1)))

                LZ0[k,0] = np.dot(Lor_val0,kernel)
                LZ1[k,0] = np.dot(Lor_val1,kernel)
        if class_prob == False or (class_prob == True and pred_out == 'predict'):
            if show_last_y == True:
                return LZ, y_k, y_base
            else:
                return LZ
        elif class_prob == True and (pred_out == 'predict_proba' or pred_out == 'predict_log_proba'):
            if show_last_y == True:
                return LZ0, LZ1, y_k0, y_k1, y_b0, y_b1
            else:
                return LZ0, LZ1
