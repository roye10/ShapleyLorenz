# ------------------------------------------------------------------------------------------
#                                    Shapley Lorenz Functions (AS ON GITHUB)
# ------------------------------------------------------------------------------------------


# Modules
import numpy as np
from scipy.special import binom, factorial
import itertools
from tqdm import tqdm
import warnings

class ShapleyLorenzShare:
    '''
    Uses the Shapley approach to calculate Shapley Lorenz marginal contributions

    Parameters:
    ---------------------------------------------------------
    model : method
        specifies the prediction model
    X : numpy.array
        n x p matrix containing the model covariates
    y : vector
        n-vector containing the (true) values to predict
    '''
    def __init__(self, model, X_background, y_background):
        self.model = convert_to_model(model) # standardise model
        self.data = convert_to_data(X_background) # standardise data
        self.y = y_background
        self.N = self.data.data.shape[0]
        self.M = self.data.data.shape[1]
        
        # Dimension of null_model
        null_model = self.model.f(self.data.data)
        self.yd = len(null_model.shape)
        
        # E[f(x)]
        self.fnull = np.sum((null_model.T*self.data.weights).T, 0)
    
        # Conditions on y
        assert(str(type(self.y)).endswith("numpy.ndarray'>")), 'response observations need to be of "numpy.ndarray" format'

        # Conditions on X
        assert len(self.data.data.shape) == 2, 'Need to specify an appropriate number of features, p. p has to be > 1'

    # Combinatoric tool
    def powerset(self, iterable):
        '''
        Creates index vectors of length 0-M of the 'iterable' list of length M
        
        Parameters:
        ---------------------------------------------------------
        iterable : list or range
            range of indices to find all possible permutations of all lengths between 0 and M

        Output:
        ---------------------------------------------------------
        iterable chain
        '''
        s = list(iterable)
        return itertools.chain.from_iterable(itertools.combinations(s,r)\
            for r in range(len(s)+1))
            # s : iterable
            # r : length

    # Shapley Kernel
    def shapleyKernel(self, M, s):
        '''
        Returns an integer, which weights the permutation instance for M covariates of size s.
        It is proportional to the inverse of the binomial coefficient 'M choose s'.

        Parameters:
        ---------------------------------------------------------
        M : integer
            number of features
        s : vector
            vector of features to regard in the model building process for iteration round i
        
        Output:
        ---------------------------------------------------------
        Kernel weight : float64
        '''
        return factorial(s)*factorial(M-s-1)/factorial(M)
    
    # # Polarisation of Lorenz Zonoid/Gini value
    # def lz_polarisation(self, Lor_y, M):
    #     '''
    #     Polarises lorenz values, enabling lorenz values to be negative as well as constraining
    #     gini coefficient to be between 0 and 1.

    #     Method is based on the paper "On the Gini coefficient normalization
    #     when attributes with negative values are considered"
    #     by Raffinetti et al. 2015
    #     See https://rdrr.io/cran/GiniWegNeg/man/Gini_RSV.html
    #     for original code in R

    #     Parameters:
    #     ---------------------------------------------------------
    #     Lor_y : vector
    #         vector of points on the lorenz curve to be polarised
        
    #     M : int
    #         number of features

    #     Output:
    #     ---------------------------------------------------------
    #     Returns Lorenz Zonoid/Gini Coefficient 
    #     '''
    #     n = Lor_y.shape[0]
    #     Gin_pol = np.zeros((1,2**(M-1)))

    #     s_all = sum(Lor_y,0)
    #     s_pos = sum(Lor_y[Lor_y > 0],0)
    #     s_neg = sum(abs(Lor_y[Lor_y <= 0]),0)
    #     del_pol = 2*((n-1)/(n**2))*(s_pos+s_neg)
    #     mu_pol = (1/2)*del_pol

    #     for i,s in enumerate(itertools.combinations(range(n),2)):
    #         Gin_pol[0,:] = (abs((s[0]*Lor_y[s[0],:]) - s[1]*(Lor_y[s[1],:]))).sum(0)
    
    #     return (1/(2*mu_pol*(n**2)))*Gin_pol

    def shapleyLorenz_val(self, X, class_prob = False, pred_out = 'predict', **kwargs):
        '''
        Computes the Shapley Lorenz marginal contribution for
        all covariates passed through in X.

        Parameters:
        ---------------------------------------------------------
        class_prob : boolean (DEFAULT: False)
            if False --> regression problem
            if True --> classification problem
        pred_out : str (DEFAULT: 'predict')
            Need to specify if class_prob = True
            prediction output to use. Available options:
            'predict' --> float 64 in regression case and 1/0 in classification case
            'predict_proba' --> outputs float64 class probabilities (ONLY FOR CLASSIFICATION PROBLEMS)
        row : int (DEFAULT: None)
            observation(s) to explain
        
        Output:
        ---------------------------------------------------------
        Lorenz marginal contribution coefficient : vector
        Function returns the Lorenz marginal contribution coefficient for each
        feature. In case of classification returns a tuple for the classes
        and a single vector in a regression case.
        '''

        # Container for output
        LZ = np.zeros((self.M, 1)) # in regression case or if 'predict' specified in classification case
        if pred_out == 'predict_proba':
            LZ0 = np.zeros((self.M,1))
            LZ1 = np.zeros((self.M,1))

        if class_prob == True:
            val, num = np.unique(self.y, return_counts = True)
            if len(val) == 1:
                raise ValueError('only observations from one class included in data provided. Need to have at least one observation from each class')
            y_class_mu = np.zeros((val.shape[0],1))
            for i in range(val.shape[0]):
                y_class_mu[i] = num[i]/self.N
            print(y_class_mu)
        else:
            y_mu = np.mean(self.y)

        # Loop over all covariates
        for k in tqdm(range(self.M), disable = kwargs.get('silent', False)):
            # Initialise variables
            V_base = np.zeros((self.N, self.M, 2**(self.M-1))) # here and in the following only (M-1) permutations, because\
                                                                # base maximally has M-1 covariates
            V_k = np.zeros((self.N, self.M, 2**(self.M-1)))
            kernel = np.zeros((2**(self.M-1),1))
            if class_prob == False or (class_prob == True and pred_out == 'predict'):
                y_base = np.zeros((self.N, 2**(self.M-1)))
                y_k = np.zeros((self.N, 2**(self.M-1)))
            elif class_prob == True and pred_out == 'predict_proba':
                y_base = np.zeros((self.N, 2, 2**(self.M-1)))
                y_b0 = np.zeros((self.N, 2, 2**(self.M-1)))
                y_b1 = np.zeros((self.N, 2, 2**(self.M-1)))
                y_k = np.zeros((self.N, 2, 2**(self.M-1)))
                y_k0 = np.zeros((self.N, 2, 2**(self.M-1)))
                y_k1 = np.zeros((self.N, 2, 2**(self.M-1)))
            
            # Initialise indexes
            s_all = list(range(self.M))
            s_base = s_all.copy()
            s_base.pop(k)
            k = [k, ]

            # loop over all possible (2**(M-1)) covariate combinations
            for i,s in enumerate(self.powerset(s_base)): 

                # Initialise background dataset
                V_k[:,:,i] = self.data.data
                V_base[:,:,i] = self.data.data

                s = list(s) # covariates in baseline (base model)
                s_k = k+s # baseline covariates + kth covariate (model k)

                V_base[:,s,i] = X[:, s]
                # print('\nV_base initial shape: {}'.format(V_base.shape))
                V_k[:,s_k,i] = X[:, s_k]

                # Compute Kernel
                kernel[i,0] = self.shapleyKernel(self.M, len(s))

                # yHat including covariate k
                # For specified rows
                self.row = kwargs.get('row', None)
                if self.row == False:
                    if class_prob == False or (class_prob == True and pred_out == 'predict'):
                        if type(self.row) == int:
                            y_k[:,i] = self.model.fit(V_k[:,s_k,i],self.y)\
                                .predict(V_k[self.row,s_k,i].reshape(1,-1))
                        elif isinstance(self.row, (list, np.ndarray)) and len(self.row) > 1:
                            y_k[:,i] = self.model.fit(V_k[:,s_k,i],self.y)\
                                .predict(V_k[self.row,s_k,i]).reshape(len(self.row))
                            y_k[:,i] = np.sort(y_k[:,i],0)
                    elif class_prob == True:
                        if pred_out == 'predict_proba' or pred_out == 'predict_log_proba':
                            if type(self.row) == int:
                                y_k[:,:,i] = getattr(self.model.fit(V_k[:,s_k,i],self.y)\
                                    ,pred_out)(V_k[self.row,s_k,i].reshape(1,-1))
                                y_k0[:,i] = y_k[:,0,i]
                                y_k1[:,i] = y_k[:,1,i]
                            elif len(self.row) > 1:
                                y_k[:,:,i] = getattr(self.model.fit(V_k[:,s_k,i],self.y)\
                                    ,pred_out)(V_k[self.row,s_k,i]).reshape(len(self.row),2)
                                y_k0[:,i] = np.sort(y_k[:,0,i],0)
                                y_k1[:,i] = np.sort(y_k[:,1,i],0)
                        else:
                            raise ValueError\
                                ('No valid method. Valid methods are: predict, predict_proba')
                    
                # if no row(s) specified
                elif self.row == None:
                    if class_prob == False or (class_prob == True and pred_out == 'predict'):
                        y_k[:, i] = self.model.f(V_k[:,:,i])
                        # y_k[:, i] = self.model.fit(V_k[:,s_k,i],self.y)\
                        #     .predict(V_k[:,s_k,i]).reshape(self.N)
                        y_k[:,i] = np.sort(y_k[:,i],0)
                    if class_prob == True:
                        if pred_out == 'predict_proba' or pred_out == 'predict_log_proba':
                            y_k[:, :, i] = self.model.f(V_k[:,:,i]).reshape(self.N,2)
                            # y_k[:,:,i] = getattr(self.model.fit(V_k[:,s_k,i],self.y)\
                            #     ,pred_out)(V_k[:,s_k,i]).reshape(n,2)
                            y_k0[:,i] = np.sort(y_k[:,0,i],0)
                            y_k1[:,i] = np.sort(y_k[:,1,i],0)
                        elif pred_out not in ('predict'  ,'predict_proba', 'predict_log_proba'):
                            raise ValueError\
                                ('No valid method. Valid methods are: predict, predict_proba')
                
                # yHat baseline, (w/o covariate k)
                # For specified rows
                if self.row == False:
                    if len(s) == 0:
                        s = np.arange(M)
                        if i != 0:
                            raise ValueError('s is empty for i not equal 0')
                        # Shuffle rows for each feature
                        for j in range(X.shape[1]):
                                perm_indx = np.random.randint(0, self.N, self.N)
                                V_base[:,j,i] = X[perm_indx,j]
                    else:
                        V_base[:,s,i] = X[:,s]

                    if class_prob == False or (class_prob == True and pred_out == 'predict'):
                        if type(self.row) == int:
                            y_base[:,i] = self.model.fit(V_base[:,s,i],self.y)\
                                .predict(V_base[self.row,s,i].reshape(1,-1))
                        elif len(self.row) > 1:
                            y_base[:,i] = self.model.fit(V_base[:,s,i],self.y)\
                                .predict(V_base[self.row,s,i]).reshape(len(self.row))
                            y_base[:,i] = np.sort(y_base[:,i],0)
                    elif class_prob == True:
                        if pred_out == 'predict_proba' or pred_out == 'predict_log_proba':
                            if type(self.row) == int:
                                y_base[:,:,i] = getattr(self.model.fit(V_base[:,s,i],self.y)\
                                    ,pred_out)(V_k[self.row,s,i].reshape(1,-1))
                                y_b0[:,i] = y_base[:,0,i]
                                y_b1[:,i] = y_base[:,1,i]
                            elif len(self.row) > 1:
                                y_base[:,:,i] = getattr(self.model.fit(V_base[:,s,i],self.y)\
                                        ,pred_out)(V_k[self.row,s,i]).reshape(len(self.row),2)
                                y_b0[:,i] = np.sort(y_base[:,0,i],0)
                                y_b1[:,i] = np.sort(y_base[:,1,i],0)

                elif self.row == None:
                    if len(s) == 0:
                        s = np.arange(self.M)
                        if i != 0:
                            warnings.warn('s is empty for i not equal 0')
                        # Shuffle rows for each feature
                        # for j in range(X.shape[1]):
                        #         perm_indx = np.random.randint(0, self.N, self.N)
                        #         V_base[:,j,i] = X[perm_indx,j]
                        V_base[:,:,i] = self.data.data
                        # print('\nV_base shape: {}'.format(V_base.shape))
                        # print('\ny_base shape: {}'.format(y_base.shape))
                    else:
                        V_base[:,s,i] = X[:,s]
                        
                    if class_prob == False or (class_prob == True and pred_out == 'predict'):
                        y_base[:,i] = self.model.f(V_base[:,:,i])
                        # y_base[:,i] = self.model.fit(V_base[:,s,i],self.y)\
                        #     .predict(V_base[:,s,i]).reshape(n)
                        y_base[:,i] = np.sort(y_base[:,i],0)
                    elif class_prob == True:
                        if pred_out == 'predict_proba' or pred_out == 'predict_log_proba':
                            y_base[:,:,i] = self.model.f(V_base[:,:,i]).reshape(n,2)
                            # y_base[:,:,i] = getattr(self.model.fit(V_base[:,s,i],self.y)\
                            #     ,pred_out)(V_base[:,s,i]).reshape(n,2)
                            y_b0[:,i] = np.sort(y_base[:,0,i],0)
                            y_b1[:,i] = np.sort(y_base[:,1,i],0)

            # Compute Lorenz Zenoid values
            Lor_val_temp = np.zeros((self.N,2**(self.M-1)))
            Lor_val_temp0 = np.zeros((self.N,2**(self.M-1)))
            Lor_val_temp1 = np.zeros((self.N,2**(self.M-1)))

            if class_prob == False or (class_prob == True and pred_out == 'predict'):
                for j in range(self.N):
                    Lor_val_temp[j,:] = j*(y_k[j,:]-y_base[j,:]) # for all feature combinations simultaneously
                Lor_val_temp_sum = np.sum(Lor_val_temp,0)
                # Lor_val_pol = self.lz_polarisation(Lor_val_temp,M) # polarisation in case of negative values

                # if show_y_for_k == True:
                #     self.y_kShow = y_k
                #     self.y_baseShow = y_base

                Lor_val = ((2/(self.N**2))*np.mean(self.y))*Lor_val_temp_sum
                Lor_val = Lor_val.reshape((1,2**(self.M-1)))

                LZ[k,0] = np.dot(Lor_val,kernel) # equation 19 on page 10 of Giudiuci and Raffinetti (Feb 2020) paper

            elif class_prob == True and (pred_out == 'predict_proba' or pred_out == 'predict_log_proba'):
                for j in range(self.N):
                    Lor_val_temp0[j,:] = j*(y_k0[j,:]-y_b0[j,:])
                    Lor_val_temp1[j,:] = j*(y_k1[j,:]-y_b1[j,:])
                
                # Lor_val0_pol = self.lz_polarisation(Lor_val_temp0,M)
                # Lor_val1_pol = self.lz_polarisation(Lor_val_temp1,M)

                # if show_last_y == True:
                #     self.y_k0Show = y_k0
                #     self.y_b0Show = y_b0
                #     self.y_k1Show = y_k1
                #     self.y_b1Show = y_b1
                #     self.testing = k

                Lor_val0 = ((2/(self.N**2))*y_class_mu[0])*Lor_val_temp0
                Lor_val1 = ((2/(self.N**2))*y_class_mu[1])*Lor_val_temp1
                Lor_val0 = Lor_val0.reshape((1,2**(self.M-1)))
                Lor_val1 = Lor_val1.reshape((1,2**(self.M-1)))

                LZ0[k,0] = np.dot(Lor_val0,kernel)
                LZ1[k,0] = np.dot(Lor_val1,kernel)

        if class_prob == False or (class_prob == True and pred_out == 'predict'):
            return LZ

        elif class_prob == True and (pred_out == 'predict_proba' or pred_out == 'predict_log_proba'):
            return LZ0, LZ1;


# Auxiliary functions

# standardised data format
class Data:
    def __init__(self, data, col_names):
        self.data = data
        self.col_names = col_names
        n = data.shape[0]
        self.weights = np.ones(n)
        self.weights /= n

def convert_to_data(value):
    if isinstance(value, Data):
        return value
    elif type(value) == np.ndarray:
        return Data(value, [str(i) for i in range(value.shape[1])])
    elif str(type(value)).endswith("pandas.core.series.Series'>"):
        return Data(value.values.reshape((1,len(values))), value.index.tolist())
    elif str(type(value)).endswith("pandas.core.frame.DataFrame'>"):
        return Data(value.values, value.columns.tolist())
    else:
        assert False, str(type(value)) + "is currently not a supported format type"   

# Convert model to standard model class
class Model:
    def __init__(self, f):
        self.f = f

def convert_to_model(value):
    if isinstance(value, Model):
        return value
    else:
        return Model(value)


def powerset(iterable):
        '''
        Creates index vectors of length 0-M of 'iterable'
        '''
        s = list(iterable)
        return itertools.chain.from_iterable(itertools.combinations(s,r)\
            for r in range(len(s)+1))
            # s : iterable
            # r : length

# ---------------------------------
# Tests
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# Sim data
# background data
X_bg = np.random.normal(0,1,(100,16))
betas = np.random.uniform(1,6,16)
y_bg = np.dot(X_bg,betas)

# data to explain
X = np.random.normal(0,1,(100,16))

model = LinearRegression()
model.fit(X_bg,y_bg)

lorenzshare = ShapleyLorenzShare(model.predict, X_bg, y_bg)
lorenzshare.shapleyLorenz_val(X)
