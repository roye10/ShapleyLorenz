# # ------------------------------------------------------------------------------------------
# #                             Shapley Lorenz Function
# # ------------------------------------------------------------------------------------------

# # Modules

# import random
# import itertools
# import warnings
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.special import binom, factorial
# from sklearn.utils import shuffle
# from tqdm import tqdm

class ShapleyLorenzShare:
    '''
    Uses the Shapley approach to calculate Shapley Lorenz marginal contributions

    Parameters:
    ---------------------------------------------------------
    model : method
        specifies the prediction model
    X_dist : numpy.array
        n x p matrix containing the model covariates
    y_dist : vector
        n-vector containing the (true) values to predict
    '''
    def __init__(self, model, X_dist, y_dist):
        assert(callable(model), "need to specify the model prediction method, e.g. 'predict' or 'predict_proba'")
        self.model = standardise_model_format(model) # standardise model
        self.data = standardise_data_format(X_dist) # standardise data
        self.y_bg = y_dist
        self.N = self.data.data.shape[0]
        self.M = self.data.data.shape[1]

        # warn user of long runtime for large background dataset
        if self.N > 50:
            warnings.warn('A background dataset larger than 50, may cause prohibitive long runtime. Consider using a sample of maximally 100 observations.')
            user_input = str(input("Enter 's' to sample 50 observations and enter 'c', to continue with current background dataset"))
            if user_input == 's':
                idx = np.random.randint(self.N, size=50)
                self.data.data = self.data.data[idx,:]
                self.data = standardise_data_format(self.data.data)
                if len(self.y_bg.shape) == 1:
                    try:
                        self.y_bg = self.y_bg.iloc[idx]
                    except:
                        self.y_bg = self.y_bg[idx]
                elif len(self.y_bg) == 2:
                    try:
                        self.y_bg = self.y_bg.iloc[idx,:]
                    except:
                        self.y_bg = self.y_bg[idx,:]
                self.N = 50
            elif user_input == 'c':
                self.data.data = self.data.data
                self.y_bg = self.y_bg
        
        #Dimension of null_model
        null_model = self.model.f(self.data.data)
        self.yd = len(null_model.shape)
        
        #E[f(x)]
        self.fnull = np.sum((null_model.T*self.data.weights).T, 0)
    
        #Conditions on y
        assert(str(type(self.y_bg)).endswith("numpy.ndarray'>"), 'response observations need to be of "numpy.ndarray" format')

        #Conditions on X
        assert(len(self.data.data.shape) == 2, 'Need to specify an appropriate number of features, p. p has to be > 1')

        # shuffle background dataset, for the case that it is the same as the one to be explained
        self.data.data, self.y_bg = shuffle(self.data.data, self.y_bg)


        # Old version
        # assert(callable(model), "need to specify the model prediction method, e.g. 'predict' or 'predict_proba'")
        # self.model = convert_to_model(model) # standardise model
        # self.data = convert_to_data(X_background) # standardise data
        # self.y_bg = y_background
        # self.N = self.data.data.shape[0]
        # self.M = self.data.data.shape[1]
        # self.data.data = self.data.data.shuffle()

        # if self.N > 50:
        #     warnings.warn('a large background dataset may cause prohibitive long runtime')
        
        # #Dimension of null_model
        # null_model = self.model.f(self.data.data)
        # self.yd = len(null_model.shape)
        
        # #E[f(x)]
        # self.fnull = np.sum((null_model.T*self.data.weights).T, 0)
    
        # #Conditions on y
        # assert(str(type(self.y_bg)).endswith("numpy.ndarray'>"), 'response observations need to be of "numpy.ndarray" format')

        # #Conditions on X
        # assert(len(self.data.data.shape) == 2, 'Need to specify an appropriate number of features, p. p has to be > 1')

#   Combinatoric tool
    def powerset(self, iterable, max_full_subsetsize):
        '''
        Creates index vectors of length 0-M of the 'iterable' list of length M
        
        Parameters:
        ---------------------------------------------------------
        iterable : list or range
            range of indices to find all possible permutations of all lengths between 0 and M
        max_full_subsetsize : int
            number of full subset sizes to consider

        Output:
        ---------------------------------------------------------
        iterable chain
        '''
        s = list(iterable)
        return itertools.chain.from_iterable(itertools.combinations(s,r)\
            for r in range(len(s)+1))
            #s : iterable
            #r : length

    #Shapley Kernel
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
        return (factorial(s)*factorial(M-s-1))/factorial(M)
    
    #Polarisation of Lorenz Zonoid/Gini value
    def lz_polarisation(self, Lor_y, M):
        '''
        Polarises lorenz values, enabling lorenz values to be negative as well as constraining
        gini coefficient to be between 0 and 1.

        Method is based on the paper "On the Gini coefficient normalization
        when attributes with negative values are considered"
        by Raffinetti et al. 2015
        See https://rdrr.io/cran/GiniWegNeg/man/Gini_RSV.html
        for original code in R

        Parameters:
        ---------------------------------------------------------
        Lor_y : vector
            vector of points on the lorenz curve to be polarised
        
        M : int
            number of features

        Output:
        ---------------------------------------------------------
        Returns Lorenz Zonoid/Gini Coefficient 
        '''
        n = Lor_y.shape[0]
        Gin_pol = np.zeros((1,2**(M-1)))

        s_all = sum(Lor_y,0)
        s_pos = sum(Lor_y[Lor_y > 0],0)
        s_neg = sum(abs(Lor_y[Lor_y <= 0]),0)
        del_pol = 2*((n-1)/(n**2))*(s_pos+s_neg)
        mu_pol = (1/2)*del_pol

        for i,s in enumerate(itertools.combinations(range(n),2)):
            Gin_pol[0,:] = (abs((s[0]*Lor_y[s[0],:]) - s[1]*(Lor_y[s[1],:]))).sum(0)
    
        return (1/(2*mu_pol*(n**2)))*Gin_pol

    # Plotting tool
    def slz_plots(self, LZ_shares, tuple = False, feature_labels = None):
        '''
        Creates a plot of the lorenz zonoid values
        
        Parameter:
        tuple : boolean (Default = False)
            specifies, whether to plot a seperate graph for the
            class tuple, in case of multiple classes
        '''
        lz_sorted = LZ_shares[np.argsort(LZ_shares[:,1])]
        if tuple == False:
            if feature_labels == None:
                x_val = lz_sorted[:,0]
            elif feature_labels != None:
                x_val = feature_labels
            plt.bar(x_val, lz_sorted[:,1], color = (0.0,0.36,0.8,0.8))
            plt.title('Shapley Lorenz Zonoid Values', fontsize = '15')
            plt.ylabel('shapley LZ value')
            plt.xlabel('features')
            plt.show()

    # Shapley Lorenz Zonoid function
    def shapleyLorenz_val(self, k, X, y, n_iter = 3000, class_prob = False, pred_out = 'predict', **kwargs):
        '''
        Computes the Shapley Lorenz marginal contribution for
        all covariates passed through in X.

        Parameters:
        ---------------------------------------------------------
        X : array
            covariate matrix
        y : array
            response variable
        n_iter : integer (DEFAULT: 3000)
            number of subset permutations to consider
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

        #Conditions on y
        assert(str(type(y)).endswith("numpy.ndarray'>"), 'response observations need to be of "numpy.ndarray" format')

        #Conditions on X
        assert X.shape[1] == self.M, 'Need to have the same number of features as in background dataset'
        
        assert X.shape[0] == len(y), 'Covariate matrix and response vector need to have the same number of observations'

        #Initiate variables globally
        X = standardise_data_format(X) # standardise data
        self.N_test = X.data.shape[0]
        self.row = kwargs.get('row', None)
        if self.row == False:
            assert isinstance(self.row, (int, list, np.ndarray)), "not a valid row type. Needs to be either 'int', 'list', or 'array'"

        if class_prob == True and (pred_out == 'predict_proba' or pred_out == 'predict_log_proba'):
            val, num = np.unique(self.y_bg, return_counts = True)
            if len(val) == 1:
                raise ValueError('only observations from one class included in provided data. Need to have at least one observation from each class')
            self.y_class_mu = np.zeros((val.shape[0],1))
            for i in range(val.shape[0]):
                self.y_class_mu[i] = num[i]/self.N
        elif class_prob == False or (class_prob == True and pred_out == 'predict'):
            self.y_mu = np.mean(y)

        #Container for output
        self.LZ = np.zeros((self.M, 1)) # in regression case or if 'predict' specified in classification case
        if pred_out == 'predict_proba':
            self.LZ0 = np.zeros((self.M,1))
            self.LZ1 = np.zeros((self.M,1))

        # Set up kernel
        self.kernel = np.zeros((2**(self.M-1),1))
        for i,s in enumerate(self.powerset(range(self.M-1), 2**(self.M-1))):
            self.kernel[i,0] = self.shapleyKernel(self.M, len(s))

        # Determine how many full subset sizes can be considered given "n_iter"
        num_subsets = np.int(np.floor((self.M-1)/2)) if np.mod((self.M-1),2) != 0 else np.int((self.M-1)/2)
        num_sub_sizes = 0
        sub_sizes = 0
        self.full_subsetsizes = 0
        for m in range(num_subsets+1):
            if sub_sizes < n_iter and m != num_subsets:
                if np.mod(self.M-1,2) != 0 or (np.mod(self.M-1,2) == 0 and m != (self.M-1)/2):
                    swtch = 0
                    sub_sizes += 2*binom(self.M-1, m)
                    num_sub_sizes += 1
                elif np.mod(self.M-1,2) == 0 and m == ((self.M-1)/2):
                    swtch = 1
                    sub_sizes += binom(self.M-1,m)
                    num_sub_sizes += 1
            elif sub_sizes > n_iter:
                self.full_subsetsizes = (sub_sizes - 2*binom(self.M-1,m-1)) if swtch == 0 else sub_sizes - binom(self.M-1,m-1)
                num_sub_sizes -= 2
                break
            elif sub_sizes == n_iter or (sub_sizes < n_iter and m == num_subsets):
                self.full_subsetsizes = sub_sizes
        self.full_subsetsizes = np.int(self.full_subsetsizes)

        #Loop over all covariates (this is parallalised therefore no loop here)

        #Initialise indexes
        s_all = list(range(self.M))
        s_base = s_all.copy()
        print(f'started covariates: {k}')
        s_base.pop(k)
        k = [k, ]

        # Create dictionary of all subset sizes for which all permutations are considered
        self.full_permts = {}
        for i,s in enumerate(self.powerset(s_base,len(s_base)+1)):
            if len(s) in range(num_sub_sizes + 1) or len(s) in range(self.M-1-num_sub_sizes,self.M):
                self.full_permts[i] = s
        self.full_permts = pd.DataFrame(list(self.full_permts.items()))

        # Create dictionary of all remaining subset permutations, after considering full subset permutations, given "n_iter"
        remaining_permts = {}
        for i,s in enumerate(self.powerset(s_base, len(s_base)+1)):
            if len(s) not in range(num_sub_sizes + 1) and len(s) not in range(self.M-1-num_sub_sizes,self.M):
                remaining_permts[i] = s
        remaining_permts = pd.DataFrame(list(remaining_permts.items()))

        #Initialise variables within loop
        V_base = np.zeros((self.N, self.M, len(self.full_permts))) # here and in the following only (M-1) permutations, because
                                                        # base maximally has M-1 covariates
        V_k = np.zeros((self.N, self.M, len(self.full_permts)))
        Lor_k = np.zeros((len(self.full_permts),1))
        Lor_base = np.zeros((len(self.full_permts),1))

        Lor_k0 = np.zeros((len(self.full_permts),1))
        Lor_k1 = np.zeros((len(self.full_permts),1))
        Lor_base0 = np.zeros((len(self.full_permts),1))
        Lor_base1 = np.zeros((len(self.full_permts),1))

        ranks = np.arange(self.N_test)

        if class_prob == False or (class_prob == True and pred_out == 'predict'):
            y_base = np.zeros((self.N_test, len(self.full_permts)))
            y_k = np.ones((self.N_test, len(self.full_permts)))
        elif class_prob == True and pred_out == 'predict_proba':
            y_base = np.zeros((self.N_test, 2, len(self.full_permts)))
            y_b0 = np.zeros((self.N_test, len(self.full_permts)))
            y_b1 = np.zeros((self.N_test, len(self.full_permts)))
            y_k = np.zeros((self.N_test, 2, len(self.full_permts)))
            y_k0 = np.zeros((self.N_test, len(self.full_permts)))
            y_k1 = np.zeros((self.N_test, len(self.full_permts)))
        
        # update kernel
        kernel = np.zeros((len(self.full_permts),1))
        kernel_row = 0
        for i in range(len(self.kernel)):
            if i in self.full_permts.iloc[:,0]:
                kernel[kernel_row,0] = self.kernel[i,0]
                kernel_row += 1

        #loop over all possible full permutations, given "n_iter"
        i = 0
        for s in self.full_permts.iloc[:,1]:
            #Initialise Kernel
            s = list(s) # covariates in baseline (base model)
            s_k = k+s # baseline covariates + kth covariate (model k)

            #Initialise background datasets for model including kth covariate and model excluding kth covariate
            V_base[:,:,i] = self.data.data
            V_k[:,:,i] = self.data.data

            #for single row
            if self.row == False:
                if type(self.row) == int:
                    if len(s) == 0:
                            if i != 0:
                                raise ValueError('s is empty for i not equal 0')
                            continue
                    else:
                        V_base[:,s,i] = X.data[n_test, s]
                    V_k[:,s_k,i] = X.data[n_test, s_k]
                    if class_prob == False or (class_prob == True and pred_out == 'predict'):
                        y_k[:,i] = self.model.f(V_k[self.row,:,i])

                        y_base[:,i] = self.model.f(V_base[self.row,:,i])
                
                    elif class_prob == True and (pred_out == 'predict_proba' or pred_out == 'predict_log_proba'):
                        y_k[:,:,i] = self.model.f(V_k[self.row,:,i])
                        y_k0[:,i] = y_k[:,0,i]
                        y_k1[:,i] = y_k[:,1,i]

                        y_base[:,:,i] = self.model.f(V_base[self.row,:,i])
                        y_b0[:,i] = y_base[:,0,i]
                        y_b1[:,i] = y_base[:,1,i]
                    else:
                        raise ValueError\
                            ("Not a valid method. Valid methods are: 'predict', 'predict_proba' and 'predict_log_proba'")

                #For specified rows
                elif isinstance(self.row, (list, np.ndarray)):
                    for n_test in range(len(self.row)):
                        if len(s) == 0:
                            if i != 0:
                                raise ValueError('s is empty for i not equal 0')
                            continue
                        else:
                            V_base[:,s,i] = X.data[n_test, s]
                        V_k[:,s_k,i] = X.data[n_test,s_k]

                        if self.row == False and isinstance(self.row, (list, np.ndarray)):
                            if class_prob == False or (class_prob == True and pred_out == 'predict'):
                                yk_temp = self.model.f(V_k[:,:,i])
                                y_k[n_test, i] = np.mean(yk_temp,0)
                                
                                ybase_temp = self.model.f(V_base[:,:,i])
                                y_base[n_test,i] = np.mean(ybase_temp,0)

                        elif class_prob == True and (pred_out == 'predict_proba' or pred_out == 'predict_log_proba'):
                            yk_temp = self.model.f(V_k[:,:,i]).reshape(self.N,2)
                            y_k[n_test, 0, i] = np.mean(yk_temp[:,0],0)
                            y_k[n_test, 1, i] = np.mean(yk_temp[:,1],0)

                            ybase_temp = self.model.f(V_base[:,:,i]).reshape(self.N,2)
                            y_base[n_test,0,i] = np.mean(ybase_temp[:,0],0)
                            y_base[n_test,1,i] = np.mean(ybase_temp[:,1],0)
                        else:
                            raise ValueError\
                                ("Not a valid method. Valid methods are: 'predict', 'predict_proba' and 'predict_log_proba'")

            # No specified rows
            elif self.row in (True, None):
                for n_test in range(self.N_test):
                    if len(s) == 0:
                        V_k[:,s_k,i] = X.data[n_test,s_k]
                    else:
                        V_base[:,s,i] = X.data[n_test, s]
                        V_k[:,s_k,i] = X.data[n_test,s_k]
                    #print('\nV_base initial shape: {}'.format(V_base.shape))

                    #Compute predicted values with model w and w/o kth covariat, if no row(s) specified

                    if class_prob == False or (class_prob == True and pred_out == 'predict'):
                        ybase_temp = self.model.f(V_base[:,:,i])
                        y_base[n_test,i] = np.mean(ybase_temp,0)

                        yk_temp = self.model.f(V_k[:,:,i])
                        y_k[n_test, i] = np.mean(yk_temp,0)

                    elif class_prob == True and (pred_out == 'predict_proba' or pred_out == 'predict_log_proba'):
                        yk_temp = self.model.f(V_k[:,:,i]).reshape(self.N,2)
                        y_k[n_test, 0, i] = np.mean(yk_temp[:,0],0)
                        y_k[n_test, 1, i] = np.mean(yk_temp[:,1],0)

                        ybase_temp = self.model.f(V_base[:,:,i]).reshape(self.N,2)
                        y_base[n_test,0,i] = np.mean(ybase_temp[:,0],0)
                        y_base[n_test,1,i] = np.mean(ybase_temp[:,1],0)

                    elif pred_out not in ('predict', 'predict_proba', 'predict_log_proba'):
                        raise ValueError\
                            ('No valid method. Valid methods are: predict, predict_proba')
            
                #Sort predicted values
                if class_prob == False or (class_prob == True and pred_out == 'predict'):
                    y_base[:,i] = np.sort(y_base[:,i],0)
                    y_k[:,i] = np.sort(y_k[:,i],0)

                elif class_prob == True and (pred_out == 'predict_proba' or pred_out == 'predict_log_proba'):    
                    y_b0[:,i] = np.sort(y_base[:,0,i],0)
                    y_b1[:,i] = np.sort(y_base[:,1,i],0)

                    y_k0[:,i] = np.sort(y_k[:,0,i],0)
                    y_k1[:,i] = np.sort(y_k[:,1,i],0)

            #Compute Lorenz Zenoid values
            # Lor_val_temp = np.zeros((self.N_test,2**(self.M-1)))
            # Lor_val_temp0 = np.zeros((self.N_test,2**(self.M-1)))
            # Lor_val_temp1 = np.zeros((self.N_test,2**(self.M-1)))

            if class_prob == False or (class_prob == True and pred_out == 'predict'):
                Lor_k[i,0] = (2/(self.N_test*self.y_mu))*np.cov(y_k[:,i], ranks, bias = True)[0][1]
                Lor_base[i,0] = (2/(self.N_test*self.y_mu))*np.cov(y_base[:,i], ranks, bias = True)[0][1]

            elif class_prob == True and (pred_out == 'predict_proba' or pred_out == 'predict_log_proba'):
                Lor_k0[i,0] = (2/(self.N_test*self.y_class_mu[0]))*np.cov(y_k0[:,i], ranks, bias = True)[0][1]
                Lor_k1[i,0] = (2/(self.N_test*self.y_class_mu[1]))*np.cov(y_k1[:,i], ranks, bias = True)[0][1]
                Lor_base0[i,0] = (2/(self.N_test*self.y_class_mu[0]))*np.cov(y_b0[:,i], ranks, bias = True)[0][1]
                Lor_base1[i,0] = (2/(self.N_test*self.y_class_mu[1]))*np.cov(y_b1[:,i], ranks, bias = True)[0][1]
                    
            # for j in range(self.N_test):
            #     Lor_val_temp[j,:] = j*(y_k[j,:]-y_base[j,:]) # for all feature combinations simultaneously
            # Lor_val_temp_sum = np.sum(Lor_val_temp,0)
            #Lor_val_pol = self.lz_polarisation(Lor_val_temp,self.M) # polarisation in case of negative values
            i += 1
        
        if class_prob == False or (class_prob == True and pred_out == 'predict'):
            val_bool = 0
            Lor_val = Lor_k - Lor_base
            self.LZ[k,0] = np.dot(Lor_val.T,kernel) # equation 19 on page 10 of Giudiuci and Raffinetti (Feb 2020) paper
            
        elif class_prob == True and (pred_out == 'predict_proba' or pred_out == 'predict_log_proba'):
            val_bool = 1
            Lor_val0 = Lor_k0 - Lor_base0
            Lor_val1 = Lor_k1 - Lor_base1
            self.LZ0[k,0] = np.dot(Lor_val0.T,kernel) # equation 19 on page 10 of Giudiuci and Raffinetti (Feb 2020) paper
            self.LZ1[k,0] = np.dot(Lor_val1.T,kernel) # equation 19 on page 10 of Giudiuci and Raffinetti (Feb 2020) paper
                
                #Lor_val0_pol = self.lz_polarisation(Lor_val_temp0,self.M)
                #Lor_val1_pol = self.lz_polarisation(Lor_val_temp1,self.M)

        col_names = np.array(X.col_names).reshape((self.M,1))
        if val_bool == 0:
            return np.hstack((col_names, self.LZ))
        elif val_bool == 1:
            return np.hstack((col_names, self.LZ0)), np.hstack((col_names, self.LZ1));

#Auxiliary functions

#standardised data format
class Data:
    def __init__(self, data, col_names):
        self.data = data
        self.col_names = col_names
        n = data.shape[0]
        self.weights = np.ones(n)
        self.weights /= n

def standardise_data_format(value):
    if isinstance(value, Data):
        return value
    elif type(value) == np.ndarray:
        return Data(value, [str(i) for i in range(value.shape[1])])
    elif str(type(value)).endswith("pandas.core.series.Series'>"):
        return Data(value.values.reshape((1,len(value))), value.index.tolist())
    elif str(type(value)).endswith("pandas.core.frame.DataFrame'>"):
        return Data(value.values, value.columns.tolist())
    else:
        assert False, str(type(value)) + "is currently not a supported format type"   

#Convert model to standard model class
class Model:
    def __init__(self, f):
        if str(type(f)).endswith('BinaryResultsWrapper'):
            self.f = f.values
        else:
            self.f = f

def standardise_model_format(value):
    if isinstance(value, Model):
        return value
    else:
        return Model(value)
