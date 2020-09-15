class ShapleyLorenzShare:
    '''
    Uses the Shapley approach to calculate Shapley Lorenz marginal contributions

    Parameters:
    ---------------------------------------------------------
    model : method
        specifies the prediction model
    X : matrix
        nxp matrix containing the model covariates
    y : vector
        n-vector containing the values to predict
    '''
    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y

# Combinatoric tool
    def powerset(self, iterable):
        s = list(iterable)
        return itertools.chain.from_iterable(itertools.combinations(s,r)\
            for r in range(len(s)+1))
            # s : iterable
            # r : length

    # Shapley Kernel
    def ShapleyKernel(self, M, s):
        '''
        Returns an integer, which weights the permutated model outcomes.

        Parameters:
        ---------------------------------------------------------
        M : integer
            number of features
        s : vector
            vector of features to regard in the model building process for iteration round i
        '''
        return factorial(s)*factorial(M-s-1)/factorial(M)

    def ShapleyLorenz_val(self, class_prob = False, row = None, pred_out = None, show_last_y = False):
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
            'predict' --> and 1/0 in classification caes
            'predict_proba' --> outputs float64 class probabilities (ONLY FOR CLASSIFICATION PROBLEMS)
        row : int (DEFAULT: None)
            observation(s) to explain
        show_last_y : boolean (DEFAULT: False)
            True --> shows last vector of predicted values for last included feature
            False --> does not output last vector of predicted values
        show_y_for_k : int (DEFAULT: None)
            integer specifying to show predicted values of model_k, y_k, and base model y_base
        '''

        # Transform to array
        self.X = np.array(self.X)
        self.y = np.array(self.y)

        # Conditions
        if len(self.X.shape) == 1:
            raise ValueError('Need to specify an appropriate number of features. p has to be >= 1')
        if self.X.shape[1] > 10:
            raise Warning('For features larger than 10, runtime is prohibitively long, due to problem to solve becoming NP hard\
                            a value less than 10 is suggested')
        M = self.X.shape[1]
        if class_prob == True and pred_out == None:
            raise ValueError('Need to specify if class_prob = True')

        if self.X.shape[0] == 1:
            raise ValueError('Need to specify an appropriate number of observations. n >= 100 is suggested')
        else:
            n = self.X.shape[0]

        LZ = np.zeros((M,1))
        if pred_out == 'predict_proba' or pred_out == 'predict_log_proba':
            LZ0 = np.zeros((M,1))
            LZ1 = np.zeros((M,1))

        if class_prob == True:
            val, num = np.unique(self.y, return_counts = True)
            y_class_mu = np.zeros(len(val))
            for i in range(val.shape[0]):
                y_class_mu[i] = num[i]/n
            print(y_class_mu)
        
        if pred_out == 'predict_proba' or pred_out == 'predict_log_proba':
            pred = True
        else:
            pred = False

        # Loop over number of covariates
        for k in range(M):
            # Initialise
            V_base = np.zeros((n,M,2**(M-1)))
            V_k = np.zeros((n,M,2**(M-1)))
            if class_prob == False or (class_prob == True and pred_out == 'predict'):
                y_base = np.zeros((n,2**(M-1)))
                y_k = np.zeros((n,2**(M-1)))
            elif class_prob == True and (pred_out == 'predict_proba' or 'predict_loga_proba'):
                y_base = np.zeros((n,2,2**(M-1)))
                y_b0 = np.zeros((n,2**(M-1)))
                y_b1 = np.zeros((n,2**(M-1)))
                y_k = np.zeros((n,2,2**(M-1)))
                y_k0 = np.zeros((n,2**(M-1)))
                y_k1 = np.zeros((n,2**(M-1)))

            kernel = np.zeros((2**(M-1),1))
            
            # Initialise indexes
            s_all = list(range(M))
            s_base = s_all.copy()
            s_base.pop(k)
            k = [k, ]
            # loop over all possible (2**(M-1)) covariate
            # combinations
            for i,s in enumerate(self.powerset(s_base)):
                s = list(s)
                s_k = k+s

                V_base[:,s,i] = self.X[:,s]
                V_k[:,s_k,i] = self.X[:,s_k]

                # Compute Kernel
                kernel[i,0] = self.ShapleyKernel(M,len(s))

                # yHat including covariate k
                # For specified rows
                if row != None:
                    if class_prob == False or (class_prob == True and pred_out == 'predict'):
                        if len(row) == 1:
                            y_k[:,i] = self.model.fit(V_k[:,s_k,i],self.y)\
                                .predict(V_k[row,s_k,i].reshape(1,-1))
                        elif len(row) > 1:
                            y_k[:,i] = self.model.fit(V_k[:,s_k,i],self.y)\
                                .predict(V_k[row,s_k,i]).reshape(len(row))
                            y_k[:,i] = np.sort(y_k[:,i],0)
                    elif class_prob == True:
                        if pred_out == 'predict_proba' or pred_out == 'predict_log_proba':
                            if len(row) == 1:
                                y_k[:,:,i] = getattr(self.model.fit(V_k[:,s_k,i],self.y)\
                                    ,pred_out)(V_k[row,s_k,i].reshape(1,-1))
                                y_k0[:,i] = y_k[:,0,i]
                                y_k1[:,i] = y_k[:,1,i]
                            elif len(row) > 1:
                                y_k[:,:,i] = getattr(self.model.fit(V_k[:,s_k,i],self.y)\
                                    ,pred_out)(V_k[row,s_k,i]).reshape(len(row),2)
                                y_k0[:,i] = np.sort(y_k[:,0,i],0)
                                y_k1[:,i] = np.sort(y_k[:,1,i],0)
                        else:
                            raise ValueError\
                                ('No valid method. Valid methods are: predict, predict_proba')
                    
                # if no row(s) is(are) specified
                elif row == None:
                    if class_prob == False or (class_prob == True and pred_out == 'predict'):
                        y_k[:,i] = self.model.fit(V_k[:,s_k,i],self.y)\
                            .predict(V_k[:,s_k,i]).reshape(n)
                        y_k[:,i] = np.sort(y_k[:,i],0)
                    if class_prob == True:
                        if pred_out == 'predict_proba' or pred_out == 'predict_log_proba':
                            y_k[:,:,i] = getattr(self.model.fit(V_k[:,s_k,i],self.y)\
                                ,pred_out)(V_k[:,s_k,i]).reshape(n,2)
                            y_k0[:,i] = np.sort(y_k[:,0,i],0)
                            y_k1[:,i] = np.sort(y_k[:,1,i],0)
                        elif pred_out not in ('predict'  ,'predict_proba', 'predict_log_proba'):
                            raise ValueError\
                                ('No valid method. Valid methods are: predict, predict_proba')

                # yHat baseline, (w/o covariate k)
                if row != None:
                    if len(s) == 0:
                        s = np.arange(M)
                        if i != 0:
                            raise ValueError('s is empty for i not equal 0')
                        # Shuffle rows for each feature
                        for j in range(self.X.shape[1]):
                                perm_indx = np.random.randint(0,n,n)
                                V_base[:,j,i] = self.X[perm_indx,j]
                    else:
                        V_base[:,s,i] = self.X[:,s]

                    if class_prob == False or (class_prob == True and pred_out == 'predict'):
                        if len(row) == 1:
                            y_base[:,i] = self.model.fit(V_base[:,s,i],self.y)\
                                .predict(V_base[row,:,i].reshape(1,-1))
                        elif len(row) > 1:
                            y_base[:,i] = self.model.fit(V_base[:,s,i],self.y)\
                                .predict(V_base[row,:,i]).reshape(len(row))
                            y_base[:,i] = np.sort(y_base[:,i],0)
                    elif class_prob == True:
                        if pred_out == 'predict_proba' or pred_out == 'predict_log_proba':
                            if len(row) == 1:
                                y_base[:,:,i] = getattr(self.model.fit(V_base[:,s,i],self.y)\
                                    ,pred_out)(V_k[row,:,i].reshape(1,-1))
                                y_b0[:,i] = y_base[:,0,i]
                                y_b1[:,i] = y_base[:,1,i]
                        if len(row) > 1:
                            y_base[:,:,i] = getattr(self.model.fit(V_base[:,s,i],self.y)\
                                    ,pred_out)(V_k[row,:,i]).reshape(len(row),2)
                            y_b0[:,i] = np.sort(y_base[:,0,i],0)
                            y_b1[:,i] = np.sort(y_base[:,1,i],0)

                else:
                    if len(s) == 0:
                        s = np.arange(M)
                        if i != 0:
                            warnings.warn('s is empty for i not equal 0')
                        # Shuffle rows for each feature
                        for j in range(self.X.shape[1]):
                                perm_indx = np.random.randint(0,n,n)
                                V_base[:,j,i] = self.X[perm_indx,j]
                    else:
                        V_base[:,s,i] = self.X[:,s]
                        
                    if class_prob == False or (class_prob == True and pred_out == 'predict'):
                        y_base[:,i] = self.model.fit(V_base[:,:,i],self.y)\
                            .predict(V_base[:,:,i]).reshape(n)
                        y_base[:,i] = np.sort(y_base[:,i],0)
                    elif class_prob == True:
                        if pred_out == 'predict_proba' or pred_out == 'predict_log_proba':
                            y_base[:,:,i] = getattr(self.model.fit(V_base[:,s,i],self.y)\
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

                if type(show_y_for_k) == True:
                    self.y_kShow = y_k
                    self.y_baseShow = y_base

                Lor_val = ((2/(n**2))*np.mean(self.y))*(Lor_val_temp.sum(0))
                Lor_val = Lor_val.reshape((1,2**(M-1)))

                LZ[k,0] = np.dot(Lor_val,kernel)

            elif class_prob == True and (pred_out == 'predict_proba' or pred_out == 'predict_log_proba'):
                for j in range(n):
                    Lor_val_temp0[j,:] = j*(y_k0[j,:]-y_b0[j,:])
                    Lor_val_temp1[j,:] = j*(y_k1[j,:]-y_b1[j,:])

                if show_last_y == True:
                    self.y_k0Show = y_k0
                    self.y_b0Show = y_b0
                    self.y_k1Show = y_k1
                    self.y_b1Show = y_b1
                    self.testing = k

                Lor_val0 = ((2/(n**2))*y_class_mu[0])*(Lor_val_temp0.sum(0))
                Lor_val1 = ((2/(n**2))*y_class_mu[1])*(Lor_val_temp1.sum(0))
                Lor_val0 = Lor_val0.reshape((1,2**(M-1)))
                Lor_val1 = Lor_val1.reshape((1,2**(M-1)))

                LZ0[k,0] = np.dot(Lor_val0,kernel)
                LZ1[k,0] = np.dot(Lor_val1,kernel)

        if class_prob == False or (class_prob == True and pred_out == 'predict'):
            return LZ

        elif class_prob == True and (pred_out == 'predict_proba' or pred_out == 'predict_log_proba'):
            return LZ0, LZ1;
