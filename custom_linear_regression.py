class CustomLinearRegression():
    
    def __init__( self,coefs_=None, intercept_=None, MSE_seq=None ):
        
        self.coefs_     = None         # weights of your model
        self.intercept_ = None         # bias of your model
        self.MSE_seq    = None        
   
    #########################################################################################################
    def fit(self, X, Y, regularization=None, lambd=None, loss_fn="mse", batch_size=1, lr=1e-5, max_iter=1000):
        
        self.coefs_     = np.random.random( (np.shape(X)[1],1 ) )  # initializing wights vector with random values
        self.intercept_ = np.random.random((1,1) )                 # initializing bias with random value
        
        self.MSE_seq    = np.empty(max_iter)                  # array for model Error(MSE) sequence during learning process  
        
        
        if regularization==None:
                
            if loss_fn =="mse":
                if     batch_size==np.shape(X)[0]  :self.fit_MSE_GD (X, Y, lr, max_iter)
                elif   batch_size==1               :self.fit_MSE_SGD(X, Y, lr, max_iter)
                elif 1<batch_size<np.shape(X)[0]   :self.fit_MSE_minibatch(X, Y, lr, max_iter , batch_size)
                else: print('batch size is not valid!')
                    
            elif loss_fn == "mae":
                if     batch_size==np.shape(X)[0]  :self.fit_MAE_GD (X, Y, lr, max_iter)
                else: print('batch size is not valid!')
                
        elif regularization=='L1': self.fit_GD_L1(X, Y, lr, lambd, max_iter)
        elif regularization=='L2': self.fit_GD_L2(X, Y, lr, lambd, max_iter)
        
        else: print('input is not valid')
         
    #########################################################################################################
    def fit_MSE_GD(self, X, Y, lr, max_iter):    # training with Gradient Descent and MSE cost function
        W = self.coefs_
        b = self.intercept_
        m = np.shape(X)[0]                 # number of data in X matrix
        n = np.shape(X)[1]                 # number of features in X matrix
        
        ############################
        for iter in range(max_iter):
            dJ_dW = np.zeros_like(W)
            dJ_db = 0
            
            ############################ Error derivatives for all instances
            for i in range(m):
                for j in range(n):
                    dJ_dW[j] = dJ_dW[j] + ( (X[i]@W) - Y[i] ) * X[i,j]
                    
                dJ_db = dJ_db    + (  X[i]@W  - Y[i] )
                
            ############################ W,b Updating
            W = W - ( lr/m * dJ_dW)
            b = b - ( lr/m * dJ_db)
            
            self.MSE_seq [iter] = np.mean( (X@W-Y)**2 ) / 2


        self.coefs_     = W
        self.intercept_ = b
        
    #########################################################################################################
    def fit_MSE_SGD(self, X, Y, lr, max_iter):  # training with Stochastic GD and MSE error function
        W = self.coefs_
        b = self.intercept_
        m = np.shape(X)[0]               # number of data in X matrix
        n = np.shape(X)[1]               # number of features in X matrix
        
        ########################### shuffling data
        data = np.hstack((X, Y))
        np.random.shuffle(data)
        X=data[:,:-1]
        Y=data[:,-1].reshape(-1,1)
        ########################### 
        for k in range(max_iter):
            
            i = k % m
            dJi_dW = np.zeros_like(W)
            dJi_db = 0

            ############################## Error derivatives for 1 instance
            for j in range(n):
                dJi_dW[j] = dJi_dW[j] + 2*( (X[i]@W) - Y[i] ) * X[i,j]
                
            dJi_db    = dJi_db    + 2*(  X[i]@W  - Y[i] )
            
            ########################## W,b Updating    
            W = W - ( lr * dJi_dW)
            b = b - ( lr * dJi_db)
            
            self.MSE_seq [k] = np.mean( (X@W-Y)**2 ) / 2
            
            ########################### shuffling data after each epoch
            if k==m-1: 
                data = np.hstack((X, Y))
                np.random.shuffle(data)
                X=data[:,:-1]
                Y=data[:,-1].reshape(-1,1)
            ###########################
            
        self.coefs_     = W
        self.intercept_ = b

    #########################################################################################################  
    def fit_MSE_minibatch(self, X, Y, lr, max_iter, batch_size):  # training with Mini-batch GD and MSE error function
        
        W = self.coefs_
        b = self.intercept_
        m = np.shape(X)[0]               # number of data in X matrix
        n = np.shape(X)[1]               # number of features in X matrix
        
        steps = np.arange(0,m,batch_size)      # starting indexes of each batch
        nb    = math.ceil(m/batch_size)        # number of batches
        
        ################################# shuffling data
        data = np.hstack((X, Y))
        np.random.shuffle(data)
        X=data[:,:-1]
        Y=data[:,-1].reshape(-1,1)
        
        ###################################          
        for k in range(max_iter):
            
            dJ_dW = np.zeros_like(W)
            dJ_db = 0

            ############################### creating batch data 
            min_index = steps[k%nb]
            max_index =  min( steps[(k)%nb]+batch_size, m)
    
            Bx= X[ min_index: max_index][:]
            By= Y[ min_index : max_index]
            
            ################################ Error derivatives for data in batch
            for i in range(np.shape(Bx)[0]):
                for j in range(n):
                    dJ_dW[j] = dJ_dW[j] + ( (Bx[i]@W) - By[i] ) * Bx[i,j]
                dJ_db = dJ_db    + (  Bx[i]@W  - By[i] )
                
            ################################# W,b Updating
            W = W - ( lr/batch_size * dJ_dW)
            b = b - ( lr/batch_size * dJ_db)
            
            self.MSE_seq [k] = np.mean( (X@W-Y)**2 ) / 2
            
            ################################# shuffling data after each epoch
            if (k+1)%nb==0:
                data = np.hstack((X, Y))
                np.random.shuffle(data)
                X=data[:,:-1]
                Y=data[:,-1].reshape(-1,1)
                
            ################################
            
        self.coefs_     = W
        self.intercept_ = b 
 
    ####################################################################################################
    def fit_lsqr(self, X, Y):  # training with least square method - it's not iterative 
        

        m = np.shape(X)[0]   # number of features in X matrix
        n = np.shape(X)[1]   # number of features in X matrix
        
        X = np.hstack( ( np.ones([m,1]), X) )  # adding a 1-column to X as bias factor
        Weights = np.empty([n+1,1])
        
        ################## calculating the best weights by least-squares method formula
        X_t = X.transpose()
        Weights = ( np.linalg.inv( X_t @ X ) @ X_t @ Y )  
        
        ################## Updating weghts and bias(separating bias and wights from "Weights" vector)
        self.coefs_ = Weights[1:]
        self.intercept_ = Weights[0]

    ####################################################################################################
# =============================================================================
#     def fit_MAE_GD(self, X, Y, lr, max_iter):
#         
#         W = self.coefs_
#         b = self.intercept_
#         m = np.shape(X)[0]               # number of data in X matrix
#         n = np.shape(X)[1]               # number of features in X matrix
#         Error = np.empty([1,max_iter])   # creating an array for Model Error after each update
#         lm_n = 0                         # number of encountered local minima(s)
#         
#         for iter in range(max_iter):
#             dJ_dW = np.zeros_like(W)
#             dJ_db = 0
#             sign = np.sign(X@W - Y)
#             ######################### Error derivatives
#             for i in range(m):
#                 
#                 for j in range(n):
#                     dJ_dW[j] = dJ_dW[j] + sign[i] * X[i,j]
#                         
#                 dJ_db = dJ_db + sign[i]
#                 
#             ######################### W,b Updating
#             W = W - ( lr/m * dJ_dW)
#             b = b - ( lr/m * dJ_db)
#             
#             ######################### checking the global minima
#             if X@W == Y : break
#         
#             ######################### checking local minima
#             if dJ_dW==np.zeros_like(W) and dJ_db==0:
#                 lm_n = lm_n + 1 
#                 # W = W + ?
#                 # b = b + ?

#             #########################    
#         self.coefs_     = W
#         self.intercept_ = b
#         self.MSE_seq    = Error 
# =============================================================================

    ####################################################################################################
    
    def fit_GD_L1(self, X, Y, lr, lambd, max_iter) : # training by GD and L1 method (Lasso)
        W = self.coefs_
        b = self.intercept_
        m = np.shape(X)[0]               # number of data in X matrix
        n = np.shape(X)[1]               # number of features in X matrix
        
        for iter in range(max_iter):
            dJ_dW = np.zeros_like(W)
            dJ_db = 0
            
            ####################### Error derivatives for all instances
            for i in range(m):
                for j in range(n):
                    dJ_dW[j] = dJ_dW[j] + ( (X[i]@W) - Y[i] ) * X[i,j] + lambd * np.sign(W[j])
                    
                dJ_db = dJ_db    + (  X[i]@W  - Y[i] )
                
            ######################## W,b Updating
            W = W - ( lr/m * dJ_dW)
            b = b - ( lr/m * dJ_db)
            
            self.MSE_seq [iter] = np.mean( (X@W-Y)**2 ) / 2
            
            ########################
            
        self.coefs_     = W
        self.intercept_ = b
        
    ####################################################################################################
    def fit_GD_L2(self, X, Y, lr, lambd, max_iter) : # training by GD and L2 method (Ridge)
        W = self.coefs_
        b = self.intercept_
        m = np.shape(X)[0]               # number of data in X matrix
        n = np.shape(X)[1]               # number of features in X matrix
        
        for iter in range(max_iter):
            dJ_dW = np.zeros_like(W)
            dJ_db = 0
            
            ################## Error derivatives for all instances
            for i in range(m):
                for j in range(n):
                    dJ_dW[j] = dJ_dW[j] + ( (X[i]@W) - Y[i] ) * X[i,j] + lambd*W[j]
                    
                dJ_db = dJ_db    + (  X[i]@W  - Y[i] )
                
            ###################### W,b Updating
            W = W - ( lr/m * dJ_dW)
            b = b - ( lr/m * dJ_db)
            
            self.MSE_seq [iter] = np.mean( (X@W-Y)**2 ) / 2
            ########################

        self.coefs_     = W
        self.intercept_ = b
    
    ####################################################################################################
    ####################################################################################################
    def predict(self, X):     # it returns the pridiction of the current model 
        return (X @ self.coefs_ + self.intercept_)


    ####################################################################################################
    def score(self, X, Y):     # it returns the R-squared score for a trained model on input X and targets Y
        
        Y_m = np.mean(Y)                 # mean of observed data
        Y_p = self.predict(X)            # model prediction 
        
        SS_res = np.sum( (Y - Y_p)**2 )  # regression sum of squares
        SS_tot = np.sum( (Y - Y_m)**2 )  # total sum of squares
        
        return (1 - (SS_res/SS_tot))
    
     ####################################################################################################
    def Model_MSE(self, X, Y):   # it returns Mean Square Error using predicted and desired output
        return np.mean((self.predict(X)-Y)**2)/2
