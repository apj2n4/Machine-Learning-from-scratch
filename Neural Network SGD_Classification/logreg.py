import numpy as np

class logregmodel():
    def __init__(self):
        self.X=0
        self.y = 0
        
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    def scaler_fit(self,X):
        self.X_max = X.max(0)
        self.X_min = X.min(0)
            
    def scaler_transform(self,X):
        return (X-self.X_min)/(self.X_max-self.X_min)
    
    def fit(self,X,y,alpha=1e-3,epoch=200):
        '''
        The fit function implements gradient descent algorithm in fitting the coefficients of
        the logistic regression model. The training set X, the training labels y in the form of 0 and 1,
        the learning rate and number of epochs (iterations) are needed as input.
        '''
        #self.scaler = MinMaxScaler().fit(X)
        self.scaler_fit(X)
        X_scaled = self.scaler_transform(X)
        self.X = np.insert(X_scaled,0,1,axis=1)
        self.y = y
        m = (self.X).shape[0]
        n = (self.X).shape[1]
        self.m = m
        self.n  = n
        theta = np.ones(shape=(n,1))
        theta_t = np.transpose(theta)
        X_t = np.transpose(self.X)
        y_t =self.y.reshape(1,-1)
        cost_list = []
        for i in range(1,epoch):
            y_pred = self.sigmoid(np.dot(theta_t,X_t))
            error = y_pred - y_t
            ### binary cross entropy cost function
            cost = -np.sum(y_t*np.log(y_pred)+(1-y_t)*np.log(1-y_pred))/m
            cost_list.append(cost)
            theta_t = theta_t - alpha*np.dot(error,self.X)/m
        self.coefficients_ = theta_t
        self.cost = cost_list
        self.y_pred = self.sigmoid(np.dot(theta_t,X_t))
        
    def predict(self,X_new):
        #X_new = self.scaler.transform(X_new)
        X_new = self.scaler_transform(X_new)
        X_new = np.insert(X_new,0,1,axis=1)
        m = X_new.shape[0]
        X_t = np.transpose(X_new)        
        y_class = np.zeros(m)
        y_pred = self.sigmoid(np.dot(self.coefficients_,X_t))
        y_pred = y_pred.reshape(-1)
        y_class[y_pred>0.5]=1
        return y_pred,y_class
    
    def confusion_matrix(self,X_new,y_new):
        y_pred,y_class = self.predict(X_new)
        y_mat = np.append(y_new.reshape(-1,1),y_class.reshape(-1,1),axis=1)
        TP = np.sum((y_mat[:,0]==0) & (y_mat[:,1]==0))
        FN = np.sum((y_mat[:,0]==0) & (y_mat[:,1]==1))
        FP = np.sum((y_mat[:,0]==1) & (y_mat[:,1]==0))
        TN = np.sum((y_mat[:,0]==1) & (y_mat[:,1]==1))
        cf = np.zeros(shape=(2,2))
        cf[0,0] = TP
        cf[0,1] = FN
        cf[1,0] = FP
        cf[1,1] = TN
        return cf
        
    def scores(self,X_new,y_new):
        y_pred,y_class = self.predict(X_new)
        accuracy = np.sum(y_class==y_new)/y_new.shape[0]
        return accuracy 