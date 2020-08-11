import numpy as np
import pandas as pd
from nn_sgd import dense_NN_sgd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve,roc_auc_score

df = pd.read_csv("loan_simplified_3.csv")
df = df.dropna(axis=0)
df['loan_status_new']=0
df.loc[df['loan_status']=='Default','loan_status_new']=1
df = pd.concat([df,pd.get_dummies(df['term'])],axis=1)
df = pd.concat([df,pd.get_dummies(df['grade'])],axis=1)
df = pd.concat([df,pd.get_dummies(df['emp_length'])],axis=1)
df = pd.concat([df,pd.get_dummies(df['home_ownership'])],axis=1)
df = pd.concat([df,pd.get_dummies(df['verification_status'])],axis=1)
df = pd.concat([df,pd.get_dummies(df['purpose'])],axis=1)
df = df.drop(columns=['term','grade','emp_length','home_ownership','verification_status','purpose'],axis=1)
df = df.drop(columns=['loan_status'],axis=1)

X = df.drop(columns=['loan_status_new'],axis=1).values
y = df['loan_status_new'].values

nn4 = dense_NN_sgd()
nn4.load_model(name="dnn_SGD")
y_prob_nn4,y_class_nn4 = nn4.predict(X)

nn4 = dense_NN_sgd()
nn4.load_model(name="dnn_SGD")
y_prob_nn4,y_class_nn4 = nn4.predict(X)

cf_nn4 = confusion_matrix(y,y_class_nn4,normalize="true")
print(cf_nn4)