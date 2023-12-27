# -*- encoding:utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

path='CarSeats.csv'
orignal_data = pd.read_csv(path)
data=orignal_data[['Sales','CompPrice','Income','Advertising','Population','Price',"Age",'Education']]
Y=data[['Sales']]
X=data[['CompPrice','Income','Advertising','Population','Price',"Age",'Education']]

scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)
CompPrice=X_scaled[:,0]
Income=X_scaled[:,1]
Advertising=X_scaled[:,2]
Population=X_scaled[:,3]
Price=X_scaled[:,4]
Age=X_scaled[:,5]
Education=X_scaled[:,6]

print('mean of CompPrice is:{} , variance of CompPrice is {}'.format(np.mean(CompPrice),np.var(CompPrice)))
print('mean of Income is:{} , variance of Income is {}'.format(np.mean(Income),np.var(Income)))
print('mean of Advertising is:{} , variance of Advertising is {}'.format(np.mean(Advertising),np.var(Advertising)))
print('mean of Population is:{} , variance of Population is {}'.format(np.mean(Population),np.var(Population)))
print('mean of Price is:{} , variance of Price is {}'.format(np.mean(Price),np.var(Price)))
print('mean of Age is:{} , variance of Age is {}'.format(np.mean(Age),np.var(Age)))
print('mean of Education is:{} , variance of Education is {}'.format(np.mean(Education),np.var(Education)))
X_scaled[:,0]-=np.mean(CompPrice)
X_scaled[:,1]-=np.mean(Income)
X_scaled[:,2]-=np.mean(Advertising)
X_scaled[:,3]-=np.mean(Population)
X_scaled[:,4]-=np.mean(Price)
X_scaled[:,5]-=np.mean(Age)
X_scaled[:,6]-=np.mean(Education)

new_data = np.concatenate((Y, X_scaled), axis=1)
train_data=new_data[:len(new_data)//2,:]
test_data=new_data[len(new_data)//2:,:]
Y_train=train_data[:,0]
X_train=train_data[:,1:]
Y_test=test_data[:,0]
X_test=test_data[:,1:]
print("The first row of Y_train: {}, and the first row of X_train{}".format(Y_train[0],X_train[0]))
print("The first row of Y_test: {}, and the first row of X_test{}".format(Y_test[0],X_test[0]))