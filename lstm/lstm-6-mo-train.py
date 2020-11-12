from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request
import json
import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.feature_selection import SelectKBest, chi2, f_regression
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from math import sqrt
import itertools
from keras import backend as K 

'''
MODEL TRAINING
'''

# Disable Tensorflow GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Import train data
train_data = []
companies = []
for file in os.listdir('./data/6mo_train'):
    df = pd.read_csv('./data/6mo_train/' + file)
    train_data.append(df)
    companies.append(file)


# Permutate hyperparameters
epoch_set = [1,2,3,4] 
batch_size = [8,16,32,64]
dropouts = [0.1,0.2,0.3,0.4]
temp = list(itertools.product(epoch_set, batch_size, dropouts))
hyperparam = []

# Append extra element to store error for corresponding set of hyperparameter
for perm in temp:
    hyperparam.append([perm[0],perm[1], perm[2], 0])
    
# Find size of each fold in 10-fold cross validation
k = math.floor(len(train_data)/10)
validation_start_index = list(range(0, len(train_data), k))

# Store hyperparameters and error for future use
f = open('log.txt','w')


for idx in range(len(validation_start_index)):
    # train n - k companies
    start_index = validation_start_index[idx]

    if idx == len(validation_start_index) - 1: 
        end_index = 297
    else:
        end_index = start_index + k

    for h in hyperparam:
        # Prevents system from slowing down 
        K.clear_session()

        # Create a new model using the current set of hyperparameters
        regressor = Sequential()

        regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (180,84)))
        regressor.add(Dropout(h[2]))

        regressor.add(LSTM(units = 50, return_sequences = True))
        regressor.add(Dropout(h[2]))

        regressor.add(LSTM(units = 50, return_sequences = True))
        regressor.add(Dropout(h[2]))

        regressor.add(LSTM(units = 50))
        regressor.add(Dropout(h[2]))

        regressor.add(Dense(units = 1))

        regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

        # Train the N-k companies
        for i in range(len(train_data)):
            if i in range(start_index, end_index):
                continue
            print("=============== Training company " + companies[i] + "===============")
            df = train_data[i]

            # Initialize MinMaxScaler
            sc_X = MinMaxScaler(feature_range=(0, 1))
            sc_y = MinMaxScaler(feature_range=(0, 1))

            company_train_data_X = []
            company_train_data_y = []

            # Normalize all columns
            company_train_data_X.append(sc_X.fit_transform(df.iloc[:, 1:]))
            company_train_data_y.append(sc_y.fit_transform(df.values[:, 0].reshape(-1, 1))) 
            
            company_train_data_X = company_train_data_X[0]
            company_train_data_y = company_train_data_y[0]

            # Obtain indices of blocks of non-overlapping rows of 180
            block_indices = list(range(180, len(company_train_data_X), 180))

            X_train = []
            y_train = []

            # For each closing 6-month price (y), create a 3D vector of 180 rows, and 84 columns
            for j in block_indices:
                X_train.append(company_train_data_X[j - 180: j])
                y_train.append(company_train_data_y[j])
            
            # Convert list to a numpy array
            X_train, y_train = np.array(X_train), np.array(y_train)

            # Reshape array before feeding to the model
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))

            regressor.fit(X_train, y_train, epochs = h[0], batch_size = h[1])

            # Reset state of model between each training
            regressor.reset_states()
        
        # Validate K companies
        rmse = 0
        for i in range(start_index, end_index):
            df = train_data[i]

            # Initialize MinMaxScaler
            sc_X = MinMaxScaler(feature_range=(0, 1))
            sc_y = MinMaxScaler(feature_range=(0, 1))

            company_validation_data_X = []
            company_validation_data_y = []

            # Normalize all columns
            company_validation_data_X.append(sc_X.fit_transform(df.values[:, 1:]))
            company_validation_data_y.append(sc_y.fit_transform(df.values[:, 0].reshape(-1, 1))) 

            company_validation_data_X = company_validation_data_X[0]
            company_validation_data_y = company_validation_data_y[0]

            # Obtain indices of blocks of non-overlapping rows of 180
            block_indices = list(range(180, len(company_validation_data_X), 180))

            X_validation = []
            real_stock_price = []

            # For each closing 6-month price (y), create a 3D vector of 180 rows, and 84 columns
            for j in block_indices:
                X_validation.append(company_validation_data_X[j - 180: j])
                real_stock_price.append(company_validation_data_y[j]) 
                            
            # Convert list to a numpy array
            X_validation, real_stock_price = np.array(X_validation), np.array(real_stock_price)

            # Reshape array before feeding to the model
            X_validation = np.reshape(X_validation, (X_validation.shape[0], X_validation.shape[1], X_validation.shape[2]))

            predicted_stock_price = regressor.predict(X_validation)

            # Append error to this hyperparameter
            h[3] += sqrt(mean_squared_error(real_stock_price, predicted_stock_price))

            # Plot graph
            predicted_stock_price = sc_y.inverse_transform(predicted_stock_price)
            plt.plot(real_stock_price, color = 'black', label = 'Real Stock Price')
            plt.plot(predicted_stock_price, color = 'green', label = 'Predicted Stock Price')
            plt.title('Stock Price Prediction')
            plt.xlabel('Time')
            plt.ylabel('Stock Price')
            plt.legend()
            plt.close()
            regressor.reset_states() 

f.close()
f = open('hyperparameters.txt','w')
f.writelines(str(hyperparam))
f.close()

# Write the average error for each hyperparamter (N=297)
for perm in hyperparam:
    perm[2] = perm[2]/297

f = open('hyperparameters_average.txt','w')
f.writelines(str(hyperparam))
f.close()
