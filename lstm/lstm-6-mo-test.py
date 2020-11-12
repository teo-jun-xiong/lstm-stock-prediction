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
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from math import sqrt
import itertools
import copy

# Disable Tensorflow from using GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Import training data 
train_data = []
companies = []
for file in os.listdir('./data/6mo_train/'):
    df = pd.read_csv('./data/6mo_train/' + file)
    companies.append(file)
    train_data.append(df)

# Initializing model using best hyperparameters obtained from 10-fold cross validation
# Number of epochs: 1
# Batch size: 32
# Dropout rate: 0.3
regressor = Sequential()

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (180,84)))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.3))

regressor.add(Dense(units = 1))

# Adam optimizer
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Train companies
for i in range(len(train_data)):
    print("=============== Training company " + companies[i] + "===============")
    df = train_data[i]

    # Initialize MinMaxScaler 
    sc_X = MinMaxScaler(feature_range=(0, 1))
    sc_y = MinMaxScaler(feature_range=(0, 1))

    company_test_data_X = []
    company_test_data_y = []

    # Normalize all columns using MinMaxScaler
    company_test_data_X.append(sc_X.fit_transform(df.iloc[:, 1:]))
    company_test_data_y.append(sc_y.fit_transform(df.values[:, 0].reshape(-1, 1)))
    
    company_test_data_X = company_test_data_X[0]
    company_test_data_y = company_test_data_y[0]

    # Obtain indices of blocks of non-overlapping rows of 180
    block_indices = list(range(180, len(company_test_data_X), 180))

    X_train = []
    y_train = []

    # For each closing 6-month price (y), create a 3D vector of 180 rows, and 84 columns
    for j in block_indices:
        X_train.append(company_test_data_X[j - 180: j])
        y_train.append(company_test_data_y[j])     
    
    # Convert list to a numpy array
    X_train, y_train = np.array(X_train), np.array(y_train)

    # Reshape array before feeding to the model
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))

    regressor.fit(X_train, y_train, epochs = 1, batch_size =32)

    # Reset state of model between each training
    regressor.reset_states()

# Save model for future use
regressor.save("model")

# Load saved model
regressor_model = load_model("model")
print(regressor_model.summary())
companies = os.listdir('./data/6mo_train')

# Import test data
test_data = []
for file in companies:
    df = pd.read_csv('./data/6mo_test/' + file)
    test_data.append(df)

# Store total RMSE error
total_e = 0.0

# Test companies
for i in range(len(test_data)):
    df = test_data[i]

    # Initialize MinMaxScaler
    sc_X = MinMaxScaler(feature_range=(0, 1))
    sc_y = MinMaxScaler(feature_range=(0, 1))

    company_test_data_X = []
    company_test_data_y = []

    # Normalize all columns using MinMaxScaler
    company_test_data_X.append(sc_X.fit_transform(df.values[:, 1:]))
    company_test_data_y.append(sc_y.fit_transform(df.values[:, 0].reshape(-1, 1))) # is reshape in right place

    company_test_data_X = company_test_data_X[0]
    company_test_data_y = company_test_data_y[0]

    # Obtain indices of blocks of non-overlapping rows of 180
    block_indices = list(range(180, len(company_test_data_X), 180))

    X_test = []
    real_stock_price = []

    # For each closing 6-month price (y), create a 3D vector of 180 rows, and 84 columns
    for j in block_indices:
        X_test.append(company_test_data_X[j - 180: j])
        real_stock_price.append(company_test_data_y[j]) 

    real_real = copy.deepcopy(real_stock_price)  

    # Convert list to numpy array
    X_test, real_stock_price = np.array(X_test), np.array(real_stock_price)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

    # Reshape array before feeding to the model
    predicted_stock_price = regressor_model.predict(X_test)

    # Inverse transform the normalized columns before plotting and calculating RMSE
    real_real = sc_y.inverse_transform(real_real)
    predicted_stock_price = sc_y.inverse_transform(predicted_stock_price)

    # Increment total error
    e = float(1.0*sqrt(mean_squared_error(real_real, predicted_stock_price)))
    total_e += e

    # Plot graph and save it as an image
    plt.plot(real_real, color = 'black', label = 'Real Stock Price')
    plt.plot(predicted_stock_price, color = 'green', label = 'Predicted Stock Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()

    name = companies[i]
    name = name.replace('.csv', '')

    plt.savefig('./plots/6-months/'+name)
    plt.close()
    regressor_model.reset_states() 

# Prints average RMSE
print(total_e/len(test_data))
