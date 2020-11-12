import pandas as pd
import os
import numpy as np

for file in os.listdir('./data/1mo_train'):
    df = pd.read_csv('./data/1mo_train/'+ file)
    df = df.drop(columns=['date'], axis=1)
    df.to_csv('./data/1mo_train/' + file, index=False)
    
    df = pd.read_csv('./data/1mo_test/'+ file)
    df = df.drop(columns=['date'], axis=1)
    df.to_csv('./data/1mo_test/' + file, index=False)
    
for file in os.listdir('./data/6mo_train'):
    df = pd.read_csv('./data/6mo_train/'+ file)
    df = df.drop(columns=['date'], axis=1)
    df.to_csv('./data/6mo_train/' + file, index=False)
    
    df = pd.read_csv('./data/6mo_test/'+ file)
    df = df.drop(columns=['date'], axis=1)
    df.to_csv('./data/6mo_test/' + file, index=False)
