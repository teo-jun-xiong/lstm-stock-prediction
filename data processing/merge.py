import os
import glob
import pandas as pd
import numpy as np
import math
from datetime import date

combined_fundamentals_files = os.listdir('./combined-fund')

for file in os.listdir('./technicals-processed'):
    if file in combined_fundamentals_files:
        print(file)
        fundamental_data = pd.read_csv('./combined-fund/' + file)
        technical_data = pd.read_csv('./technicals-processed/' + file)
        
        
        cols = technical_data.columns[0:].tolist() + fundamental_data.columns[1:].tolist()
        df = pd.DataFrame(columns=cols)
        
#         print(df)
        for i in range(len(technical_data)):
            tech_date = technical_data.iloc[i][0]
#             print(tech_date)

            indices = fundamental_data.date[fundamental_data.date >= tech_date].index.tolist()
#             print(fundamental_data.iloc[index].tolist())
#             print(indices)
            
            if len(indices) > 0:
                index = indices[-1]
                
                if index != -1:
                    fun_date = fundamental_data.iloc[index][0]
                    if str(date(int(fun_date[0:4]), int(fun_date[5:7]), int(fun_date[8:])) - date(int(tech_date[0:4]), int(tech_date[5:7]), int(tech_date[8:]))) > '92':
                        break
                                                                                                           
                    combined_row = technical_data.iloc[i].tolist() + fundamental_data.iloc[index].tolist()[1:]
                    df.loc[len(df)+1] = combined_row
    
            
        df.to_csv('./combined-all/' + file, index = False)
        print('Done writing to ' + file)