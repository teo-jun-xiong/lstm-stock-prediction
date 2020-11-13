import os
import glob
import pandas as pd
import numpy as np
import math
from datetime import date
from datetime import datetime 
from dateutil.relativedelta import relativedelta

# os.chdir('./combined-all') #change to your own directory

def main():
    for file in os.listdir('./combined-check-ok'):
        print(file)
        mydict = {}
        all_data = pd.read_csv('./combined-check-ok/' + file)

        cols = all_data.columns[0:].tolist() + ['closePlusOneMonth'] + ['closePlusSixMonth'] 
        df = pd.DataFrame(columns=cols)

        for i in range(len(all_data)):
            date = all_data.iloc[i][0]
            close = all_data.iloc[i][1]
            mydict[date] = close 
            date_object = datetime.strptime(date, '%Y-%m-%d')

            date_plus_six_months = date_object + relativedelta(months=+6) 
            date_plus_six_months_string = date_plus_six_months.strftime('%Y-%m-%d')
        
            date_plus_one_month = date_object + relativedelta(months=+1) 
            date_plus_one_month_string = date_plus_one_month.strftime('%Y-%m-%d')
            
            combined_row = all_data.iloc[i].tolist() 

            #find +1 month
            if not date_plus_one_month_string in mydict.keys():
                date_plus_one_month_string = findNextDate(date_plus_one_month, mydict)
                if date_plus_one_month_string == None:
                    combined_row.append(0) #add value for one month
                else:
                    close_plus_one_month = mydict[date_plus_one_month_string]
                    combined_row.append(close_plus_one_month) #add value for one month
            else:
                close_plus_one_month = mydict[date_plus_one_month_string]
                combined_row.append(close_plus_one_month) #add value for one month
            
            #find +6 months
            if not date_plus_six_months_string in mydict.keys():
                date_plus_six_months_string = findNextDate(date_plus_six_months, mydict)
                if date_plus_six_months_string == None:
                    combined_row.append(0) #add value for six month
                else:
                    close_plus_six_months = mydict[date_plus_six_months_string]
                    combined_row.append(close_plus_six_months) #add value for six month
            else:
                close_plus_six_months = mydict[date_plus_six_months_string]
                combined_row.append(close_plus_six_months) #add value for six month
            

            df.loc[len(df)+1] = combined_row
        df.to_csv('./combined-check-y/' + file, index = False) 
        

#find the next date up to 3 days
def findNextDate(date, mydict):
    count = 0
    while True:
        count += 1
        date = date + relativedelta(days=+1) 
        dateString = date.strftime('%Y-%m-%d')
        if dateString in mydict.keys():
            return dateString
        if count == 3: 
            return None

if __name__ == "__main__":
    main()